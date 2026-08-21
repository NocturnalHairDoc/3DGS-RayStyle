from __future__ import annotations

import math
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from .atlas import AtlasTopology
from .backend import LegacyGaussianBackend
from .config import ExperimentConfig, load_config
from .environments import EnvironmentMap, EnvironmentPool
from .io_utils import save_image
from .method_comparison import load_method_comparison
from .multisegment import composite_independent_edits
from .multisegment_bundle import load_segment_bundle
from .project_state import load_segment
from .style_state import StyleState


def chart_id_colors(chart_ids: torch.Tensor) -> torch.Tensor:
    """Stable high-contrast colors for per-Gaussian chart inspection."""
    ids = chart_ids.float()[:, None]
    return 0.5 + 0.5 * torch.sin(
        ids * ids.new_tensor([[2.17, 3.71, 5.13]])
        + ids.new_tensor([[0.0, 2.1, 4.2]]),
    )


def chart_boundary_colors(field, base_albedo: torch.Tensor) -> torch.Tensor:
    """Dim atlas albedo with seam-adjacent Gaussians highlighted in red."""
    colors = base_albedo.detach().clone() * 0.3
    if field.seam_edges.numel():
        boundary = torch.unique(field.seam_edges.flatten())
        colors[boundary] = colors.new_tensor([1.0, 0.05, 0.02])
    return colors


def uv_collision_debug_colors(field) -> torch.Tensor:
    """Blue normal points, yellow candidates, red unresolved UV collisions."""
    colors = torch.zeros(
        len(field.chart_ids), 3, device=field.chart_ids.device, dtype=torch.float32,
    )
    colors[:] = colors.new_tensor([0.03, 0.12, 0.28])
    if not field.collision_pairs.numel():
        return colors
    pairs = field.collision_pairs
    candidates = torch.unique(pairs.flatten())
    colors[candidates] = colors.new_tensor([1.0, 0.72, 0.05])
    margin = 2.5 / max(int(field.resolution), 16)
    uv = field.current_local_uv()
    active = (uv[pairs[:, 0]] - uv[pairs[:, 1]]).norm(dim=1) < margin
    if torch.any(active):
        colliding = torch.unique(pairs[active].flatten())
        colors[colliding] = colors.new_tensor([1.0, 0.02, 0.02])
    return colors


@dataclass
class OrbitCamera:
    width: int
    height: int
    radius: float = 2.0
    fovy_degrees: float = 60.0

    def __post_init__(self):
        self.center = np.zeros(3, dtype=np.float32)
        self.rotation = Rotation.identity()

    @property
    def pose(self):
        result = np.eye(4, dtype=np.float32)
        result[2, 3] -= self.radius
        rotation = np.eye(4, dtype=np.float32)
        rotation[:3, :3] = self.rotation.as_matrix()
        result = rotation @ result
        result[:3, 3] -= self.center
        result[:3, 3] = -rotation[:3, :3].T @ result[:3, 3]
        return result

    def orbit(self, dx: float, dy: float):
        up = self.rotation.as_matrix()[:, 1]
        side = self.rotation.as_matrix()[:, 0]
        self.rotation = (
            Rotation.from_rotvec(up * np.radians(0.3 * dx))
            * Rotation.from_rotvec(side * np.radians(0.3 * dy))
            * self.rotation
        )

    def pan(self, dx: float, dy: float):
        delta = self.rotation.as_matrix() @ np.array([dx, -dy, 0.0])
        self.center += (0.001 * max(self.radius, 0.1) * delta).astype(np.float32)

    def zoom(self, delta: float):
        self.radius = max(0.05, self.radius * math.exp(-0.08 * float(delta)))

    def reset(self):
        self.center[:] = 0
        self.rotation = Rotation.identity()
        self.radius = 2.0


@dataclass(frozen=True)
class ViewerLayout:
    render_width: int
    render_height: int
    control_width: int = 440
    minimum_height: int = 1030
    panel_padding: int = 16
    panel_gap: int = 10
    render_header_height: int = 58

    @property
    def panel_height(self) -> int:
        return max(self.minimum_height, self.render_height + self.render_header_height + 48)

    @property
    def main_width(self) -> int:
        return self.render_width + 2 * self.panel_padding

    @property
    def viewport_width(self) -> int:
        return self.main_width + self.panel_gap + self.control_width

    @property
    def render_top_spacer(self) -> int:
        free = self.panel_height - self.render_header_height - self.render_height
        return max(12, free // 2)


class RayStyleViewerScene:
    """Checkpoint-backed renderer kept independent from DearPyGui."""

    def __init__(self, config: ExperimentConfig, checkpoint_path: str):
        if not torch.cuda.is_available():
            raise RuntimeError("RayStyle Viewer requires an NVIDIA CUDA GPU")
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"RayStyle checkpoint not found: {checkpoint}")

        self.config = config
        self.checkpoint_path = checkpoint
        self.backend = LegacyGaussianBackend(
            config.legacy_root, config.model_path, config.source_path,
            config.images, config.resolution, config.white_background,
        )
        self.selected, self.state, self.iteration = self._load_checkpoint_state(
            config, checkpoint,
        )
        self.segment_names = ("Segment 1",)
        self.states = {"Segment 1": self.state}
        self.selections = {"Segment 1": self.selected}
        self.segment_configs = {"Segment 1": config}
        self.comparison_names = ()
        self._initialize_scene(config)

    def _load_checkpoint_state(self, config: ExperimentConfig, checkpoint: Path):
        selected, _ = load_segment(
            config.project_state, config.segment_id, self.backend.point_count,
        )
        selected = selected.cuda()
        payload = torch.load(checkpoint, map_location="cuda", weights_only=False)
        metadata = payload.get("state_metadata", {})
        atlas_topology = AtlasTopology.from_checkpoint_state(payload["state_dict"])
        saved_method = metadata.get("method", config.method)
        if saved_method != config.method:
            raise ValueError(
                f"checkpoint method is {saved_method!r}, config method is {config.method!r}"
            )
        texture_resolution = metadata.get("texture_resolution") or config.train.texture_resolution
        state = StyleState(
            self.backend.base_albedo, selected, config.method,
            original_sh_degree=int(self.backend.gaussians.active_sh_degree),
            residual_degree=int(metadata.get("residual_degree", config.train.sh_degree)),
            residual_limit=float(metadata.get("residual_limit", config.train.residual_limit)),
            global_shift_limit=float(metadata.get(
                "global_shift_limit", config.train.global_shift_limit,
            )),
            detail_residual_limit=float(metadata.get(
                "detail_residual_limit", config.train.detail_residual_limit,
            )),
            selected_xyz=(self.backend.xyz[selected] if config.method == "ours" else None),
            selected_normals=(
                self.backend.canonical_normals(torch.where(selected)[0])
                if config.method == "ours" else None
            ),
            selected_visibility=(
                self.backend.gaussians.get_opacity.detach()[selected]
                if config.method == "ours" else None
            ),
            texture_resolution=int(texture_resolution),
            texture_logit_limit=config.train.texture_logit_limit,
            texture_mapping=str(metadata.get("texture_mapping", "planar")),
            atlas_charts=int(metadata.get("atlas_chart_count", config.train.atlas_charts)),
            atlas_neighbours=int(metadata.get("atlas_neighbours", config.train.atlas_neighbours)),
            atlas_padding=int(metadata.get("atlas_padding", config.train.atlas_padding)),
            atlas_feather=float(metadata.get("atlas_feather", config.train.atlas_feather)),
            atlas_uv_offset_limit=float(metadata.get(
                "atlas_uv_offset_limit", config.train.atlas_uv_offset_limit,
            )),
            atlas_topology=atlas_topology,
            albedo_mode=str(metadata.get("albedo_mode", "additive")),
            pbr_diffuse_white=float(metadata.get("pbr_diffuse_white", 0.0)),
            pbr_exposure=float(metadata.get("pbr_exposure", 0.0)),
            pbr_white_point=float(metadata.get("pbr_white_point", 0.0)),
        ).cuda()
        state.load_checkpoint_state(payload["state_dict"])
        state.eval()
        return selected, state, int(payload.get("iteration", 0))

    def _initialize_scene(self, config: ExperimentConfig):
        self.environments = EnvironmentPool(config.environment_dir, config.train.seed)
        self.environment_choices: dict[str, EnvironmentMap] = {
            "Neutral": self.environments.neutral,
            f"Train fixed · {self.environments.fixed.name}": self.environments.fixed,
        }
        for index, environment in enumerate(self.environments.train):
            self.environment_choices[f"Train {index + 1} · {environment.name}"] = environment
        for index, environment in enumerate(self.environments.heldout):
            self.environment_choices[f"Unseen {index + 1} · {environment.name}"] = environment

        # Evaluation cameras first: they make it easy to reproduce saved results.
        self.cameras = [*self.backend.test_cameras, *self.backend.train_cameras]
        test_count = len(self.backend.test_cameras)
        self.camera_labels = [
            (f"Test {index + 1:03d}" if index < test_count
             else f"Train {index - test_count + 1:03d}")
            for index in range(len(self.cameras))
        ]

    @property
    def segment_count(self) -> int:
        return len(self.segment_names)

    @property
    def total_selected_count(self) -> int:
        return sum(state.selected_count for state in self.states.values())

    @property
    def comparison_count(self) -> int:
        return len(self.comparison_names)

    @property
    def active_comparison_name(self) -> str:
        return ""

    def set_active_comparison(self, name: str) -> None:
        raise KeyError(f"this Viewer has no method comparison named {name!r}")

    def set_active_segment(self, name: str) -> None:
        if name not in self.states:
            raise KeyError(f"unknown Viewer segment: {name}")
        self.state = self.states[name]
        self.selected = self.selections[name]
        self.config = self.segment_configs[name]

    def environment(self, label: str, yaw_degrees: float, exposure: float):
        source = self.environment_choices[label]
        return EnvironmentMap(
            source.name, source.pixels,
            yaw=float(yaw_degrees) / 360.0,
            exposure=float(exposure),
        )

    @torch.inference_mode()
    def render(self, camera, mode: str, environment: EnvironmentMap):
        original = None
        overlay_modes = {
            "Atlas albedo", "UV albedo", "Chart boundaries", "Chart IDs",
            "UV collision debug",
        }
        if mode in {"Original", "Split original | styled", *overlay_modes}:
            original = self.backend.render_original(camera)
        if mode == "Original":
            return original
        if mode == "Segment mask":
            mask = self.backend.segment_mask(camera, self.state.selected_ids)
            return torch.cat((mask, mask * 0.55, torch.zeros_like(mask)), dim=0)
        if mode in {"Atlas albedo", "UV albedo"}:
            mask = self.backend.segment_mask(camera, self.state.selected_ids)
            divisor = mask.permute(1, 2, 0)
            albedo = self.backend._sparse_map(
                camera, self.state.selected_ids, self.state.selected_albedo(), divisor,
            ).clamp(0, 1).permute(2, 0, 1)
            return original * (1 - mask) + albedo * mask
        if mode in {"Atlas texture", "Atlas texture preview"}:
            texture = self.state.texture_preview()
            if texture is None:
                return self.backend.render_original(camera)
            return F.interpolate(
                texture.unsqueeze(0), (int(camera.image_height), int(camera.image_width)),
                mode="bilinear", align_corners=False,
            )[0]
        if mode in {"Chart IDs", "Chart boundaries", "UV collision debug"}:
            field = self.state.texture_field
            if field is None or not hasattr(field, "chart_ids"):
                mask = self.backend.segment_mask(camera, self.state.selected_ids)
                return original * (1 - mask) + torch.cat((mask, mask * 0.55, mask * 0), dim=0)
            if mode == "Chart IDs":
                colors = chart_id_colors(field.chart_ids)
            elif mode == "Chart boundaries":
                colors = chart_boundary_colors(field, self.state.selected_albedo())
            else:
                colors = uv_collision_debug_colors(field)
            mask = self.backend.segment_mask(camera, self.state.selected_ids)
            chart_image = self.backend._sparse_map(
                camera, self.state.selected_ids, colors, mask.permute(1, 2, 0),
            ).clamp(0, 1).permute(2, 0, 1)
            return original * (1 - mask) + chart_image * mask
        if self.segment_count == 1:
            styled = self._render_state(camera, self.state, self.config, environment)
        else:
            if original is None:
                original = self.backend.render_original(camera)
            edited = [
                self._render_state(
                    camera, self.states[name], self.segment_configs[name], environment,
                )
                for name in self.segment_names
            ]
            styled = composite_independent_edits(original, edited)
        if mode == "Split original | styled":
            midpoint = styled.shape[2] // 2
            result = styled.clone()
            result[:, :, :midpoint] = original[:, :, :midpoint]
            result[:, :, midpoint - 1:midpoint + 1] = 1
            return result
        return styled

    def _render_state(self, camera, state, config, environment):
        render_mode = (
            "albedo" if config.train.albedo_only_render else config.train.render_mode
        )
        return (
            self.backend.render_albedo(camera, state)
            if render_mode == "albedo"
            else self.backend.render_stylized(
                camera, state, environment, render_mode=render_mode,
            )
        )

    def render_comparison_set(self, camera, environment) -> dict[str, torch.Tensor]:
        raise RuntimeError("this Viewer was not opened from a method baseline manifest")

    def comparison_metadata(self) -> dict:
        return {}


class MultiSegmentViewerScene(RayStyleViewerScene):
    """Compose independent legacy checkpoints from one scene in one Viewer."""

    def __init__(self, bundle_path: str):
        if not torch.cuda.is_available():
            raise RuntimeError("RayStyle Viewer requires an NVIDIA CUDA GPU")
        bundle = load_segment_bundle(bundle_path)
        loaded = [
            (entry, load_config(entry.config_path)) for entry in bundle.entries
        ]
        first_config = loaded[0][1]
        signature = self._scene_signature(first_config)
        for entry, config in loaded[1:]:
            if self._scene_signature(config) != signature:
                raise ValueError(
                    f"bundle segment {entry.name!r} does not use the same scene/backend"
                )
        self.config = first_config
        self.checkpoint_path = bundle.source
        self.backend = LegacyGaussianBackend(
            first_config.legacy_root, first_config.model_path, first_config.source_path,
            first_config.images, first_config.resolution, first_config.white_background,
        )
        self.segment_names = tuple(entry.name for entry, _ in loaded)
        self.states = {}
        self.selections = {}
        self.segment_configs = {}
        iterations = []
        union = torch.zeros(self.backend.point_count, dtype=torch.bool, device="cuda")
        for entry, config in loaded:
            selected, state, iteration = self._load_checkpoint_state(
                config, entry.checkpoint_path,
            )
            if bool((union & selected).any()):
                raise ValueError(
                    f"bundle segment {entry.name!r} overlaps an earlier segment"
                )
            union |= selected
            self.states[entry.name] = state
            self.selections[entry.name] = selected
            self.segment_configs[entry.name] = config
            iterations.append(iteration)
        self.state = self.states[self.segment_names[0]]
        self.selected = self.selections[self.segment_names[0]]
        self.iteration = min(iterations)
        self.comparison_names = ()
        self._initialize_scene(first_config)

    @staticmethod
    def _scene_signature(config: ExperimentConfig) -> tuple:
        return (
            Path(config.legacy_root).resolve(), Path(config.model_path).resolve(),
            Path(config.source_path).resolve(), config.images, config.resolution,
            config.white_background,
        )


class MethodComparisonViewerScene(RayStyleViewerScene):
    """Render paired baseline methods with one shared geometry and camera state."""

    def __init__(self, manifest_path: str, experiment: str | None = None):
        if not torch.cuda.is_available():
            raise RuntimeError("RayStyle Viewer requires an NVIDIA CUDA GPU")
        comparison = load_method_comparison(manifest_path, experiment)
        loaded = [(entry, load_config(entry.config_path)) for entry in comparison.entries]
        first_config = loaded[0][1]
        signature = MultiSegmentViewerScene._scene_signature(first_config)
        for entry, config in loaded[1:]:
            if MultiSegmentViewerScene._scene_signature(config) != signature:
                raise ValueError(
                    f"comparison method {entry.method!r} does not use the same scene/backend"
                )

        self.backend = LegacyGaussianBackend(
            first_config.legacy_root, first_config.model_path, first_config.source_path,
            first_config.images, first_config.resolution, first_config.white_background,
        )
        self.checkpoint_path = comparison.source
        self.comparison_experiment = comparison.experiment
        self.comparison_states = {}
        self.comparison_configs = {}
        self.comparison_checkpoints = {}
        self.comparison_iterations = {}
        selected_reference = None
        selected_by_label = {}
        for entry, config in loaded:
            selected, state, iteration = self._load_checkpoint_state(
                config, entry.checkpoint_path,
            )
            if selected_reference is None:
                selected_reference = selected
            elif not torch.equal(selected_reference, selected):
                raise ValueError(
                    f"comparison method {entry.method!r} does not use the same segment mask"
                )
            self.comparison_states[entry.label] = state
            self.comparison_configs[entry.label] = config
            self.comparison_checkpoints[entry.label] = entry.checkpoint_path
            self.comparison_iterations[entry.label] = iteration
            selected_by_label[entry.label] = selected

        self.comparison_names = tuple(self.comparison_states)
        default = "Atlas Ours" if "Atlas Ours" in self.comparison_states else self.comparison_names[0]
        self._active_comparison = default
        self.segment_names = ("Segment 1",)
        self.states = {"Segment 1": self.comparison_states[default]}
        self.selections = {"Segment 1": selected_by_label[default]}
        self.segment_configs = {"Segment 1": self.comparison_configs[default]}
        self.state = self.comparison_states[default]
        self.selected = selected_by_label[default]
        self.config = self.comparison_configs[default]
        self.iteration = self.comparison_iterations[default]
        self._comparison_selections = selected_by_label
        self.capture_root = comparison.source.parent / "viewer_comparisons" / comparison.experiment
        self._initialize_scene(first_config)

    @property
    def active_comparison_name(self) -> str:
        return self._active_comparison

    @property
    def total_selected_count(self) -> int:
        return self.state.selected_count

    def set_active_comparison(self, name: str) -> None:
        if name not in self.comparison_states:
            raise KeyError(f"unknown comparison method: {name}")
        self._active_comparison = name
        self.state = self.comparison_states[name]
        self.selected = self._comparison_selections[name]
        self.config = self.comparison_configs[name]
        self.iteration = self.comparison_iterations[name]
        self.states["Segment 1"] = self.state
        self.selections["Segment 1"] = self.selected
        self.segment_configs["Segment 1"] = self.config

    @torch.inference_mode()
    def render_comparison_set(self, camera, environment) -> dict[str, torch.Tensor]:
        images = {"Original": self.backend.render_original(camera)}
        for name in self.comparison_names:
            images[name] = self._render_state(
                camera, self.comparison_states[name], self.comparison_configs[name],
                environment,
            )
        return images

    def comparison_metadata(self) -> dict:
        return {
            "experiment": self.comparison_experiment,
            "methods": {
                name: {
                    "method": self.comparison_configs[name].method,
                    "iteration": self.comparison_iterations[name],
                    "checkpoint": str(self.comparison_checkpoints[name]),
                }
                for name in self.comparison_names
            },
        }


class RayStyleViewer:
    MODES = (
        "Styled", "Original", "Split original | styled", "Atlas albedo",
        "Chart boundaries", "Chart IDs", "Atlas texture preview",
        "UV collision debug", "Segment mask",
    )

    def __init__(self, scene: RayStyleViewerScene, scale: float = 2.0):
        # Delayed import keeps training, evaluation and --help usable on machines
        # without a desktop/OpenGL stack.
        import dearpygui.dearpygui as dpg

        self.dpg = dpg
        self.scene = scene
        source = scene.cameras[0]
        scale = max(float(scale), 1.0)
        self.width = max(320, int(source.image_width / scale))
        self.height = max(240, int(source.image_height / scale))
        self.layout = ViewerLayout(self.width, self.height)
        self.control_width = self.layout.control_width
        self.camera_index = 0
        self.camera_mode = "Calibrated"
        self.orbit = OrbitCamera(
            self.width, self.height, radius=2.0,
            fovy_degrees=math.degrees(float(source.FoVy)),
        )
        self._Camera = self._camera_class()
        self._dirty = True
        self._drag_left = False
        self._drag_middle = False
        self._mouse = (0.0, 0.0)
        self.last_image = torch.zeros(3, self.height, self.width)
        self.render_buffer = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.capture_path = Path(scene.config.output_dir) / "viewer_capture.png"
        self._build_gui()

    def _camera_class(self):
        # LegacyGaussianBackend has already placed legacy_root on sys.path.
        from scene.cameras import Camera
        return Camera

    def _new_camera(self, R, T, fovx, fovy):
        return self._Camera(
            colmap_id=0, R=np.asarray(R), T=np.asarray(T),
            FoVx=float(fovx), FoVy=float(fovy),
            image=torch.zeros(3, self.height, self.width),
            gt_alpha_mask=None, image_name="raystyle_viewer", uid=0,
        )

    def _current_camera(self):
        if self.camera_mode == "Calibrated":
            source = self.scene.cameras[self.camera_index]
            return self._new_camera(source.R, source.T, source.FoVx, source.FoVy)
        pose = self.orbit.pose
        fovy = math.radians(self.orbit.fovy_degrees)
        fy = self.height / (2 * math.tan(fovy / 2))
        fovx = 2 * math.atan(self.width / (2 * fy))
        return self._new_camera(pose[:3, :3], pose[:3, 3], fovx, fovy)

    def _mark_dirty(self, *_):
        self._dirty = True

    def _select_segment(self, _sender, value):
        self.scene.set_active_segment(value)
        self._dirty = True

    def _overview_text(self) -> str:
        method = (
            f"    ·    {self.scene.active_comparison_name}"
            if self.scene.comparison_count else ""
        )
        return (
            f"Iteration {self.scene.iteration:,}    ·    "
            f"{self.scene.total_selected_count:,} selected Gaussians    ·    "
            f"{self.scene.segment_count} segment(s){method}"
        )

    def _select_comparison(self, _sender, value):
        self.scene.set_active_comparison(value)
        self.dpg.set_value("_rs_overview", self._overview_text())
        self._dirty = True

    def _change_camera_index(self, delta: int):
        self.camera_index = int(np.clip(
            self.camera_index + delta, 0, len(self.scene.cameras) - 1,
        ))
        self.dpg.set_value("_rs_camera_index", self.camera_index)
        self.dpg.set_value("_rs_camera_label", self.scene.camera_labels[self.camera_index])
        self._dirty = True

    def _select_camera(self, _sender, value):
        self.camera_index = int(value)
        self.dpg.set_value("_rs_camera_label", self.scene.camera_labels[self.camera_index])
        self._dirty = True

    def _set_camera_mode(self, _sender, value):
        self.camera_mode = value
        calibrated = value == "Calibrated"
        for tag in ("_rs_camera_index", "_rs_previous", "_rs_next"):
            self.dpg.configure_item(tag, enabled=calibrated)
        self.dpg.configure_item("_rs_reset_orbit", enabled=not calibrated)
        self.dpg.set_value(
            "_rs_camera_hint",
            (
                "Evaluation cameras reproduce saved results."
                if calibrated else
                "Drag the render: left rotates, middle pans, wheel zooms."
            ),
        )
        self._dirty = True

    def _save_capture(self):
        save_image(self.capture_path, self.last_image)
        self.dpg.set_value("_rs_capture_status", f"Saved to\n{self.capture_path}")

    def _current_environment(self):
        return self.scene.environment(
            self.dpg.get_value("_rs_environment"),
            self.dpg.get_value("_rs_yaw"), self.dpg.get_value("_rs_exposure"),
        )

    def _save_comparison_set(self):
        if not self.scene.comparison_count:
            return
        camera = self._current_camera()
        environment = self._current_environment()
        images = self.scene.render_comparison_set(camera, environment)
        stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}"
        output = self.scene.capture_root / stamp
        output.mkdir(parents=True, exist_ok=False)
        saved = []
        for index, (name, image) in enumerate(images.items()):
            slug = name.lower().replace(" ", "_").replace("-", "_")
            path = output / f"{index:02d}_{slug}.png"
            save_image(path, image)
            saved.append(image.detach().clamp(0, 1))
        save_image(output / "comparison_row.png", torch.cat(saved, dim=2))
        metadata = {
            **self.scene.comparison_metadata(),
            "camera_mode": self.camera_mode,
            "camera_index": self.camera_index if self.camera_mode == "Calibrated" else None,
            "camera_label": (
                self.scene.camera_labels[self.camera_index]
                if self.camera_mode == "Calibrated" else "Free orbit"
            ),
            "environment": self.dpg.get_value("_rs_environment"),
            "hdr_rotation_degrees": float(self.dpg.get_value("_rs_yaw")),
            "exposure_ev": float(self.dpg.get_value("_rs_exposure")),
            "image_order": list(images),
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.dpg.set_value("_rs_capture_status", f"Saved method set to\n{output}")

    def _reset_orbit(self):
        self.orbit.reset()
        self._dirty = True

    def _build_gui(self):
        dpg = self.dpg
        layout = self.layout
        dpg.create_context()
        with dpg.theme() as viewer_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 16, 16)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 9)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 9, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)
        dpg.bind_theme(viewer_theme)
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.width, self.height, self.render_buffer,
                format=dpg.mvFormat_Float_rgb, tag="_rs_texture",
            )
        with dpg.window(
            tag="_rs_main", width=layout.main_width, height=layout.panel_height,
            pos=(0, 0),
            no_title_bar=True, no_move=True, no_resize=True, no_collapse=True,
        ):
            dpg.add_text("RAYSTYLE RENDER")
            dpg.add_text(
                f"{self.width} × {self.height}  ·  hover the image to navigate",
            )
            dpg.add_separator()
            dpg.add_spacer(height=layout.render_top_spacer)
            dpg.add_image("_rs_texture", tag="_rs_image")
        with dpg.window(
            tag="_rs_controls", width=self.control_width, height=layout.panel_height,
            pos=(layout.main_width + layout.panel_gap, 0),
            no_title_bar=True, no_move=True, no_resize=True, no_collapse=True,
        ):
            with dpg.child_window(height=82, border=True):
                dpg.add_text("OVERVIEW")
                dpg.add_text(self._overview_text(), tag="_rs_overview")

            extra_selector = self.scene.segment_count > 1 or self.scene.comparison_count
            display_height = 154 if extra_selector else 102
            with dpg.child_window(height=display_height, border=True):
                dpg.add_text("DISPLAY")
                dpg.add_combo(
                    self.MODES, default_value="Styled", width=-1,
                    tag="_rs_mode", callback=self._mark_dirty,
                )
                if self.scene.segment_count > 1:
                    dpg.add_text("Active diagnostic segment")
                    dpg.add_combo(
                        self.scene.segment_names,
                        default_value=self.scene.segment_names[0], width=-1,
                        tag="_rs_segment", callback=self._select_segment,
                    )
                elif self.scene.comparison_count:
                    dpg.add_text("Comparison method")
                    dpg.add_combo(
                        self.scene.comparison_names,
                        default_value=self.scene.active_comparison_name, width=-1,
                        tag="_rs_comparison", callback=self._select_comparison,
                    )

            with dpg.child_window(height=232, border=True):
                dpg.add_text("CAMERA")
                dpg.add_radio_button(
                    ("Calibrated", "Free orbit"), default_value="Calibrated",
                    horizontal=True, callback=self._set_camera_mode,
                )
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Previous", width=118, tag="_rs_previous",
                        callback=lambda: self._change_camera_index(-1),
                    )
                    dpg.add_button(
                        label="Next", width=118, tag="_rs_next",
                        callback=lambda: self._change_camera_index(1),
                    )
                    dpg.add_button(
                        label="Reset", width=118, tag="_rs_reset_orbit", enabled=False,
                        callback=self._reset_orbit,
                    )
                dpg.add_text("Camera index")
                dpg.add_slider_int(
                    min_value=0, max_value=len(self.scene.cameras) - 1,
                    default_value=0, width=-1, tag="_rs_camera_index",
                    callback=self._select_camera,
                )
                dpg.add_text(self.scene.camera_labels[0], tag="_rs_camera_label")
                dpg.add_text(
                    "Evaluation cameras reproduce saved results.",
                    tag="_rs_camera_hint", wrap=self.control_width - 54,
                )

            environment_labels = tuple(self.scene.environment_choices)
            with dpg.child_window(height=222, border=True):
                dpg.add_text("LIGHTING")
                dpg.add_text("Environment")
                dpg.add_combo(
                    environment_labels, default_value=environment_labels[1], width=-1,
                    tag="_rs_environment", callback=self._mark_dirty,
                )
                dpg.add_text("HDR rotation")
                dpg.add_slider_float(
                    min_value=0, max_value=360, default_value=0, width=-1,
                    format="%.0f°", tag="_rs_yaw", callback=self._mark_dirty,
                )
                dpg.add_text("Exposure")
                dpg.add_slider_float(
                    min_value=-4, max_value=4, default_value=0, width=-1,
                    format="%+.2f EV", tag="_rs_exposure", callback=self._mark_dirty,
                )

            capture_height = 170 if self.scene.comparison_count else 130
            with dpg.child_window(height=capture_height, border=True):
                dpg.add_text("CAPTURE & STATUS")
                dpg.add_button(
                    label="Save screenshot", width=-1, callback=self._save_capture,
                )
                if self.scene.comparison_count:
                    dpg.add_button(
                        label="Save Original + all methods", width=-1,
                        callback=self._save_comparison_set,
                    )
                dpg.add_text("Waiting for first render…", tag="_rs_render_status")
                dpg.add_text("", tag="_rs_capture_status", wrap=self.control_width - 54)

            with dpg.child_window(height=70, border=True):
                dpg.add_text("NAVIGATION")
                dpg.add_text(
                    "Free orbit: left drag rotate  ·  middle drag pan  ·  wheel zoom",
                    wrap=self.control_width - 54,
                )

        def start_drag(_sender, button):
            if not dpg.is_item_hovered("_rs_image") or self.camera_mode != "Free orbit":
                return
            self._drag_left = button == dpg.mvMouseButton_Left
            self._drag_middle = button == dpg.mvMouseButton_Middle
            self._mouse = tuple(dpg.get_mouse_pos(local=False))

        def stop_drag(_sender, button):
            if button == dpg.mvMouseButton_Left:
                self._drag_left = False
            elif button == dpg.mvMouseButton_Middle:
                self._drag_middle = False

        def mouse_move(_sender, position):
            x, y = position
            dx, dy = x - self._mouse[0], y - self._mouse[1]
            self._mouse = (x, y)
            if self._drag_left:
                self.orbit.orbit(dx, dy)
                self._dirty = True
            elif self._drag_middle:
                self.orbit.pan(dx, dy)
                self._dirty = True

        def mouse_wheel(_sender, delta):
            if dpg.is_item_hovered("_rs_image") and self.camera_mode == "Free orbit":
                self.orbit.zoom(delta)
                self._dirty = True

        with dpg.handler_registry():
            dpg.add_mouse_click_handler(callback=start_drag)
            dpg.add_mouse_release_handler(callback=stop_drag)
            dpg.add_mouse_move_handler(callback=mouse_move)
            dpg.add_mouse_wheel_handler(callback=mouse_wheel)

        dpg.create_viewport(
            title="RayStyle 3D Gaussian Viewer",
            width=layout.viewport_width + 16,
            height=layout.panel_height + 40,
            resizable=False,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def _render_once(self):
        dpg = self.dpg
        started = time.perf_counter()
        camera = self._current_camera()
        environment = self._current_environment()
        image = self.scene.render(camera, dpg.get_value("_rs_mode"), environment)
        self.last_image = image.detach()
        values = image.detach().clamp(0, 1).permute(1, 2, 0).contiguous().cpu().numpy()
        dpg.set_value("_rs_texture", values)
        milliseconds = 1000 * (time.perf_counter() - started)
        dpg.set_value(
            "_rs_render_status",
            f"{self.camera_mode} · {self.width}×{self.height} · {milliseconds:.1f} ms",
        )
        print(
            f"[RayStyle Viewer] rendered {self.camera_mode.lower()} frame "
            f"at {self.width}x{self.height} in {milliseconds:.1f} ms",
            flush=True,
        )
        self._dirty = False

    def run(self):
        dpg = self.dpg
        try:
            while dpg.is_dearpygui_running():
                if self._dirty:
                    try:
                        self._render_once()
                    except Exception as error:
                        print(f"[RayStyle Viewer] render failed: {error}", flush=True)
                        dpg.set_value("_rs_render_status", f"Render failed: {error}")
                        self._dirty = False
                dpg.render_dearpygui_frame()
        finally:
            dpg.destroy_context()


def view(config: ExperimentConfig, checkpoint_path: str, scale: float = 2.0):
    scene = RayStyleViewerScene(config, checkpoint_path)
    RayStyleViewer(scene, scale=scale).run()


def view_bundle(bundle_path: str, scale: float = 2.0):
    scene = MultiSegmentViewerScene(bundle_path)
    RayStyleViewer(scene, scale=scale).run()


def view_methods(
    manifest_path: str, experiment: str | None = None, scale: float = 2.0,
):
    scene = MethodComparisonViewerScene(manifest_path, experiment)
    RayStyleViewer(scene, scale=scale).run()
