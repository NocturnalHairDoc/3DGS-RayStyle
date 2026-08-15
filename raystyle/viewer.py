from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .backend import LegacyGaussianBackend
from .config import ExperimentConfig
from .environments import EnvironmentMap, EnvironmentPool
from .io_utils import save_image
from .project_state import load_segment
from .style_state import StyleState


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
        selected, _ = load_segment(
            config.project_state, config.segment_id, self.backend.point_count,
        )
        self.selected = selected.cuda()
        payload = torch.load(checkpoint, map_location="cuda", weights_only=False)
        metadata = payload.get("state_metadata", {})
        saved_method = metadata.get("method", config.method)
        if saved_method != config.method:
            raise ValueError(
                f"checkpoint method is {saved_method!r}, config method is {config.method!r}"
            )
        texture_resolution = metadata.get("texture_resolution") or config.train.texture_resolution
        self.state = StyleState(
            self.backend.base_albedo, self.selected, config.method,
            original_sh_degree=int(self.backend.gaussians.active_sh_degree),
            residual_degree=int(metadata.get("residual_degree", config.train.sh_degree)),
            residual_limit=float(metadata.get("residual_limit", config.train.residual_limit)),
            global_shift_limit=float(metadata.get(
                "global_shift_limit", config.train.global_shift_limit,
            )),
            detail_residual_limit=float(metadata.get(
                "detail_residual_limit", config.train.detail_residual_limit,
            )),
            selected_xyz=(self.backend.xyz[self.selected] if config.method == "ours" else None),
            selected_normals=(
                self.backend.canonical_normals(torch.where(self.selected)[0])
                if config.method == "ours" else None
            ),
            texture_resolution=int(texture_resolution),
            texture_logit_limit=config.train.texture_logit_limit,
            texture_mapping=str(metadata.get("texture_mapping", "planar")),
            albedo_mode=str(metadata.get("albedo_mode", "additive")),
            pbr_diffuse_white=float(metadata.get("pbr_diffuse_white", 0.0)),
            pbr_exposure=float(metadata.get("pbr_exposure", 0.0)),
            pbr_white_point=float(metadata.get("pbr_white_point", 0.0)),
        ).cuda()
        self.state.load_checkpoint_state(payload["state_dict"])
        self.state.eval()
        self.iteration = int(payload.get("iteration", 0))

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
        if mode in {"Original", "Split original | styled", "UV albedo"}:
            original = self.backend.render_original(camera)
        if mode == "Original":
            return original
        if mode == "Segment mask":
            mask = self.backend.segment_mask(camera, self.state.selected_ids)
            return torch.cat((mask, mask * 0.55, torch.zeros_like(mask)), dim=0)
        if mode == "UV albedo":
            mask = self.backend.segment_mask(camera, self.state.selected_ids)
            divisor = mask.permute(1, 2, 0)
            albedo = self.backend._sparse_map(
                camera, self.state.selected_ids, self.state.selected_albedo(), divisor,
            ).clamp(0, 1).permute(2, 0, 1)
            return original * (1 - mask) + albedo * mask
        styled = self.backend.render_stylized(camera, self.state, environment)
        if mode == "Split original | styled":
            midpoint = styled.shape[2] // 2
            result = styled.clone()
            result[:, :, :midpoint] = original[:, :, :midpoint]
            result[:, :, midpoint - 1:midpoint + 1] = 1
            return result
        return styled


class RayStyleViewer:
    MODES = ("Styled", "Original", "Split original | styled", "UV albedo", "Segment mask")

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
        self.control_width = 370
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
        enabled = value == "Calibrated"
        self.dpg.configure_item("_rs_camera_index", enabled=enabled)
        self._dirty = True

    def _save_capture(self):
        save_image(self.capture_path, self.last_image)
        self.dpg.set_value("_rs_status", f"Saved {self.capture_path}")

    def _reset_orbit(self):
        self.orbit.reset()
        self._dirty = True

    def _build_gui(self):
        dpg = self.dpg
        dpg.create_context()
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.width, self.height, self.render_buffer,
                format=dpg.mvFormat_Float_rgb, tag="_rs_texture",
            )
        with dpg.window(tag="_rs_main", width=self.width, height=self.height):
            dpg.add_image("_rs_texture", tag="_rs_image")
        with dpg.window(
            tag="_rs_controls", label="RayStyle Controls",
            width=self.control_width, height=self.height, pos=(self.width + 8, 0),
        ):
            dpg.add_text(f"Checkpoint iteration: {self.scene.iteration}")
            dpg.add_text(f"Selected Gaussians: {self.scene.state.selected_count:,}")
            dpg.add_separator()
            dpg.add_combo(
                self.MODES, default_value="Styled", label="View",
                tag="_rs_mode", callback=self._mark_dirty,
            )
            dpg.add_combo(
                ("Calibrated", "Free orbit"), default_value="Calibrated",
                label="Camera", callback=self._set_camera_mode,
            )
            with dpg.group(horizontal=True):
                dpg.add_button(label="Previous", callback=lambda: self._change_camera_index(-1))
                dpg.add_button(label="Next", callback=lambda: self._change_camera_index(1))
                dpg.add_button(label="Reset orbit", callback=self._reset_orbit)
            dpg.add_slider_int(
                min_value=0, max_value=len(self.scene.cameras) - 1,
                default_value=0, label="Camera index", tag="_rs_camera_index",
                callback=self._select_camera,
            )
            dpg.add_text(self.scene.camera_labels[0], tag="_rs_camera_label")
            dpg.add_separator()
            environment_labels = tuple(self.scene.environment_choices)
            dpg.add_combo(
                environment_labels, default_value=environment_labels[1],
                label="Environment", tag="_rs_environment", callback=self._mark_dirty,
            )
            dpg.add_slider_float(
                min_value=0, max_value=360, default_value=0,
                label="HDR yaw", tag="_rs_yaw", callback=self._mark_dirty,
            )
            dpg.add_slider_float(
                min_value=-4, max_value=4, default_value=0,
                label="Exposure", tag="_rs_exposure", callback=self._mark_dirty,
            )
            dpg.add_separator()
            dpg.add_button(label="Save screenshot", width=-1, callback=self._save_capture)
            dpg.add_text("", tag="_rs_status", wrap=self.control_width - 20)
            dpg.add_separator()
            dpg.add_text("Free orbit: left drag rotate, middle drag pan, wheel zoom", wrap=330)

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
            width=self.width + self.control_width + 18,
            height=self.height + 40,
            resizable=False,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def _render_once(self):
        dpg = self.dpg
        started = time.perf_counter()
        camera = self._current_camera()
        environment = self.scene.environment(
            dpg.get_value("_rs_environment"),
            dpg.get_value("_rs_yaw"), dpg.get_value("_rs_exposure"),
        )
        image = self.scene.render(camera, dpg.get_value("_rs_mode"), environment)
        self.last_image = image.detach()
        values = image.detach().clamp(0, 1).permute(1, 2, 0).contiguous().cpu().numpy()
        dpg.set_value("_rs_texture", values)
        milliseconds = 1000 * (time.perf_counter() - started)
        dpg.set_value(
            "_rs_status",
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
                        dpg.set_value("_rs_status", f"Render failed: {error}")
                        self._dirty = False
                dpg.render_dearpygui_frame()
        finally:
            dpg.destroy_context()


def view(config: ExperimentConfig, checkpoint_path: str, scale: float = 2.0):
    scene = RayStyleViewerScene(config, checkpoint_path)
    RayStyleViewer(scene, scale=scale).run()
