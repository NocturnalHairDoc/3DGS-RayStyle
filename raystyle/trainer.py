from __future__ import annotations

import time
from pathlib import Path

import torch
from tqdm import trange

from .atlas import AtlasTopology
from .backend import LegacyGaussianBackend
from .config import ExperimentConfig
from .environments import EnvironmentPool
from .features import FrozenDINO, dino_content_loss
from .graph import AnchorGraph
from .io_utils import append_jsonl, load_image, save_image, seed_everything, write_json
from .losses import (
    ReferenceStyleLoss, boundary_outside_preservation_loss,
    illumination_consistency_loss, outside_preservation_loss,
)
from .project_state import load_segment
from .reference_layout import build_reference_layout, metric_tile_grid
from .style_state import StyleState


class Trainer:
    def __init__(self, config: ExperimentConfig, resume_checkpoint: str | None = None):
        if not torch.cuda.is_available():
            raise RuntimeError("the legacy Gaussian rasterizer requires an NVIDIA CUDA GPU")
        self.config = config
        seed_everything(config.train.seed)
        self.output = Path(config.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.backend = LegacyGaussianBackend(
            config.legacy_root, config.model_path, config.source_path,
            config.images, config.resolution, config.white_background,
        )
        selected_cpu, state_metadata = load_segment(
            config.project_state, config.segment_id, self.backend.point_count,
        )
        self.selected = selected_cpu.cuda()
        selected_ids = torch.where(self.selected)[0]
        self.state_metadata = state_metadata
        self.reference = load_image(config.reference_image)
        resume_payload = None
        if resume_checkpoint:
            resume_payload = torch.load(
                resume_checkpoint, map_location="cuda", weights_only=False,
            )
        atlas_topology = (
            AtlasTopology.from_checkpoint_state(resume_payload["state_dict"])
            if resume_payload is not None and config.train.texture_mapping == "atlas"
            else None
        )
        self.state = StyleState(
            self.backend.base_albedo, self.selected, config.method,
            original_sh_degree=int(self.backend.gaussians.active_sh_degree),
            residual_degree=config.train.sh_degree,
            residual_limit=config.train.residual_limit,
            global_shift_limit=config.train.global_shift_limit,
            detail_residual_limit=config.train.detail_residual_limit,
            selected_xyz=self.backend.xyz[self.selected] if config.method == "ours" else None,
            selected_normals=(
                self.backend.canonical_normals(selected_ids)
                if config.method == "ours" else None
            ),
            selected_visibility=(
                self.backend.gaussians.get_opacity.detach()[self.selected]
                if config.method == "ours" else None
            ),
            texture_resolution=config.train.texture_resolution,
            texture_logit_limit=config.train.texture_logit_limit,
            texture_mapping=config.train.texture_mapping,
            atlas_charts=config.train.atlas_charts,
            atlas_neighbours=config.train.atlas_neighbours,
            atlas_padding=config.train.atlas_padding,
            atlas_feather=config.train.atlas_feather,
            atlas_uv_offset_limit=config.train.atlas_uv_offset_limit,
            atlas_source_layout=config.train.atlas_source_layout,
            atlas_reference_repeat=config.train.atlas_reference_repeat,
            atlas_topology=atlas_topology,
            albedo_mode=config.train.albedo_mode,
            pbr_diffuse_white=config.train.pbr_diffuse_white,
            pbr_exposure=config.train.pbr_exposure,
            pbr_white_point=config.train.pbr_white_point,
        ).cuda()
        tile_grid = None
        if config.train.reference_metric_tiles:
            tile_grid = metric_tile_grid(
                self.state.texture_field,
                self.backend.xyz[self.selected],
                config.train.reference_tile_count,
            )
        self.reference_layout = build_reference_layout(
            self.reference,
            mode=config.train.reference_layout,
            patch_count=config.train.reference_saliency_patches,
            canvas_size=config.train.texture_resolution,
            focus_scale=config.train.reference_focus_scale,
            tile_count=config.train.reference_tile_count,
            tile_grid=tile_grid,
        )
        save_image(self.output / "reference_canvas.png", self.reference_layout.canvas)
        save_image(
            self.output / "reference_saliency.png",
            self.reference_layout.saliency.expand(3, -1, -1),
        )
        save_image(
            self.output / "reference_selection.png",
            self.reference_layout.selection_preview,
        )
        write_json(self.output / "reference_layout.json", {
            "mode": config.train.reference_layout,
            "regions": [list(region) for region in self.reference_layout.regions],
            "scales": list(self.reference_layout.scales),
            "tile_grid": list(self.reference_layout.tile_grid),
        })
        texture_reference = (
            self.reference_layout.canvas
            if config.train.texture_mapping == "atlas"
            else self.reference
        )
        self.state.initialize_texture(texture_reference, config.train.texture_init_strength)
        self.graph = AnchorGraph.from_points(
            self.backend.xyz[self.selected], config.train.graph_anchors,
            config.train.graph_neighbours,
        )
        self.dino = FrozenDINO(config.legacy_root, config.dino_checkpoint, config.train.image_size).cuda()
        patch_reference = (
            self.reference_layout.canvas
            if config.train.style_patch_reference == "layout"
            else self.reference
        )
        with torch.no_grad():
            reference_features = self.dino(self.reference)
            patch_reference_features = self.dino(patch_reference)
        self.style_loss = ReferenceStyleLoss(
            reference_features,
            self.reference,
            patch_dino_features=patch_reference_features,
            patch_reference_rgb=patch_reference,
        )
        self.environments = EnvironmentPool(config.environment_dir, config.train.seed)
        parameters = [p for p in self.state.parameters() if p.requires_grad and p.numel()]
        if not parameters:
            raise RuntimeError(f"method {config.method} created no trainable parameters")
        if config.method == "ours":
            texture_parameters = [
                self.state.global_albedo_shift,
                self.state.texture_field.logit_grid_raw,
            ]
            if hasattr(self.state.texture_field, "uv_offset_raw"):
                texture_parameters.append(self.state.texture_field.uv_offset_raw)
            texture_ids = {id(parameter) for parameter in texture_parameters}
            material_parameters = [
                parameter for parameter in parameters if id(parameter) not in texture_ids
            ]
            groups = [
                {"params": texture_parameters, "lr": config.train.texture_learning_rate},
                {"params": material_parameters, "lr": config.train.learning_rate},
            ]
            self.optimizer = torch.optim.Adam(groups)
        else:
            self.optimizer = torch.optim.Adam(parameters, lr=config.train.learning_rate)
        self.start_iteration = 0
        if resume_checkpoint:
            payload = resume_payload
            saved_method = payload.get("state_metadata", {}).get("method")
            if saved_method and saved_method != config.method:
                raise ValueError(
                    f"resume checkpoint method {saved_method!r} != {config.method!r}"
                )
            saved_albedo_mode = payload.get("state_metadata", {}).get(
                "albedo_mode", "additive",
            )
            if config.method == "ours" and saved_albedo_mode != config.train.albedo_mode:
                raise ValueError(
                    f"resume checkpoint albedo mode {saved_albedo_mode!r} != "
                    f"configured {config.train.albedo_mode!r}"
                )
            saved_mapping = payload.get("state_metadata", {}).get(
                "texture_mapping", "planar",
            )
            if config.method == "ours" and saved_mapping != config.train.texture_mapping:
                raise ValueError(
                    f"resume checkpoint texture mapping {saved_mapping!r} != "
                    f"configured {config.train.texture_mapping!r}"
                )
            self.state.load_checkpoint_state(payload["state_dict"])
            if payload.get("optimizer_state_dict"):
                self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            self.start_iteration = int(payload.get("iteration", 0))
            if self.start_iteration >= config.train.iterations:
                raise ValueError(
                    f"resume iteration {self.start_iteration} is not below "
                    f"configured iterations {config.train.iterations}"
                )
        self.geometry_fingerprint = self.backend.geometry_fingerprint()
        self.preview_camera = self._best_preview_camera()
        write_json(self.output / "resolved_config.json", config.to_dict())

    @torch.no_grad()
    def _best_preview_camera(self):
        best = None
        for camera in self.backend.test_cameras:
            mask = self.backend.segment_mask(camera, self.state.selected_ids)
            coverage = float((mask > 0.2).float().mean())
            if best is None or coverage > best[0]:
                best = (coverage, camera)
        return best[1] if best is not None else self.backend.train_cameras[0]

    def _render(self, camera, environment):
        render_mode = (
            "albedo" if self.config.train.albedo_only_render
            else self.config.train.render_mode
        )
        if render_mode == "albedo":
            return self.backend.render_albedo(camera, self.state)
        return self.backend.render_stylized(
            camera, self.state, environment, render_mode=render_mode,
        )

    @torch.no_grad()
    def _save_fixed_previews(self, iteration: int):
        camera = self.preview_camera
        fixed = self._render(camera, self.environments.fixed)
        save_image(
            self.output / "previews_fixed" / f"{iteration:06d}_fixed.png", fixed,
        )
        for index, source in enumerate(self.environments.heldout[:2]):
            environment = type(source)(source.name, source.pixels, 0.0, 0.0)
            image = self._render(camera, environment)
            save_image(
                self.output / "previews_fixed" /
                f"{iteration:06d}_unseen_{index + 1}.png",
                image,
            )

    def _sample_visible_camera(self):
        best = None
        for _ in range(32):
            index = int(torch.randint(len(self.backend.train_cameras), (1,)).item())
            camera = self.backend.train_cameras[index]
            with torch.no_grad():
                mask = self.backend.segment_mask(camera, self.state.selected_ids)
            coverage = float((mask > 0.2).float().mean())
            if best is None or coverage > best[0]:
                best = (coverage, camera, mask)
            if coverage >= self.config.train.min_segment_coverage:
                return camera, mask
        if best is not None and best[0] > 1e-4:
            return best[1], best[2]
        raise RuntimeError("selected segment was not visible in 32 sampled training cameras")

    def _sample_different_environment(self, first):
        cfg = self.config.train
        candidate = None
        for _ in range(8):
            candidate = self.environments.sample(
                True, (cfg.random_exposure_min, cfg.random_exposure_max),
            )
            if candidate.name != first.name:
                return candidate
        return candidate

    def save_checkpoint(self, iteration: int):
        payload = {
            "iteration": int(iteration),
            "state_dict": self.state.state_dict(),
            "state_metadata": self.state.checkpoint_metadata(),
            "config": self.config.to_dict(),
            "project_metadata": self.state_metadata,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(payload, self.output / f"checkpoint_{iteration:06d}.pt")
        torch.save(payload, self.output / "checkpoint_latest.pt")
        texture = self.state.texture_preview()
        if texture is not None:
            save_image(self.output / "textures" / f"texture_{iteration:06d}.png", texture)
            save_image(self.output / "texture_latest.png", texture)

    def train(self):
        cfg, weights = self.config.train, self.config.losses
        log_path = self.output / "train.jsonl"
        progress = trange(
            self.start_iteration + 1, cfg.iterations + 1,
            desc=f"RayStyle/{self.config.method}",
        )
        for iteration in progress:
            started = time.perf_counter()
            camera, mask = self._sample_visible_camera()
            texture_stage = (
                self.config.method == "ours"
                and iteration <= min(cfg.texture_stage_iterations, cfg.iterations)
            )
            environment = (
                self.environments.neutral
                if texture_stage else (
                    self.environments.sample(
                        True, (cfg.random_exposure_min, cfg.random_exposure_max),
                    )
                    if cfg.random_hdr else self.environments.fixed
                )
            )
            with torch.no_grad():
                original = self.backend.render_original(camera)
                original_features = self.dino(original)
            stylized = self._render(camera, environment)
            stylized_features = self.dino(stylized)

            texture_anchor, texture_delta_tv = self.state.texture_regularization()
            atlas_terms = self.state.atlas_regularization()
            texture_preview = self.state.texture_preview()
            color_mean = (
                self.style_loss.color_mean_loss(texture_preview)
                if texture_preview is not None
                else stylized.sum() * 0
            )

            terms = {
                "style": self.style_loss.global_loss(stylized, stylized_features, mask),
                "patch_style": self.style_loss.patch_loss(stylized, stylized_features, mask),
                "content": dino_content_loss(stylized_features, original_features, mask),
                "graph": self.graph.regularize(
                    self.state.graph_values(cfg.graph_scope),
                ),
                "outside": outside_preservation_loss(stylized, original, mask),
                "boundary_outside": boundary_outside_preservation_loss(
                    stylized, original, mask,
                ),
                "material_prior": self.state.material_prior(),
                "texture_anchor": texture_anchor,
                "texture_delta_tv": texture_delta_tv,
                "color_mean": color_mean,
                "render_color": self.style_loss.rendered_color_loss(
                    stylized, mask, environment.exposure,
                ),
                **atlas_terms,
            }
            primary_total = sum(
                getattr(weights, name) * value for name, value in terms.items()
            )
            self.optimizer.zero_grad(set_to_none=True)
            primary_total.backward()

            # Render the second HDR only after the DINO/PBR primary graph has
            # been backpropagated and released. Holding both full-resolution
            # Gaussian graphs simultaneously exceeds 31 GB on bicycle.
            second_environment = None
            if self.config.method in {"ours", "pbr_only"} and not texture_stage:
                second_environment = (
                    self._sample_different_environment(environment)
                    if cfg.random_hdr else self.environments.neutral
                )
                second_stylized = self._render(camera, second_environment)
                hdr_consistency = illumination_consistency_loss(
                    stylized.detach(), second_stylized, mask,
                )
                (weights.hdr_consistency * hdr_consistency).backward()
            else:
                hdr_consistency = stylized.detach().sum() * 0
            terms["hdr_consistency"] = hdr_consistency
            total = (
                primary_total.detach()
                + weights.hdr_consistency * hdr_consistency.detach()
            )
            if texture_stage:
                # Stage 1 forces the UV texture to explain the reference
                # pattern. Material and view-dependent residuals are unlocked
                # only for held-out-light refinement in stage 2.
                self.state.roughness_logits.grad = None
                self.state.metallic_logits.grad = None
                self.state.sh_residual.grad = None
            self.optimizer.step()

            record = {
                "iteration": iteration,
                "total": float(total.detach()),
                **{name: float(value.detach()) for name, value in terms.items()},
                "environment": environment.name,
                "second_environment": (
                    second_environment.name if second_environment is not None else None
                ),
                "stage": "texture" if texture_stage else "material_hdr",
                "seconds": time.perf_counter() - started,
            }
            record.update({
                f"atlas_{name}": float(value.detach())
                for name, value in self.state.atlas_diagnostics().items()
            })
            append_jsonl(log_path, record)
            progress.set_postfix(loss=f"{record['total']:.4f}")

            if iteration == 1 or iteration % cfg.preview_every == 0 or iteration == cfg.iterations:
                save_image(self.output / "previews" / f"{iteration:06d}.png", stylized)
                self._save_fixed_previews(iteration)
            if iteration % cfg.checkpoint_every == 0 or iteration == cfg.iterations:
                self.save_checkpoint(iteration)

        if self.backend.geometry_fingerprint() != self.geometry_fingerprint:
            raise RuntimeError("frozen geometry/scale/rotation/opacity changed during training")
        return self.output / "checkpoint_latest.pt"
