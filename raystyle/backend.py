from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

from .environments import EnvironmentMap


def _normalise(values: torch.Tensor):
    return F.normalize(values, dim=-1, eps=1e-7)


def transform_points_row(points: torch.Tensor, transform: torch.Tensor):
    """Apply a legacy 3DGS transform stored for row-vector multiplication."""
    homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    return homogeneous @ transform


def calibrated_tone_map(
    values: torch.Tensor, exposure_stops: float = 0.0, white_point: float = 1.0,
):
    """Exposure plus extended Reinhard with an explicit scene white point.

    A white point of one is identity below display white, which is appropriate
    after environment irradiance has been normalized to one. A non-positive
    white point retains the legacy uncalibrated Reinhard curve for old models.
    """
    radiance = values.clamp_min(0) * (2.0 ** float(exposure_stops))
    if float(white_point) <= 0:
        return radiance / (1 + radiance)
    white_square = max(float(white_point) ** 2, 1e-6)
    mapped = radiance * (1 + radiance / white_square) / (1 + radiance)
    return mapped.clamp(0, 1)


class LegacyGaussianBackend:
    """Thin adapter around the existing project's scene format and CUDA rasterizer."""

    def __init__(self, legacy_root: str, model_path: str, source_path: str,
                 images="images", resolution=-1, white_background=False):
        root = str(Path(legacy_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from gaussian_renderer import render
        from scene import GaussianModel, Scene
        from utils.general_utils import build_rotation
        from utils.sh_utils import eval_sh

        self._render = render
        self._eval_sh = eval_sh
        self._build_rotation = build_rotation
        args = Namespace(
            sh_degree=3,
            feature_dim=32,
            init_from_3dgs_pcd=False,
            source_path=str(Path(source_path).resolve()),
            model_path=str(Path(model_path).resolve()),
            feature_model_path="",
            images=str(images),
            resolution=int(resolution),
            white_background=bool(white_background),
            data_device="cuda",
            eval=True,
            need_features=False,
            need_masks=False,
            allow_principle_point_shift=False,
        )
        self.gaussians = GaussianModel(args.sh_degree)
        self.scene = Scene(
            args, self.gaussians, load_iteration=-1, shuffle=False,
            target="scene", mode="train",
        )
        # The legacy GaussianModel is a plain class rather than nn.Module.
        # Freeze its explicit tensors individually; only StyleState is passed
        # to the optimizer.
        for name in (
            "_xyz", "_features_dc", "_features_rest", "_opacity",
            "_scaling", "_rotation",
        ):
            getattr(self.gaussians, name).requires_grad_(False)
        self.pipeline = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        self.background = torch.ones(3, device="cuda") if white_background else torch.zeros(3, device="cuda")
        self.train_cameras = list(self.scene.getTrainCameras())
        self.test_cameras = list(self.scene.getTestCameras()) or self.train_cameras[::max(1, len(self.train_cameras) // 12)]
        if not self.train_cameras:
            raise ValueError("scene contains no training cameras")

    @property
    def point_count(self):
        return int(self.gaussians.get_xyz.shape[0])

    @property
    def xyz(self):
        return self.gaussians.get_xyz.detach()

    @property
    def base_albedo(self):
        dc = self.gaussians._features_dc.detach()[:, 0]
        return (0.5 + 0.28209479177387814 * dc).clamp(1e-4, 1 - 1e-4)

    def geometry_fingerprint(self):
        tensors = (
            self.gaussians._xyz, self.gaussians._scaling,
            self.gaussians._rotation, self.gaussians._opacity,
        )
        return tuple(
            (tuple(tensor.shape), float(tensor.detach().double().sum().item()),
             float(tensor.detach().double().square().sum().item()))
            for tensor in tensors
        )

    def render_colors(self, camera, colors: torch.Tensor):
        return self._render(camera, self.gaussians, self.pipeline, self.background, override_color=colors)["render"]

    def render_original(self, camera):
        return self._render(camera, self.gaussians, self.pipeline, self.background)["render"].clamp(0, 1)

    def projected_visible_samples(
        self, camera, visibility_ids: torch.Tensor, sample_ids: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Project a shared Gaussian subset and retain approximate frontmost samples."""
        height, width = int(camera.image_height), int(camera.image_width)
        projection = camera.full_proj_transform.to(self.xyz.device, dtype=self.xyz.dtype)

        def project(ids):
            xyz = self.xyz[ids]
            clip = transform_points_row(xyz, projection)
            depth = clip[:, 3]
            safe = depth.unsqueeze(1).clamp_min(1e-6)
            ndc = clip[:, :3] / safe
            u = ((ndc[:, 0] + 1) * 0.5 * (width - 1)).long().clamp(0, width - 1)
            v = ((1 - ndc[:, 1]) * 0.5 * (height - 1)).long().clamp(0, height - 1)
            valid = (
                (depth > 1e-6) & (ndc[:, 2] > 0)
                & (ndc[:, 0].abs() <= 1) & (ndc[:, 1].abs() <= 1)
            )
            return u, v, depth, ndc, valid

        all_u, all_v, all_depth, _, all_valid = project(visibility_ids)
        nearest = torch.full(
            (height * width,), torch.inf,
            dtype=all_depth.dtype, device=all_depth.device,
        )
        all_pixels = all_v[all_valid] * width + all_u[all_valid]
        nearest.scatter_reduce_(
            0, all_pixels, all_depth[all_valid], reduce="amin", include_self=True,
        )

        u, v, depth, ndc, valid = project(sample_ids)
        pixels = v * width + u
        front = nearest[pixels]
        tolerance = torch.maximum(front.abs() * 0.01, torch.full_like(front, 1e-4))
        valid &= depth <= front + tolerance
        support = mask[0] if mask.ndim == 3 else mask[0, 0]
        valid &= support[v, u] > 0.15
        ids = sample_ids[valid]
        # grid_sample uses image coordinates, whose vertical axis is -NDC y.
        grid = torch.stack((ndc[valid, 0], -ndc[valid, 1]), dim=1)
        return ids, grid

    @staticmethod
    def sample_projected_features(features: torch.Tensor, grid: torch.Tensor):
        if not len(grid):
            return features.new_empty((0, features.shape[1]))
        sampled = F.grid_sample(
            features, grid.view(1, -1, 1, 2), mode="bilinear",
            padding_mode="border", align_corners=True,
        )
        return sampled[0, :, :, 0].T

    def directions(self, camera, ids: torch.Tensor | None = None):
        center = camera.camera_center.reshape(1, 3)
        xyz = self.gaussians.get_xyz.detach()
        if ids is not None:
            xyz = xyz[ids]
        return _normalise(xyz - center)

    def native_radiance(self, camera):
        features = self.gaussians.get_features.detach().transpose(1, 2)
        values = self._eval_sh(int(self.gaussians.active_sh_degree), features, self.directions(camera))
        return (values + 0.5).clamp(0, 1)

    def segment_mask(self, camera, selected_ids: torch.Tensor):
        values = torch.zeros(self.point_count, 3, device="cuda")
        values[selected_ids] = 1
        return self.render_colors(camera, values)[:1].clamp(0, 1)

    def _sparse_map(self, camera, selected_ids: torch.Tensor, selected_values: torch.Tensor, divisor: torch.Tensor):
        full = torch.zeros(self.point_count, selected_values.shape[-1], device="cuda", dtype=selected_values.dtype)
        full = full.index_copy(0, selected_ids, selected_values)
        rendered = self.render_colors(camera, full).permute(1, 2, 0)
        return rendered / divisor.clamp_min(1e-5)

    def gaussian_normals(self, camera, ids: torch.Tensor):
        normals = self.canonical_normals(ids)
        view = -self.directions(camera, ids)
        normals = torch.where((normals * view).sum(1, keepdim=True) < 0, -normals, normals)
        return _normalise(normals)

    def canonical_normals(self, ids: torch.Tensor):
        rotation = self._build_rotation(self.gaussians.get_rotation.detach()[ids])
        axis = self.gaussians.get_scaling.detach()[ids].argmin(dim=1)
        rows = torch.arange(len(ids), device=ids.device)
        return _normalise(rotation[rows, :, axis])

    def _eval_residual(self, camera, state, degree: int):
        if state.method == "dc":
            # The single trainable value is a rendered, view-invariant RGB
            # residual.  This keeps the baseline's useful colour range equal
            # to residual_limit instead of shrinking it by SH constant C0.
            return state.residual()[:, 0]
        coefficients = state.residual().transpose(1, 2)
        directions = self.directions(camera, state.selected_ids)
        return self._eval_sh(degree, coefficients, directions)

    def render_stylized(
        self, camera, state, environment: EnvironmentMap, render_mode: str = "pbr",
    ):
        if render_mode not in {"pbr", "diffuse_only"}:
            raise ValueError("render_mode must be 'pbr' or 'diffuse_only'")
        original = self.render_original(camera)
        if state.method in {"dc", "full_sh"}:
            colors = self.native_radiance(camera)
            degree = 0 if state.method == "dc" else int(self.gaussians.active_sh_degree)
            edited = colors[state.selected_ids] + self._eval_residual(camera, state, degree)
            colors = colors.index_copy(0, state.selected_ids, edited.clamp(0, 1))
            return self.render_colors(camera, colors).clamp(0, 1)

        mask = self.segment_mask(camera, state.selected_ids)
        divisor = mask.permute(1, 2, 0)
        albedo = self._sparse_map(camera, state.selected_ids, state.selected_albedo(), divisor).clamp(0, 1)
        roughness = self._sparse_map(
            camera, state.selected_ids, state.selected_roughness().expand(-1, 3), divisor,
        )[..., :1].clamp(0.04, 1)
        metallic = self._sparse_map(
            camera, state.selected_ids, state.selected_metallic().expand(-1, 3), divisor,
        )[..., :1].clamp(0, 1)

        normals_g = self.gaussian_normals(camera, state.selected_ids)
        normals = self._sparse_map(camera, state.selected_ids, (normals_g + 1) * 0.5, divisor)
        normals = _normalise(normals * 2 - 1)
        view_g = -self.directions(camera, state.selected_ids)
        view = _normalise(self._sparse_map(camera, state.selected_ids, (view_g + 1) * 0.5, divisor) * 2 - 1)
        incident = -view
        reflected = _normalise(incident - 2 * (incident * normals).sum(-1, keepdim=True) * normals)
        env = environment.to(albedo.device)
        # Diffuse illumination is white-balanced and achromatic: environment
        # color belongs to highlights, not to the intrinsic stylized albedo.
        diffuse_target = state.pbr_diffuse_white if state.pbr_diffuse_white > 0 else None
        diffuse_light = env.diffuse_sample(
            normals, torch.full_like(roughness, 0.85),
            target_luminance=diffuse_target,
        )
        specular_light = env.sample(reflected, roughness)
        ndotv = (normals * view).sum(-1, keepdim=True).clamp(0, 1)
        if render_mode == "diffuse_only":
            surface = diffuse_light * albedo
        else:
            f0 = 0.04 * (1 - metallic) + albedo * metallic
            fresnel = f0 + (1 - f0) * (1 - ndotv).pow(5)
            surface = diffuse_light * albedo * (1 - metallic) * (1 - fresnel)
            surface = surface + specular_light * fresnel
        surface = calibrated_tone_map(
            surface, state.pbr_exposure, state.pbr_white_point,
        )

        if state.method == "ours":
            # A small high-pass signal from the UV field is added after PBR so
            # painterly strokes survive warm/cool lighting and tone mapping.
            detail_map = self._sparse_map(
                camera, state.selected_ids, state.selected_detail(), divisor,
            )
            surface = surface + detail_map
            residual = self._eval_residual(camera, state, state.residual_degree)
            residual_map = self._sparse_map(camera, state.selected_ids, residual, divisor)
            surface = (surface + residual_map).clamp(0, 1)

        mask_hwc = mask.permute(1, 2, 0)
        result = original.permute(1, 2, 0) * (1 - mask_hwc) + surface * mask_hwc
        return result.permute(2, 0, 1).clamp(0, 1)

    def render_albedo(self, camera, state):
        """Composite selected albedo directly, bypassing PBR and SH for UV diagnostics."""
        original = self.render_original(camera)
        mask = self.segment_mask(camera, state.selected_ids)
        albedo = self._sparse_map(
            camera, state.selected_ids, state.selected_albedo(), mask.permute(1, 2, 0),
        ).clamp(0, 1).permute(2, 0, 1)
        return original * (1 - mask) + albedo * mask
