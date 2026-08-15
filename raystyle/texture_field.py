from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _orient_axis(axis: torch.Tensor) -> torch.Tensor:
    dominant = axis.abs().argmax()
    return axis * torch.where(axis[dominant] < 0, -1.0, 1.0)


class PlanarTextureField(nn.Module):
    """PCA-plane UV parameterization and a differentiable 2D logit texture."""

    def __init__(self, xyz: torch.Tensor, resolution=256, logit_limit=4.0):
        super().__init__()
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3:
            raise ValueError("planar texture requires selected xyz with shape (N, 3), N >= 3")
        points = xyz.detach().float()
        center = points.mean(0)
        centered = points - center
        covariance = centered.T @ centered / max(len(points) - 1, 1)
        _, eigenvectors = torch.linalg.eigh(covariance)
        axes = torch.stack(
            (_orient_axis(eigenvectors[:, 2]), _orient_axis(eigenvectors[:, 1])), dim=1,
        )
        projected = centered @ axes
        low = torch.quantile(projected, 0.005, dim=0)
        high = torch.quantile(projected, 0.995, dim=0)
        extent = (high - low).clamp_min(1e-6)
        uv = ((projected - low) / extent).clamp(0, 1)

        self.resolution = int(resolution)
        self.logit_limit = float(logit_limit)
        self.register_buffer("plane_center", center)
        self.register_buffer("plane_axes", axes)
        self.register_buffer("uv_low", low)
        self.register_buffer("uv_extent", extent)
        self.register_buffer("selected_uv", uv)
        self.logit_grid_raw = nn.Parameter(
            torch.zeros(1, 3, self.resolution, self.resolution, device=xyz.device),
        )
        self.register_buffer(
            "reference_logit_grid",
            torch.zeros(1, 3, self.resolution, self.resolution, device=xyz.device),
        )

    def logit_grid(self):
        return self.logit_limit * torch.tanh(self.logit_grid_raw)

    def sample(self):
        return self._sample_grid(self.logit_grid())

    def _sample_grid(self, values: torch.Tensor):
        grid = self.selected_uv.mul(2).sub(1).view(1, -1, 1, 2)
        sampled = F.grid_sample(
            values, grid, mode="bilinear", padding_mode="border",
            align_corners=True,
        )
        return sampled[0, :, :, 0].T

    def sample_detail(self, kernel_size=17):
        """Sample a high-pass UV signal used as light-independent fine detail."""
        grid = self.logit_grid()
        low_frequency = F.avg_pool2d(
            grid, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
            count_include_pad=False,
        )
        return self._sample_grid(grid - low_frequency)

    @torch.no_grad()
    def initialize_from_reference(
        self, reference_chw: torch.Tensor, strength=0.6, absolute=False,
    ):
        reference = F.interpolate(
            reference_chw.unsqueeze(0), (self.resolution, self.resolution),
            mode="bilinear", align_corners=False,
        ).clamp(1e-3, 1 - 1e-3)
        logits = torch.log(reference / (1 - reference))
        mean = logits.mean((2, 3), keepdim=True)
        centered = logits - mean
        # Replacement mode stores an absolute albedo texture. Strength changes
        # only spatial contrast and therefore does not discard its colour base.
        desired = ((mean if absolute else 0) + centered * float(strength)).clamp(
            -self.logit_limit * 0.98, self.logit_limit * 0.98,
        )
        self.logit_grid_raw.copy_(torch.atanh(desired / self.logit_limit))
        self.reference_logit_grid.copy_(desired)

    def delta_regularization(self):
        delta = self.logit_grid() - self.reference_logit_grid
        anchor = delta.abs().mean()
        dx = (delta[..., 1:] - delta[..., :-1]).abs().mean()
        dy = (delta[..., 1:, :] - delta[..., :-1, :]).abs().mean()
        return anchor, dx + dy

    def preview(
        self, global_shift: torch.Tensor, base_logit: torch.Tensor | None = None,
    ):
        base = 0 if base_logit is None else base_logit.view(1, 3, 1, 1)
        return torch.sigmoid(
            base + global_shift.view(1, 3, 1, 1)
            + self.logit_grid()
        )


class TriPlanarTextureField(nn.Module):
    """PCA-aligned tri-planar texture field blended by Gaussian normals.

    Each selected Gaussian samples the two PCA coordinates perpendicular to
    its dominant normal.  Unlike one PCA plane, this keeps front/back and side
    surfaces from all being forced through the same projection.
    """

    _PLANE_DIMS = ((1, 2), (0, 2), (0, 1))

    def __init__(
        self,
        xyz: torch.Tensor,
        normals: torch.Tensor | None = None,
        resolution=256,
        logit_limit=4.0,
        blend_sharpness=4.0,
    ):
        super().__init__()
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3:
            raise ValueError("tri-planar texture requires xyz with shape (N, 3), N >= 3")
        points = xyz.detach().float()
        center = points.mean(0)
        centered = points - center
        covariance = centered.T @ centered / max(len(points) - 1, 1)
        _, eigenvectors = torch.linalg.eigh(covariance)
        axes = torch.stack(
            tuple(_orient_axis(eigenvectors[:, index]) for index in range(3)), dim=1,
        )
        local = centered @ axes
        low_3d = torch.quantile(local, 0.005, dim=0)
        high_3d = torch.quantile(local, 0.995, dim=0)
        extent_3d = (high_3d - low_3d).clamp_min(1e-6)
        uv = torch.stack([
            ((local[:, dims] - low_3d[list(dims)]) / extent_3d[list(dims)]).clamp(0, 1)
            for dims in self._PLANE_DIMS
        ])

        if normals is None:
            weights = torch.full(
                (len(points), 3), 1 / 3, device=points.device, dtype=points.dtype,
            )
        else:
            if normals.shape != points.shape:
                raise ValueError("tri-planar normals must have the same shape as xyz")
            local_normals = F.normalize(normals.detach().float(), dim=1, eps=1e-6) @ axes
            weights = local_normals.abs().pow(float(blend_sharpness))
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)

        self.resolution = int(resolution)
        self.logit_limit = float(logit_limit)
        self.blend_sharpness = float(blend_sharpness)
        self.register_buffer("plane_center", center)
        self.register_buffer("plane_axes", axes)
        self.register_buffer("axis_low", low_3d)
        self.register_buffer("axis_extent", extent_3d)
        self.register_buffer("selected_uv", uv)
        self.register_buffer("blend_weights", weights)
        self.logit_grid_raw = nn.Parameter(
            torch.zeros(3, 3, self.resolution, self.resolution, device=xyz.device),
        )
        self.register_buffer(
            "reference_logit_grid",
            torch.zeros(3, 3, self.resolution, self.resolution, device=xyz.device),
        )

    def logit_grid(self):
        return self.logit_limit * torch.tanh(self.logit_grid_raw)

    def _sample_planes(self, values: torch.Tensor):
        sampled = []
        for plane in range(3):
            grid = self.selected_uv[plane].mul(2).sub(1).view(1, -1, 1, 2)
            value = F.grid_sample(
                values[plane:plane + 1], grid, mode="bilinear",
                padding_mode="border", align_corners=True,
            )[0, :, :, 0].T
            sampled.append(value)
        stacked = torch.stack(sampled, dim=1)
        return (stacked * self.blend_weights[..., None]).sum(1)

    def sample(self):
        return self._sample_planes(self.logit_grid())

    def sample_detail(self, kernel_size=17):
        grid = self.logit_grid()
        low_frequency = F.avg_pool2d(
            grid, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2, count_include_pad=False,
        )
        return self._sample_planes(grid - low_frequency)

    @torch.no_grad()
    def initialize_from_reference(
        self, reference_chw: torch.Tensor, strength=0.6, absolute=False,
    ):
        reference = F.interpolate(
            reference_chw.unsqueeze(0), (self.resolution, self.resolution),
            mode="bilinear", align_corners=False,
        ).clamp(1e-3, 1 - 1e-3)
        logits = torch.log(reference / (1 - reference))
        mean = logits.mean((2, 3), keepdim=True)
        centered = logits - mean
        desired = ((mean if absolute else 0) + centered * float(strength)).clamp(
            -self.logit_limit * 0.98, self.logit_limit * 0.98,
        ).expand(3, -1, -1, -1).contiguous()
        self.logit_grid_raw.copy_(torch.atanh(desired / self.logit_limit))
        self.reference_logit_grid.copy_(desired)

    def delta_regularization(self):
        delta = self.logit_grid() - self.reference_logit_grid
        anchor = delta.abs().mean()
        dx = (delta[..., 1:] - delta[..., :-1]).abs().mean()
        dy = (delta[..., 1:, :] - delta[..., :-1, :]).abs().mean()
        return anchor, dx + dy

    def preview(
        self, global_shift: torch.Tensor, base_logit: torch.Tensor | None = None,
    ):
        base = 0 if base_logit is None else base_logit.view(1, 3, 1, 1)
        # All three charts start from the same reference. Their mean is a
        # compact viewer/debug representation and keeps the colour loss chart
        # balanced during optimization.
        logits = self.logit_grid().mean(0, keepdim=True)
        return torch.sigmoid(base + global_shift.view(1, 3, 1, 1) + logits)
