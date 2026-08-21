from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .atlas import ATLAS_VERSION, AtlasTopology


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


class AtlasTextureField(nn.Module):
    """Packed, chart-aware surface atlas with small trainable UV corrections."""

    def __init__(
        self,
        xyz: torch.Tensor,
        normals: torch.Tensor | None = None,
        resolution=512,
        logit_limit=4.0,
        neighbours=8,
        target_charts=8,
        padding=4,
        feather=0.15,
        uv_offset_limit=0.03,
        source_layout="packed",
        reference_repeat=1,
        topology: AtlasTopology | None = None,
    ):
        super().__init__()
        if topology is None:
            topology = AtlasTopology.from_surface(
                xyz, normals, neighbours=neighbours, target_charts=target_charts,
                atlas_resolution=resolution, padding=padding, feather=feather,
            )
        elif len(topology.chart_ids) != len(xyz):
            raise ValueError("checkpoint atlas point count does not match selected segment")
        self.resolution = int(resolution)
        self.logit_limit = float(logit_limit)
        self.atlas_version = ATLAS_VERSION
        self.uv_offset_limit = float(uv_offset_limit)
        self.atlas_neighbours = int(neighbours)
        self.atlas_padding = int(padding)
        self.atlas_feather = float(feather)
        if source_layout not in {
            "packed", "developed", "component", "chart", "projected",
        }:
            raise ValueError(
                "atlas source layout must be 'packed', 'developed', 'component', "
                "'chart', or 'projected'"
            )
        self.source_layout = str(source_layout)
        self.reference_repeat = int(reference_repeat)
        if not 1 <= self.reference_repeat <= 16:
            raise ValueError("atlas reference repeat must be in [1, 16]")
        for name in (
            "chart_ids", "component_ids", "local_uv", "atlas_uv", "chart_layout", "chart_centers",
            "chart_axes", "chart_low", "chart_extent", "reference_regions", "edges",
            "source_transforms",
            "source_island_ids",
            "edge_3d_distance", "edge_uv_per_3d",
            "seam_edges", "triangles", "collision_pairs", "feather_uv", "feather_weight",
            "chart_cells",
            "seam_weight",
        ):
            self.register_buffer(name, getattr(topology, name))
        if self.source_layout == "packed":
            self._use_packed_source_layout_for_nonplanar_components()
        elif self.source_layout == "component":
            self._normalize_source_islands()
        elif self.source_layout == "chart":
            self._use_chart_local_source_layout()
        elif self.source_layout == "projected":
            self._use_global_projection_source_layout(xyz)
        self.uv_offset_raw = nn.Parameter(
            torch.zeros(topology.local_uv.shape, device=xyz.device, dtype=torch.float32),
        )
        self.logit_grid_raw = nn.Parameter(
            torch.zeros(1, 3, self.resolution, self.resolution, device=xyz.device),
        )
        self.register_buffer(
            "reference_logit_grid",
            torch.zeros(1, 3, self.resolution, self.resolution, device=xyz.device),
        )

    @torch.no_grad()
    def _use_packed_source_layout_for_nonplanar_components(self):
        """Keep the proven v6 source layout unless development is requested.

        Planar components retain one shared affine coordinate system. Curved
        components use their disjoint packed chart cells, avoiding the failed
        experimental cut-island composition as an implicit default.
        """
        chart_components = torch.empty(
            self.chart_count, dtype=torch.long, device=self.chart_ids.device,
        )
        for chart in range(self.chart_count):
            member = torch.where(self.chart_ids == chart)[0][0]
            chart_components[chart] = self.component_ids[member]
        for component in torch.unique(chart_components):
            charts = torch.where(chart_components == component)[0]
            root = int(charts[0])
            alignment = []
            for chart in charts:
                singular = torch.linalg.svdvals(
                    self.chart_axes[int(chart)].T @ self.chart_axes[root]
                )
                alignment.append(float(singular.min()))
            if min(alignment, default=1.0) >= 0.98:
                continue
            cells = self.chart_cells[charts]
            transforms = torch.zeros(
                len(charts), 2, 3, device=cells.device, dtype=cells.dtype,
            )
            transforms[:, 0, 0] = cells[:, 2] - cells[:, 0]
            transforms[:, 1, 1] = cells[:, 3] - cells[:, 1]
            transforms[:, 0, 2] = cells[:, 0]
            transforms[:, 1, 2] = cells[:, 1]
            self.source_transforms[charts] = transforms
            self.reference_regions[charts] = cells
            self.source_island_ids[charts] = charts

    @torch.no_grad()
    def _normalize_source_islands(self):
        """Give every disconnected surface component a full reference domain.

        Atlas storage remains disjoint through ``chart_layout``. Only reference
        sampling is normalized here, so a small disconnected component is not
        restricted to a tiny crop of the style image. Charts in one connected
        component retain their shared developed coordinates and seam phase.
        """
        self._normalize_source_groups(self.source_island_ids)

    @torch.no_grad()
    def _use_chart_local_source_layout(self):
        """Map every chart's normalized local UV directly to the reference."""
        self.source_transforms.zero_()
        self.source_transforms[:, 0, 0] = 1
        self.source_transforms[:, 1, 1] = 1
        self.reference_regions[:] = self.reference_regions.new_tensor((0, 0, 1, 1))
        self.source_island_ids.copy_(
            torch.arange(self.chart_count, device=self.source_island_ids.device)
        )

    @torch.no_grad()
    def _use_global_projection_source_layout(self, xyz: torch.Tensor):
        """Bake one continuous world-space planar reference field into all charts."""
        points = xyz.detach().float().to(self.local_uv.device)
        center = points.mean(dim=0)
        centered = points - center
        covariance = centered.T @ centered / max(len(points) - 1, 1)
        _, eigenvectors = torch.linalg.eigh(covariance)
        axes = torch.stack(
            (_orient_axis(eigenvectors[:, 2]), _orient_axis(eigenvectors[:, 1])), dim=1,
        )
        projected = centered @ axes
        low = torch.quantile(projected, 0.005, dim=0)
        high = torch.quantile(projected, 0.995, dim=0)
        target = ((projected - low) / (high - low).clamp_min(1e-7)).clamp(0, 1)
        homogeneous = torch.cat(
            (self.local_uv, torch.ones_like(self.local_uv[:, :1])), dim=1,
        )
        for chart in range(self.chart_count):
            ids = torch.where(self.chart_ids == chart)[0]
            solution = torch.linalg.lstsq(homogeneous[ids], target[ids]).solution
            self.source_transforms[chart] = solution.T
            values = target[ids]
            self.reference_regions[chart] = torch.cat(
                (values.amin(dim=0), values.amax(dim=0)), dim=0,
            )
        self.source_island_ids.zero_()

    @torch.no_grad()
    def _normalize_source_groups(self, chart_groups: torch.Tensor):
        """Normalize each chart group to a full unit-square reference domain."""
        for group in torch.unique(chart_groups):
            charts = torch.where(chart_groups == group)[0]
            if not charts.numel():
                continue
            regions = self.reference_regions[charts]
            low = regions[:, :2].amin(dim=0)
            high = regions[:, 2:].amax(dim=0)
            extent = (high - low).clamp_min(1e-7)
            transforms = self.source_transforms[charts].clone()
            transforms[:, :, :2] = transforms[:, :, :2] / extent.view(1, 2, 1)
            transforms[:, :, 2] = (
                transforms[:, :, 2] - low.view(1, 2)
            ) / extent.view(1, 2)
            self.source_transforms[charts] = transforms
            normalized = regions.clone()
            normalized[:, :2] = (regions[:, :2] - low) / extent
            normalized[:, 2:] = (regions[:, 2:] - low) / extent
            self.reference_regions[charts] = normalized.clamp(0, 1)

    @property
    def chart_count(self):
        return int(self.chart_layout.shape[0])

    def _pixel_bounds(self, rectangle: torch.Tensor):
        x0 = max(0, int(torch.floor(rectangle[0] * (self.resolution - 1)).item()))
        y0 = max(0, int(torch.floor(rectangle[1] * (self.resolution - 1)).item()))
        x1 = min(
            self.resolution,
            int(torch.ceil(rectangle[2] * (self.resolution - 1)).item()) + 1,
        )
        y1 = min(
            self.resolution,
            int(torch.ceil(rectangle[3] * (self.resolution - 1)).item()) + 1,
        )
        return x0, y0, x1, y1

    def _dilate_chart_padding(self, grid: torch.Tensor):
        result = grid.clone()
        for layout, cell in zip(self.chart_layout, self.chart_cells):
            x0, y0, x1, y1 = self._pixel_bounds(layout)
            cx0, cy0, cx1, cy1 = self._pixel_bounds(cell)
            if x1 <= x0 or y1 <= y0 or cx1 <= cx0 or cy1 <= cy0:
                continue
            source = grid[..., y0:y1, x0:x1]
            padded = F.pad(
                source,
                (x0 - cx0, cx1 - x1, y0 - cy0, cy1 - y1),
                mode="replicate",
            )
            result[..., cy0:cy1, cx0:cx1] = padded
        return result

    def logit_grid(self):
        bounded = self.logit_limit * torch.tanh(self.logit_grid_raw)
        return self._dilate_chart_padding(bounded)

    def current_local_uv(self):
        return (
            self.local_uv + self.uv_offset_limit * torch.tanh(self.uv_offset_raw)
        ).clamp(0, 1)

    def current_atlas_uv(self):
        layout = self.chart_layout[self.chart_ids]
        return layout[:, :2] + self.current_local_uv() * (layout[:, 2:] - layout[:, :2])

    def current_source_uv(self):
        """Reference-canvas coordinates sampled by every selected Gaussian."""
        local = self.current_local_uv()
        homogeneous = torch.cat((local, torch.ones_like(local[:, :1])), dim=1)
        return torch.einsum(
            "ni,nji->nj", homogeneous, self.source_transforms[self.chart_ids],
        ).clamp(0, 1)

    def current_reference_uv(self):
        """Reference-image coordinates after applying the explicit texture scale."""
        source = self.current_source_uv()
        if self.reference_repeat == 1:
            return source
        return torch.remainder(source * self.reference_repeat, 1.0)

    @staticmethod
    def _sample_at(values: torch.Tensor, uv: torch.Tensor):
        grid = uv.mul(2).sub(1).view(1, -1, 1, 2)
        sampled = F.grid_sample(
            values, grid, mode="bilinear", padding_mode="border", align_corners=True,
        )
        return sampled[0, :, :, 0].T

    def _sample_grid(self, values: torch.Tensor):
        primary = self._sample_at(values, self.current_atlas_uv())
        if not torch.any(self.feather_weight > 0):
            return primary
        secondary = self._sample_at(values, self.feather_uv)
        return primary * (1 - self.feather_weight) + secondary * self.feather_weight

    def sample(self):
        return self._sample_grid(self.logit_grid())

    def sample_detail(self, kernel_size=17):
        grid = self.logit_grid()
        low_frequency = F.avg_pool2d(
            grid, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2, count_include_pad=False,
        )
        return self._sample_grid(grid - low_frequency)

    @torch.no_grad()
    def initialize_from_reference(
        self, reference_chw: torch.Tensor, strength=0.6, absolute=False,
    ):
        reference = reference_chw.unsqueeze(0).clamp(1e-3, 1 - 1e-3)
        reference_mean = reference.mean((2, 3), keepdim=True)
        atlas = reference_mean.expand(
            1, 3, self.resolution, self.resolution,
        ).clone()
        for chart, layout in enumerate(self.chart_layout):
            x0 = max(0, int(torch.floor(layout[0] * (self.resolution - 1)).item()))
            y0 = max(0, int(torch.floor(layout[1] * (self.resolution - 1)).item()))
            x1 = min(self.resolution, int(torch.ceil(layout[2] * (self.resolution - 1)).item()) + 1)
            y1 = min(self.resolution, int(torch.ceil(layout[3] * (self.resolution - 1)).item()) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            local_y, local_x = torch.meshgrid(
                torch.linspace(0, 1, y1 - y0, device=reference.device),
                torch.linspace(0, 1, x1 - x0, device=reference.device),
                indexing="ij",
            )
            local = torch.stack((local_x, local_y, torch.ones_like(local_x)), dim=-1)
            source = torch.einsum("hwi,ji->hwj", local, self.source_transforms[chart])
            if self.reference_repeat > 1:
                source = torch.remainder(source * self.reference_repeat, 1.0)
            atlas[..., y0:y1, x0:x1] = F.grid_sample(
                reference, source.mul(2).sub(1).unsqueeze(0), mode="bilinear",
                padding_mode="border", align_corners=True,
            )
        atlas = self._dilate_chart_padding(atlas)
        logits = torch.log(atlas / (1 - atlas))
        occupied = torch.zeros_like(logits[:, :1])
        for layout in self.chart_layout:
            x0 = max(0, int(torch.floor(layout[0] * (self.resolution - 1)).item()))
            y0 = max(0, int(torch.floor(layout[1] * (self.resolution - 1)).item()))
            x1 = min(self.resolution, int(torch.ceil(layout[2] * (self.resolution - 1)).item()) + 1)
            y1 = min(self.resolution, int(torch.ceil(layout[3] * (self.resolution - 1)).item()) + 1)
            occupied[..., y0:y1, x0:x1] = 1
        if absolute:
            desired = logits
        else:
            mean = (logits * occupied).sum((2, 3), keepdim=True) / occupied.sum((2, 3), keepdim=True).clamp_min(1)
            desired = (logits - mean) * float(strength)
        if absolute:
            mean = (desired * occupied).sum((2, 3), keepdim=True) / occupied.sum((2, 3), keepdim=True).clamp_min(1)
            desired = mean + (desired - mean) * float(strength)
        desired = desired.clamp(-self.logit_limit * 0.98, self.logit_limit * 0.98)
        self.logit_grid_raw.copy_(torch.atanh(desired / self.logit_limit))
        self.reference_logit_grid.copy_(desired)

    def delta_regularization(self):
        delta = self.logit_grid() - self.reference_logit_grid
        anchor = delta.abs().mean()
        dx = (delta[..., 1:] - delta[..., :-1]).abs().mean()
        dy = (delta[..., 1:, :] - delta[..., :-1, :]).abs().mean()
        return anchor, dx + dy

    def geometric_losses(self):
        zero = self.logit_grid_raw.sum() * 0
        uv = self.current_local_uv()
        if self.edges.numel():
            left, right = self.edges[:, 0], self.edges[:, 1]
            current_delta = uv[left] - uv[right]
            continuity = (
                torch.tanh(self.uv_offset_raw[left])
                - torch.tanh(self.uv_offset_raw[right])
            ).square().sum(1).mean()
            current_distance = current_delta.norm(dim=1)
            base_distance = self.edge_uv_per_3d * self.edge_3d_distance
            positive = base_distance[base_distance > 0]
            distance_floor = (
                positive.median() * 0.05 if positive.numel() else base_distance.new_tensor(1e-6)
            )
            valid = base_distance > distance_floor.clamp_min(1e-7)
            if torch.any(valid):
                current_uv_per_3d = (
                    current_distance[valid] / self.edge_3d_distance[valid].clamp_min(1e-7)
                )
                target_uv_per_3d = self.edge_uv_per_3d[valid]
                distortion = (
                    (current_uv_per_3d - target_uv_per_3d)
                    / target_uv_per_3d.clamp_min(1e-6)
                ).square().mean()
            else:
                distortion = zero
        else:
            continuity = distortion = zero
        if self.seam_edges.numel():
            sampled = self._sample_at(self.logit_grid(), self.current_atlas_uv())
            seam = (
                sampled[self.seam_edges[:, 0]] - sampled[self.seam_edges[:, 1]]
            ).abs().mean()
        else:
            seam = zero
        if self.triangles.numel():
            a, b, c = self.triangles.T
            base_area = torch.linalg.cross(
                F.pad(self.local_uv[b] - self.local_uv[a], (0, 1)),
                F.pad(self.local_uv[c] - self.local_uv[a], (0, 1)),
                dim=-1,
            )[:, 2]
            current_area = torch.linalg.cross(
                F.pad(uv[b] - uv[a], (0, 1)), F.pad(uv[c] - uv[a], (0, 1)), dim=-1,
            )[:, 2]
            signed_ratio = current_area * base_area.sign() / base_area.abs().clamp_min(1e-7)
            foldover = F.relu(0.1 - signed_ratio).square().mean()
        else:
            foldover = zero
        if self.collision_pairs.numel():
            left, right = self.collision_pairs.T
            margin = 2.5 / max(self.resolution, 16)
            collision = F.relu(margin - (uv[left] - uv[right]).norm(dim=1)).square().mean()
        else:
            collision = zero
        return {
            "uv_continuity": continuity,
            "uv_distortion": distortion,
            "chart_seam": seam,
            "uv_foldover": foldover,
            "uv_collision": collision,
        }

    def diagnostics(self):
        losses = self.geometric_losses()
        local_uv = self.current_local_uv()
        atlas_uv = self.current_atlas_uv()
        current = torch.sigmoid(self.logit_grid())
        reference = torch.sigmoid(self.reference_logit_grid)
        current_gradient = (
            (current[..., 1:] - current[..., :-1]).abs().mean()
            + (current[..., 1:, :] - current[..., :-1, :]).abs().mean()
        )
        reference_gradient = (
            (reference[..., 1:] - reference[..., :-1]).abs().mean()
            + (reference[..., 1:, :] - reference[..., :-1, :]).abs().mean()
        )
        if self.collision_pairs.numel():
            left, right = self.collision_pairs.T
            margin = 2.5 / max(self.resolution, 16)
            active = (local_uv[left] - local_uv[right]).norm(dim=1) < margin
            colliding_points = (
                torch.unique(self.collision_pairs[active]).numel()
                if torch.any(active) else 0
            )
            intra_collision_rate = local_uv.new_tensor(
                float(colliding_points) / max(len(local_uv), 1),
            )
        else:
            intra_collision_rate = current_gradient * 0

        if self.triangles.numel():
            a, b, c = self.triangles.T
            base_a = self.local_uv[b] - self.local_uv[a]
            base_b = self.local_uv[c] - self.local_uv[a]
            current_a = local_uv[b] - local_uv[a]
            current_b = local_uv[c] - local_uv[a]
            base_area = base_a[:, 0] * base_b[:, 1] - base_a[:, 1] * base_b[:, 0]
            current_area = current_a[:, 0] * current_b[:, 1] - current_a[:, 1] * current_b[:, 0]
            foldover_rate = (
                current_area * base_area.sign() <= 0
            ).float().mean()
        else:
            foldover_rate = current_gradient * 0

        layout = self.chart_layout
        cells = self.chart_cells
        if len(layout) > 1:
            left = cells[:, None, :]
            right = cells[None, :, :]
            width = (
                torch.minimum(left[..., 2], right[..., 2])
                - torch.maximum(left[..., 0], right[..., 0])
            ).clamp_min(0)
            height = (
                torch.minimum(left[..., 3], right[..., 3])
                - torch.maximum(left[..., 1], right[..., 1])
            ).clamp_min(0)
            upper = torch.triu(torch.ones_like(width, dtype=torch.bool), diagonal=1)
            inter_chart_overlap = (width * height)[upper].sum().clamp(0, 1)
        else:
            inter_chart_overlap = current_gradient * 0
        atlas_occupancy = (
            (layout[:, 2] - layout[:, 0]).clamp_min(0)
            * (layout[:, 3] - layout[:, 1]).clamp_min(0)
        ).sum().clamp(0, 1)
        point_layout = layout[self.chart_ids]
        outside_inner = (
            (atlas_uv < point_layout[:, :2] - 1e-7)
            | (atlas_uv > point_layout[:, 2:] + 1e-7)
        ).any(1)
        pad = float(self.atlas_padding) / max(self.resolution, 1)
        cell_size = cells[:, 2:] - cells[:, :2]
        invalid_padding_chart = (cell_size <= 2 * pad + 1e-7).any(1)
        padding_violation = (
            outside_inner | invalid_padding_chart[self.chart_ids]
        ).float().mean()
        gradient_retention = torch.where(
            reference_gradient > 1e-6,
            current_gradient / reference_gradient.clamp_min(1e-7),
            torch.zeros_like(current_gradient),
        )
        source_uv = self.current_source_uv()
        if self.edges.numel():
            edge_left, edge_right = self.edges.T
            source_edge_distance = (
                source_uv[edge_left] - source_uv[edge_right]
            ).norm(dim=1)
            source_density = source_edge_distance / self.edge_3d_distance.clamp_min(1e-7)
            chart_density_sum = source_density.new_zeros(self.chart_count)
            chart_density_count = source_density.new_zeros(self.chart_count)
            edge_charts = self.chart_ids[edge_left]
            chart_density_sum.scatter_add_(0, edge_charts, source_density)
            chart_density_count.scatter_add_(
                0, edge_charts, torch.ones_like(source_density),
            )
            valid_density = chart_density_count > 0
            chart_density = (
                chart_density_sum[valid_density]
                / chart_density_count[valid_density].clamp_min(1)
            )
            source_texel_density_cv = (
                chart_density.std(unbiased=False) / chart_density.mean().clamp_min(1e-7)
                if chart_density.numel() > 1 else source_density.sum() * 0
            )
            source_edge_energy = source_edge_distance.mean()
        else:
            source_texel_density_cv = source_edge_energy = current_gradient * 0
        if self.seam_edges.numel():
            seam_left, seam_right = self.seam_edges.T
            source_seam_energy = (
                source_uv[seam_left] - source_uv[seam_right]
            ).norm(dim=1).mean()
            source_seam_ratio = source_seam_energy / source_edge_energy.clamp_min(1e-7)
        else:
            source_seam_energy = source_seam_ratio = current_gradient * 0
        return {
            # Keep the original name as an alias for old reports.
            "uv_collision_rate": intra_collision_rate,
            "intra_chart_collision_rate": intra_collision_rate,
            "inter_chart_overlap_rate": inter_chart_overlap,
            "uv_foldover_rate": foldover_rate,
            "atlas_occupancy": atlas_occupancy,
            "padding_violation_rate": padding_violation,
            "chart_seam_energy": losses["chart_seam"],
            "uv_distortion": losses["uv_distortion"],
            "reference_texture_gradient_retention": gradient_retention,
            "source_uv_seam_energy": source_seam_energy,
            "source_uv_seam_ratio": source_seam_ratio,
            "source_texel_density_cv": source_texel_density_cv,
        }

    def preview(
        self, global_shift: torch.Tensor, base_logit: torch.Tensor | None = None,
    ):
        base = 0 if base_logit is None else base_logit.view(1, 3, 1, 1)
        return torch.sigmoid(base + global_shift.view(1, 3, 1, 1) + self.logit_grid())
