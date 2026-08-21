from __future__ import annotations

import torch
from torch import nn

from .atlas import AtlasTopology
from .config import METHODS
from .texture_field import AtlasTextureField, PlanarTextureField, TriPlanarTextureField


def _logit(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp(1e-4, 1 - 1e-4)
    return torch.log(values / (1 - values))


class StyleState(nn.Module):
    """Only selected appearance parameters are trainable; geometry stays external/frozen."""

    def __init__(
        self,
        base_albedo: torch.Tensor,
        selected: torch.Tensor,
        method: str,
        original_sh_degree: int = 3,
        residual_degree: int = 1,
        residual_limit: float = 0.08,
        global_shift_limit: float = 0.7,
        detail_residual_limit: float = 0.08,
        selected_xyz: torch.Tensor | None = None,
        selected_normals: torch.Tensor | None = None,
        selected_visibility: torch.Tensor | None = None,
        texture_resolution: int = 256,
        texture_logit_limit: float = 4.0,
        texture_mapping: str = "triplanar",
        atlas_charts: int = 8,
        atlas_neighbours: int = 8,
        atlas_padding: int = 4,
        atlas_feather: float = 0.15,
        atlas_uv_offset_limit: float = 0.03,
        atlas_source_layout: str = "packed",
        atlas_reference_repeat: int = 1,
        atlas_topology: AtlasTopology | None = None,
        albedo_mode: str = "replacement",
        pbr_diffuse_white: float = 1.0,
        pbr_exposure: float = 0.0,
        pbr_white_point: float = 1.0,
    ):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
        selected = selected.bool().to(base_albedo.device)
        ids = torch.where(selected)[0]
        self.method = method
        self.residual_degree = residual_degree
        self.residual_limit = float(residual_limit)
        self.global_shift_limit = float(global_shift_limit)
        self.detail_residual_limit = float(detail_residual_limit)
        if texture_mapping not in {"planar", "triplanar", "atlas"}:
            raise ValueError(f"unknown texture mapping {texture_mapping!r}")
        self.texture_mapping = texture_mapping
        self.pbr_diffuse_white = float(pbr_diffuse_white)
        self.pbr_exposure = float(pbr_exposure)
        self.pbr_white_point = float(pbr_white_point)
        if albedo_mode not in {"replacement", "additive"}:
            raise ValueError(f"unknown albedo mode {albedo_mode!r}")
        self.albedo_mode = albedo_mode
        self.register_buffer("selected_ids", ids)
        self.register_buffer("base_albedo", base_albedo.detach().clone())
        if selected_visibility is None:
            selected_visibility = torch.ones(len(ids), device=base_albedo.device)
        if selected_visibility.numel() != len(ids):
            raise ValueError("selected_visibility must contain one value per selected Gaussian")
        self.register_buffer(
            "selected_visibility",
            selected_visibility.detach().flatten().to(base_albedo.device).clamp(0, 1),
            persistent=False,
        )
        selected_base = base_albedo[ids].detach().clamp(1e-4, 1 - 1e-4)
        self.albedo_logits = nn.Parameter(_logit(selected_base))
        self.global_albedo_shift = nn.Parameter(torch.zeros(1, 3, device=base_albedo.device))
        self.roughness_logits = nn.Parameter(torch.zeros(len(ids), 1))
        self.metallic_logits = nn.Parameter(torch.full((len(ids), 1), -3.0))

        if method == "dc":
            coefficient_count = 1
        elif method == "full_sh":
            coefficient_count = (original_sh_degree + 1) ** 2
        elif method == "ours":
            coefficient_count = (residual_degree + 1) ** 2
        else:
            coefficient_count = 0
        self.sh_residual = nn.Parameter(torch.zeros(len(ids), coefficient_count, 3))

        self.texture_field = None
        if method == "ours":
            if selected_xyz is None:
                raise ValueError("ours requires selected_xyz for the planar texture field")
            if texture_mapping == "atlas":
                self.texture_field = AtlasTextureField(
                    selected_xyz, selected_normals, texture_resolution,
                    texture_logit_limit, atlas_neighbours, atlas_charts,
                    atlas_padding, atlas_feather, atlas_uv_offset_limit,
                    atlas_source_layout, atlas_reference_repeat, atlas_topology,
                )
            elif texture_mapping == "triplanar":
                self.texture_field = TriPlanarTextureField(
                    selected_xyz, selected_normals, texture_resolution,
                    texture_logit_limit,
                )
            else:
                self.texture_field = PlanarTextureField(
                    selected_xyz, texture_resolution, texture_logit_limit,
                )
            self.albedo_logits.requires_grad_(False)
        else:
            self.global_albedo_shift.requires_grad_(False)

        if method in {"dc", "full_sh"}:
            self.roughness_logits.requires_grad_(False)
            self.metallic_logits.requires_grad_(False)
        if method in {"dc", "full_sh"}:
            self.albedo_logits.requires_grad_(False)

    @property
    def selected_count(self) -> int:
        return int(self.selected_ids.numel())

    def selected_albedo(self):
        if self.method == "ours":
            logits = self.texture_field.sample() + self.bounded_global_shift()
            if self.albedo_mode == "additive":
                logits = logits + self.albedo_logits.detach()
            return torch.sigmoid(logits)
        return torch.sigmoid(self.albedo_logits)

    def bounded_global_shift(self):
        return self.global_shift_limit * torch.tanh(self.global_albedo_shift)

    def selected_detail(self):
        if self.texture_field is None or self.detail_residual_limit == 0:
            return self.albedo_logits.new_zeros(self.selected_count, 3)
        return self.detail_residual_limit * torch.tanh(self.texture_field.sample_detail())

    def selected_roughness(self):
        return 0.04 + 0.96 * torch.sigmoid(self.roughness_logits)

    def selected_metallic(self):
        return torch.sigmoid(self.metallic_logits)

    def residual(self):
        return self.residual_limit * torch.tanh(self.sh_residual)

    @staticmethod
    def _balanced_graph_parts(parts: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate fields while giving each semantic field equal L1 weight."""
        flat = [part.flatten(1) for part in parts]
        total_width = sum(part.shape[1] for part in flat)
        group_count = len(flat)
        scaled = [
            part * (total_width / (group_count * part.shape[1]))
            for part in flat
        ]
        return torch.cat(scaled, dim=1)

    def graph_values(self, scope: str = "appearance"):
        """Per-Gaussian values regularized by the spatial anchor graph.

        ``appearance`` covers every spatially varying trainable appearance
        field. For ``ours`` this is albedo, light-independent detail, PBR
        material, and the low-order SH residual. ``material`` reproduces the
        former Ours behavior and restricts it to roughness and metallic.
        """
        if scope not in {"appearance", "material"}:
            raise ValueError("graph scope must be 'appearance' or 'material'")
        parts = []
        if self.method == "ours":
            if scope == "appearance":
                parts.extend((self.selected_albedo(), self.selected_detail()))
            parts.extend((self.selected_roughness(), self.selected_metallic()))
            if scope == "appearance" and self.sh_residual.numel():
                parts.append(self.residual())
        elif self.method == "pbr_only":
            parts.extend((self.selected_albedo(), self.selected_roughness(), self.selected_metallic()))
        if self.sh_residual.numel() and self.method != "ours":
            parts.append(self.residual())
        return self._balanced_graph_parts(parts)

    def material_prior(self):
        if self.method not in {"ours", "pbr_only"}:
            return self.sh_residual.sum() * 0
        extreme = torch.relu(0.08 - self.selected_roughness()).mean()
        if self.method == "ours" and self.albedo_mode == "replacement":
            # A replacement texture must not be pulled back toward the old
            # material. Only discourage needless global colour compensation.
            return 0.05 * self.bounded_global_shift().abs().mean() + extreme
        albedo_delta = (
            self.selected_albedo() - self.base_albedo[self.selected_ids]
        ).abs().mean()
        return 0.25 * albedo_delta + extreme

    def texture_regularization(self):
        if self.texture_field is None:
            zero = self.albedo_logits.sum() * 0
            return zero, zero
        return self.texture_field.delta_regularization()

    def atlas_regularization(self):
        if not isinstance(self.texture_field, AtlasTextureField):
            zero = self.albedo_logits.sum() * 0
            return {
                name: zero for name in (
                    "uv_continuity", "uv_distortion", "chart_seam",
                    "uv_foldover", "uv_collision",
                )
            }
        losses = self.texture_field.geometric_losses()
        seam = self._weighted_seam_statistics()
        if seam is not None:
            losses["chart_seam"] = seam["excess"]
        return losses

    def _weighted_seam_statistics(self, albedo: torch.Tensor | None = None):
        """Measure boundary jumps relative to the local surface gradient.

        A raw colour difference across a chart boundary is not necessarily a
        seam: a continuous gradient has the same difference across ordinary
        within-chart edges.  Estimate the local expected variation from the
        two charts and penalize only the unexplained boundary excess.
        """
        if not isinstance(self.texture_field, AtlasTextureField):
            return None
        if not self.texture_field.seam_edges.numel():
            zero = self.albedo_logits.sum() * 0
            return {"raw": zero, "surface_gradient": zero, "excess": zero}
        if albedo is None:
            albedo = self.selected_albedo()
        edges = self.texture_field.edges
        local_sum = albedo.new_zeros(self.selected_count)
        local_weight = albedo.new_zeros(self.selected_count)
        if edges.numel():
            edge_left, edge_right = edges.T
            edge_visibility = torch.sqrt(
                self.selected_visibility[edge_left]
                * self.selected_visibility[edge_right]
            )
            edge_difference = (
                albedo[edge_left] - albedo[edge_right]
            ).abs().mean(1)
            weighted_difference = edge_difference * edge_visibility
            local_sum.scatter_add_(0, edge_left, weighted_difference)
            local_sum.scatter_add_(0, edge_right, weighted_difference)
            local_weight.scatter_add_(0, edge_left, edge_visibility)
            local_weight.scatter_add_(0, edge_right, edge_visibility)
        local_gradient = local_sum / local_weight.clamp_min(1e-7)
        left, right = self.texture_field.seam_edges.T
        visibility = torch.sqrt(
            self.selected_visibility[left] * self.selected_visibility[right]
        )
        weight = self.texture_field.seam_weight.flatten() * visibility
        difference = (albedo[left] - albedo[right]).abs().mean(1)
        expected = 0.5 * (local_gradient[left] + local_gradient[right])
        denominator = weight.sum().clamp_min(1e-7)
        return {
            "raw": (difference * weight).sum() / denominator,
            "surface_gradient": (expected * weight).sum() / denominator,
            # Mutual-kNN edges do not all have exactly the same length.  The
            # modest tolerance prevents normal gradient variation from being
            # mislabeled as a discontinuity while retaining a hinge on jumps.
            "excess": (torch.relu(difference - 1.5 * expected) * weight).sum()
            / denominator,
        }

    def atlas_diagnostics(self):
        if not isinstance(self.texture_field, AtlasTextureField):
            return {}
        diagnostics = self.texture_field.diagnostics()
        seam = self._weighted_seam_statistics()
        diagnostics["chart_seam_energy"] = seam["excess"]
        diagnostics["chart_seam_raw_energy"] = seam["raw"]
        diagnostics["surface_gradient_energy"] = seam["surface_gradient"]
        return diagnostics

    @torch.no_grad()
    def initialize_texture(self, reference_chw: torch.Tensor, strength=0.6):
        if self.texture_field is None:
            return
        replacement = self.albedo_mode == "replacement"
        self.texture_field.initialize_from_reference(
            reference_chw, strength, absolute=replacement,
        )
        if replacement:
            self.global_albedo_shift.zero_()
            return
        reference_mean = reference_chw.mean((1, 2)).clamp(1e-3, 1 - 1e-3)
        base_mean = self.base_albedo[self.selected_ids].mean(0).clamp(1e-3, 1 - 1e-3)
        shift = _logit(reference_mean) - _logit(base_mean)
        desired = shift.view(1, 3).clamp(
            -self.global_shift_limit * 0.98, self.global_shift_limit * 0.98,
        )
        self.global_albedo_shift.copy_(
            torch.atanh(desired / self.global_shift_limit)
        )

    def texture_preview(self):
        if self.texture_field is None:
            return None
        base_logit = None
        if self.albedo_mode == "additive":
            base_mean = self.base_albedo[self.selected_ids].mean(0).clamp(1e-3, 1 - 1e-3)
            base_logit = _logit(base_mean)
        return self.texture_field.preview(
            self.bounded_global_shift(), base_logit=base_logit,
        )[0]

    def checkpoint_metadata(self):
        metadata = {
            "method": self.method,
            "selected_count": self.selected_count,
            "residual_degree": self.residual_degree,
            "residual_limit": self.residual_limit,
            "global_shift_limit": self.global_shift_limit,
            "detail_residual_limit": self.detail_residual_limit,
            "albedo_mode": self.albedo_mode,
            "texture_mapping": self.texture_mapping,
            "pbr_diffuse_white": self.pbr_diffuse_white,
            "pbr_exposure": self.pbr_exposure,
            "pbr_white_point": self.pbr_white_point,
            "coefficient_count": int(self.sh_residual.shape[1]),
            "texture_resolution": (
                self.texture_field.resolution if self.texture_field is not None else None
            ),
        }
        if isinstance(self.texture_field, AtlasTextureField):
            metadata.update({
                "atlas_version": self.texture_field.atlas_version,
                "atlas_chart_count": self.texture_field.chart_count,
                "atlas_neighbours": self.texture_field.atlas_neighbours,
                "atlas_padding": self.texture_field.atlas_padding,
                "atlas_feather": self.texture_field.atlas_feather,
                "atlas_uv_offset_limit": self.texture_field.uv_offset_limit,
                "atlas_source_layout": self.texture_field.source_layout,
                "atlas_reference_repeat": self.texture_field.reference_repeat,
                "atlas_state_keys": {
                    "chart_id": "texture_field.chart_ids",
                    "uv": "texture_field.local_uv",
                    "packed_uv": "texture_field.atlas_uv",
                    "layout": "texture_field.chart_layout",
                    "reference_regions": "texture_field.reference_regions",
                },
            })
        return metadata

    @torch.no_grad()
    def load_checkpoint_state(self, state_dict):
        incompatible = self.load_state_dict(state_dict, strict=False)
        allowed_missing = {
            "texture_field.reference_logit_grid",
            # Atlas v1 checkpoints predate explicit surface-component IDs.
            "texture_field.component_ids",
            # Atlas v1/v2 checkpoints stored only the padded inner layout.
            "texture_field.chart_cells",
            # Atlas v1-v3 checkpoints predate distance-weighted seam pairs.
            "texture_field.seam_weight",
            # Atlas v6 and earlier derived reference sampling from rectangles.
            "texture_field.source_transforms",
            "texture_field.source_island_ids",
        }
        unexpected_missing = set(incompatible.missing_keys) - allowed_missing
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "incompatible RayStyle checkpoint: missing="
                f"{sorted(unexpected_missing)}, unexpected="
                f"{sorted(incompatible.unexpected_keys)}"
            )
        if (
            self.texture_field is not None
            and "texture_field.reference_logit_grid" in incompatible.missing_keys
        ):
            self.texture_field.reference_logit_grid.copy_(
                self.texture_field.logit_grid()
            )
