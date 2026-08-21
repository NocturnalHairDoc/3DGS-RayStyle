from __future__ import annotations

import torch


def _oriented(axis: torch.Tensor) -> torch.Tensor:
    dominant = int(axis.abs().argmax())
    return axis if axis[dominant] >= 0 else -axis


def _pca_axes(points: torch.Tensor) -> torch.Tensor:
    centered = points.double() - points.double().mean(0, keepdim=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    _, axes = torch.linalg.eigh(covariance)
    axes = axes[:, [2, 1, 0]]
    return torch.stack([_oriented(axes[:, index]) for index in range(3)], dim=1)


def _mask_from_local_ids(
    point_count: int, selected_ids: torch.Tensor, local_ids: torch.Tensor,
) -> torch.Tensor:
    mask = torch.zeros(point_count, dtype=torch.bool)
    mask[selected_ids[local_ids]] = True
    return mask


def build_segment_stress_masks(
    xyz: torch.Tensor,
    normals: torch.Tensor,
    selected: torch.Tensor,
    *,
    thin_fraction: float = 0.10,
    curved_fraction: float = 0.20,
    distant_fraction: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Create deterministic adjacent, thin, curved and separated stress masks."""
    xyz = xyz.detach().float().cpu()
    normals = normals.detach().float().cpu()
    selected = selected.detach().bool().cpu().flatten()
    if xyz.shape != normals.shape or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz and normals must have matching shape (N, 3)")
    if len(selected) != len(xyz):
        raise ValueError("selected mask length must match xyz")
    selected_ids = torch.where(selected)[0]
    if len(selected_ids) < 16:
        raise ValueError("stress-mask construction requires at least 16 selected points")
    points = xyz[selected_ids]
    point_normals = torch.nn.functional.normalize(normals[selected_ids], dim=1)
    axes = _pca_axes(points)
    centered = points.double() - points.double().median(0).values
    projected = centered @ axes

    primary_order = torch.argsort(projected[:, 0], stable=True)
    middle = len(primary_order) // 2
    adjacent_a = _mask_from_local_ids(len(xyz), selected_ids, primary_order[:middle])
    adjacent_b = _mask_from_local_ids(len(xyz), selected_ids, primary_order[middle:])

    thin_count = max(3, min(len(selected_ids), round(len(selected_ids) * thin_fraction)))
    thin_order = torch.argsort(projected[:, 1].abs(), stable=True)
    thin_band = _mask_from_local_ids(len(xyz), selected_ids, thin_order[:thin_count])

    normal_covariance = point_normals.double().T @ point_normals.double()
    _, normal_axes = torch.linalg.eigh(normal_covariance)
    dominant_normal = _oriented(normal_axes[:, -1])
    curvature_score = 1.0 - (point_normals.double() @ dominant_normal).abs()
    curved_count = max(3, min(len(selected_ids), round(len(selected_ids) * curved_fraction)))
    curved_order = torch.argsort(curvature_score, descending=True, stable=True)
    nonplanar = _mask_from_local_ids(len(xyz), selected_ids, curved_order[:curved_count])

    tail_count = max(3, min(len(selected_ids) // 2, round(len(selected_ids) * distant_fraction)))
    distant_a = _mask_from_local_ids(len(xyz), selected_ids, primary_order[:tail_count])
    distant_b = _mask_from_local_ids(len(xyz), selected_ids, primary_order[-tail_count:])
    return {
        "adjacent_a": adjacent_a,
        "adjacent_b": adjacent_b,
        "thin_band": thin_band,
        "nonplanar": nonplanar,
        "distant_a": distant_a,
        "distant_b": distant_b,
    }


def stress_mask_diagnostics(
    selected: torch.Tensor, masks: dict[str, torch.Tensor],
) -> dict:
    selected = selected.detach().bool().cpu().flatten()
    adjacent_overlap = masks["adjacent_a"] & masks["adjacent_b"]
    adjacent_union = masks["adjacent_a"] | masks["adjacent_b"]
    distant_overlap = masks["distant_a"] & masks["distant_b"]
    return {
        "source_count": int(selected.sum()),
        "counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "all_subsets_of_source": all(
            not bool((mask & ~selected).any()) for mask in masks.values()
        ),
        "adjacent_pair_disjoint": not bool(adjacent_overlap.any()),
        "adjacent_pair_covers_source": bool(torch.equal(adjacent_union, selected)),
        "distant_pair_disjoint": not bool(distant_overlap.any()),
    }
