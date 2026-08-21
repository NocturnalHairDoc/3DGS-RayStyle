from __future__ import annotations

import torch
import torch.nn.functional as F


def composite_independent_edits(
    original: torch.Tensor, edited_images: list[torch.Tensor],
) -> torch.Tensor:
    """Combine disjoint Gaussian edits as additive deltas from one original render."""
    result = original.clone()
    for edited in edited_images:
        if edited.shape != original.shape:
            raise ValueError("all edited renders must match the original image shape")
        result = result + edited - original
    return result.clamp(0, 1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    selected = values[mask]
    return float(selected.mean()) if selected.numel() else None


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    selected = values[mask]
    return float(selected.max()) if selected.numel() else None


def region_change_metrics(
    before: torch.Tensor,
    after: torch.Tensor,
    own_mask: torch.Tensor,
    other_mask: torch.Tensor,
    *,
    core_threshold: float = 0.5,
    exclusion_threshold: float = 0.02,
    boundary_radius: int = 8,
) -> dict[str, float | int | None]:
    """Measure intended, cross-segment, boundary and far-field image changes.

    ``outside`` retains the strict definition used by the original diagnostic.
    It includes sub-threshold Gaussian tails. ``near_boundary`` and
    ``far_outside`` split that set spatially so a projected splat footprint is
    not mistaken for contamination of a remote surface.
    """
    if before.shape != after.shape or before.ndim != 3:
        raise ValueError("before and after must have matching CHW image shapes")
    own = own_mask.squeeze().float()
    other = other_mask.squeeze().float()
    if own.shape != before.shape[1:] or other.shape != before.shape[1:]:
        raise ValueError("segment masks must match the image spatial shape")
    if boundary_radius < 0:
        raise ValueError("boundary_radius must be non-negative")
    change = (after - before).abs().mean(0)
    own_core = (own >= core_threshold) & (other <= exclusion_threshold)
    other_core = (other >= core_threshold) & (own <= exclusion_threshold)
    shared_boundary = (own > exclusion_threshold) & (other > exclusion_threshold)
    outside = (own <= exclusion_threshold) & (other <= exclusion_threshold)
    visible_support = ((own > exclusion_threshold) | (other > exclusion_threshold)).float()
    if boundary_radius:
        kernel = 2 * boundary_radius + 1
        expanded_support = F.max_pool2d(
            visible_support[None, None], kernel, stride=1, padding=boundary_radius,
        )[0, 0].bool()
    else:
        expanded_support = visible_support.bool()
    near_boundary = outside & expanded_support
    far_outside = outside & ~expanded_support
    own_mean = _masked_mean(change, own_core)
    other_mean = _masked_mean(change, other_core)
    return {
        "own_core_pixels": int(own_core.sum()),
        "other_core_pixels": int(other_core.sum()),
        "shared_boundary_pixels": int(shared_boundary.sum()),
        "near_boundary_pixels": int(near_boundary.sum()),
        "far_outside_pixels": int(far_outside.sum()),
        "outside_pixels": int(outside.sum()),
        "own_core_change_mean": own_mean,
        "other_core_change_mean": other_mean,
        "other_core_change_max": _masked_max(change, other_core),
        "shared_boundary_change_mean": _masked_mean(change, shared_boundary),
        "near_boundary_change_mean": _masked_mean(change, near_boundary),
        "near_boundary_change_max": _masked_max(change, near_boundary),
        "far_outside_change_mean": _masked_mean(change, far_outside),
        "far_outside_change_max": _masked_max(change, far_outside),
        "outside_change_mean": _masked_mean(change, outside),
        "outside_change_max": _masked_max(change, outside),
        "cross_to_own_change_ratio": (
            other_mean / max(own_mean, 1e-12)
            if own_mean is not None and other_mean is not None else None
        ),
    }
