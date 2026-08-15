from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def load_segment(path: str | Path, segment_id: int, expected_points: int | None = None):
    """Read either a full GUI project NPZ or a single-segment PT export.

    GUI segment 1 is encoded as integer 2, segment 2 as 3, and so on.
    """
    source = Path(path).expanduser()
    if source.suffix.lower() == ".pt":
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if isinstance(payload, dict):
            payload = payload.get("mask", payload.get("segment_mask"))
        if not isinstance(payload, torch.Tensor):
            raise ValueError(f"{source} does not contain a tensor mask")
        selected = payload.detach().cpu().squeeze().bool()
        if selected.ndim != 1:
            raise ValueError(f"single-segment mask must be one-dimensional, got {tuple(selected.shape)}")
        if expected_points is not None and selected.numel() != expected_points:
            raise ValueError(
                f"segment mask has {selected.numel()} Gaussians but the scene has {expected_points}"
            )
        if not bool(selected.any()):
            raise ValueError(f"single-segment mask is empty: {source}")
        metadata = {
            "format": "single_segment_pt",
            "source": str(source.resolve()),
            "selected_gaussians": int(selected.sum()),
        }
        return selected, metadata

    if source.suffix.lower() != ".npz":
        raise ValueError(f"segment path must end in .pt or .npz, got {source}")
    with np.load(source, allow_pickle=False) as archive:
        if "mask" not in archive.files:
            raise ValueError(f"{source} does not contain a 'mask' entry")
        mask = np.asarray(archive["mask"], dtype=np.int64)
        metadata = {}
        if "metadata" in archive.files:
            metadata = json.loads(str(archive["metadata"].item()))
    if mask.ndim != 1:
        raise ValueError(f"project mask must be one-dimensional, got {mask.shape}")
    if expected_points is not None and mask.size != expected_points:
        raise ValueError(
            f"project mask has {mask.size} Gaussians but the scene has {expected_points}"
        )
    selected = torch.from_numpy(mask == (int(segment_id) + 1))
    if not bool(selected.any()):
        present = sorted(int(v - 1) for v in np.unique(mask) if v >= 2)
        raise ValueError(f"segment {segment_id} is empty; available GUI segment ids: {present}")
    return selected, metadata


def segment_inventory(path: str | Path):
    source = Path(path).expanduser()
    if source.suffix.lower() == ".pt":
        selected, metadata = load_segment(source, segment_id=1)
        return [{
            "format": metadata["format"],
            "segment_id": 1,
            "gaussians": int(selected.sum()),
            "point_count": int(selected.numel()),
        }]
    with np.load(source, allow_pickle=False) as archive:
        mask = np.asarray(archive["mask"], dtype=np.int64)
    return [
        {"format": "project_npz", "segment_id": int(encoded - 1),
         "gaussians": int((mask == encoded).sum()), "point_count": int(mask.size)}
        for encoded in np.unique(mask) if encoded >= 2
    ]
