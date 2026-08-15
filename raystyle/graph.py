from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree


@dataclass
class AnchorGraph:
    point_to_anchor: torch.Tensor
    edges: torch.Tensor
    weights: torch.Tensor
    anchor_count: int

    @classmethod
    def from_points(cls, xyz: torch.Tensor, target_anchors=512, neighbours=8):
        points = xyz.detach().float().cpu().numpy()
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("xyz must have shape (N, 3), N > 0")
        target = min(max(1, int(target_anchors)), len(points))
        extent = np.maximum(np.ptp(points, axis=0), 1e-7)
        voxel = float(np.cbrt(np.prod(extent) / target))
        voxel = max(voxel, float(np.linalg.norm(extent)) * 1e-5)
        keys = np.floor((points - points.min(0)) / voxel).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        count = int(inverse.max()) + 1
        sizes = np.bincount(inverse, minlength=count)
        centers = np.zeros((count, 3), dtype=np.float32)
        np.add.at(centers, inverse, points)
        centers /= sizes[:, None]
        pairs: set[tuple[int, int]] = set()
        if count > 1:
            k = min(count, max(2, int(neighbours) + 1))
            distances, indices = cKDTree(centers).query(centers, k=k)
            distances = np.atleast_2d(distances)
            indices = np.atleast_2d(indices)
            for left in range(count):
                for right in indices[left, 1:]:
                    right = int(right)
                    pairs.add((min(left, right), max(left, right)))
        edges = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
        if len(edges):
            distance = np.linalg.norm(centers[edges[:, 0]] - centers[edges[:, 1]], axis=1)
            weights = np.exp(-distance / max(voxel * 2.5, 1e-6)).astype(np.float32)
        else:
            weights = np.empty(0, dtype=np.float32)
        return cls(
            torch.from_numpy(inverse.astype(np.int64)),
            torch.from_numpy(edges),
            torch.from_numpy(weights),
            count,
        )

    def regularize(self, point_values: torch.Tensor) -> torch.Tensor:
        if self.edges.numel() == 0:
            return point_values.sum() * 0
        assignment = self.point_to_anchor.to(point_values.device)
        edges = self.edges.to(point_values.device)
        weights = self.weights.to(point_values.device)
        flat = point_values.flatten(1)
        sums = torch.zeros(self.anchor_count, flat.shape[1], device=flat.device, dtype=flat.dtype)
        sums.index_add_(0, assignment, flat)
        counts = torch.bincount(assignment, minlength=self.anchor_count).to(flat.dtype).clamp_min(1)
        means = sums / counts[:, None]
        delta = (means[edges[:, 0]] - means[edges[:, 1]]).abs().mean(1)
        return (delta * weights).sum() / weights.sum().clamp_min(1e-6)

