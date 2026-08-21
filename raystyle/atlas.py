from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from scipy.cluster.vq import kmeans2
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree


ATLAS_VERSION = 12


def _oriented(axis: np.ndarray) -> np.ndarray:
    dominant = int(np.argmax(np.abs(axis)))
    return axis if axis[dominant] >= 0 else -axis


def _limited_rows(values: np.ndarray, limit: int) -> np.ndarray:
    if len(values) <= limit:
        return values
    ids = np.linspace(0, len(values) - 1, limit, dtype=np.int64)
    return values[ids]


def _limited_indices(length: int, limit: int) -> np.ndarray:
    if length <= limit:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, limit, dtype=np.int64)


def _pack_weighted_rectangles(
    weights: np.ndarray,
    minimum_area_fraction: float = 0.0,
) -> np.ndarray:
    """Deterministic guillotine packing with area proportional to surface size."""
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 1.0)
    if minimum_area_fraction > 0:
        minimum_weight = weights.sum() * float(minimum_area_fraction)
        weights = np.maximum(weights, minimum_weight)
    layout = np.zeros((len(weights), 4), dtype=np.float32)

    def split(indices: np.ndarray, bounds: tuple[float, float, float, float]):
        if len(indices) == 1:
            layout[indices[0]] = bounds
            return
        values = weights[indices]
        cumulative = np.cumsum(values)
        cut = int(np.searchsorted(cumulative, cumulative[-1] * 0.5)) + 1
        cut = min(max(1, cut), len(indices) - 1)
        first, second = indices[:cut], indices[cut:]
        fraction = float(weights[first].sum() / weights[indices].sum())
        x0, y0, x1, y1 = bounds
        if (x1 - x0) >= (y1 - y0):
            middle = x0 + (x1 - x0) * fraction
            split(first, (x0, y0, middle, y1))
            split(second, (middle, y0, x1, y1))
        else:
            middle = y0 + (y1 - y0) * fraction
            split(first, (x0, y0, x1, middle))
            split(second, (x0, middle, x1, y1))

    split(np.arange(len(weights), dtype=np.int64), (0.0, 0.0, 1.0, 1.0))
    return layout


def _pack_fixed_aspect_rectangles(
    widths: np.ndarray,
    heights: np.ndarray,
    margin: float = 0.004,
) -> tuple[np.ndarray, float]:
    """Pack rectangles with one shared scale and preserved aspect ratios."""
    widths = np.maximum(np.asarray(widths, dtype=np.float64), 1e-7)
    heights = np.maximum(np.asarray(heights, dtype=np.float64), 1e-7)
    order = sorted(
        range(len(widths)),
        key=lambda index: (-heights[index], -widths[index] * heights[index], index),
    )

    def attempt(scale: float):
        layout = np.zeros((len(widths), 4), dtype=np.float64)
        x = y = row_height = margin
        for index in order:
            width = widths[index] * scale
            height = heights[index] * scale
            if x + width + margin > 1.0 and x > margin:
                y += row_height + margin
                x = margin
                row_height = 0.0
            if y + height + margin > 1.0:
                return None
            layout[index] = (x, y, x + width, y + height)
            x += width + margin
            row_height = max(row_height, height)
        return layout

    low = 0.0
    high = min(
        (1 - 2 * margin) / widths.max(),
        (1 - 2 * margin) / heights.max(),
    )
    best = attempt(0.0)
    for _ in range(48):
        middle = 0.5 * (low + high)
        candidate = attempt(middle)
        if candidate is None:
            high = middle
        else:
            low = middle
            best = candidate
    return best.astype(np.float32), float(low)


def _develop_source_transforms(
    chart_ids: np.ndarray,
    component_ids: np.ndarray,
    local_uv: np.ndarray,
    axes: np.ndarray,
    centers: np.ndarray,
    lows: np.ndarray,
    extents: np.ndarray,
    cross_pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unfold adjacent charts into a reference plane independent of atlas packing.

    Packed atlas rectangles solve storage overlap.  They are deliberately not a
    surface parameterization, so using their positions as reference-image crops
    cuts large motifs at arbitrary chart boundaries.  This routine develops the
    tangent frames along the chart graph, then solves all chart translations from
    seam correspondences.  Disconnected surface components receive disjoint
    reference rectangles.
    """
    chart_count = len(axes)
    rotations = np.zeros((chart_count, 2, 2), dtype=np.float64)
    translations = np.zeros((chart_count, 2), dtype=np.float64)
    chart_sizes = np.bincount(chart_ids, minlength=chart_count)
    chart_components = np.asarray(
        [component_ids[np.flatnonzero(chart_ids == chart)[0]] for chart in range(chart_count)],
        dtype=np.int64,
    )
    tree_parent = np.full(chart_count, -1, dtype=np.int64)
    placement_order: list[int] = []
    planar_components: set[int] = set()

    seam_groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for left, right in np.asarray(cross_pairs, dtype=np.int64).reshape(-1, 2):
        first, second = int(chart_ids[left]), int(chart_ids[right])
        if first == second:
            continue
        if first > second:
            first, second = second, first
            left, right = right, left
        seam_groups.setdefault((first, second), []).append((int(left), int(right)))

    for component in np.unique(component_ids):
        charts = np.flatnonzero(chart_components == component)
        if not len(charts):
            continue
        root = int(charts[np.argmax(chart_sizes[charts])])
        rotations[root] = np.eye(2)
        root_axes = axes[root].astype(np.float64)
        alignment = []
        for chart in charts:
            singular = np.linalg.svd(
                axes[chart].astype(np.float64).T @ root_axes,
                compute_uv=False,
            )
            alignment.append(float(singular.min()))
        if min(alignment, default=1.0) >= 0.98:
            # Planar components already have a valid shared world-space frame.
            # Use it directly rather than accumulating approximate seam-pair
            # translations across an otherwise flat road.
            for chart in charts:
                rotations[chart] = axes[chart].astype(np.float64).T @ root_axes
                translations[chart] = (
                    centers[chart].astype(np.float64) - centers[root].astype(np.float64)
                ) @ root_axes
                tree_parent[chart] = root if chart != root else -1
            placement_order.extend([root] + [int(value) for value in charts if value != root])
            planar_components.add(int(component))
            continue
        placement_order.append(root)
        placed = {root}
        remaining = set(map(int, charts)) - placed
        while remaining:
            candidates = []
            for (first, second), pairs in seam_groups.items():
                if first in placed and second in remaining:
                    candidates.append((len(pairs), -first, -second, first, second))
                elif second in placed and first in remaining:
                    candidates.append((len(pairs), -second, -first, second, first))
            if not candidates:
                # This should only occur for degenerate graph fragments.  Keep
                # deterministic behavior and give the fragment its own frame.
                child = min(remaining)
                rotations[child] = np.eye(2)
                placed.add(child)
                remaining.remove(child)
                continue
            _, _, _, parent, child = max(candidates)
            target = axes[parent].astype(np.float64) @ rotations[parent]
            u, _, vt = np.linalg.svd(axes[child].astype(np.float64).T @ target)
            rotations[child] = u @ vt
            key = (min(parent, child), max(parent, child))
            parent_values = []
            child_values = []
            for left, right in seam_groups[key]:
                if int(chart_ids[left]) == parent:
                    parent_id, child_id = left, right
                else:
                    parent_id, child_id = right, left
                q_parent = lows[parent] + local_uv[parent_id] * extents[parent]
                q_child = lows[child] + local_uv[child_id] * extents[child]
                parent_values.append(q_parent @ rotations[parent] + translations[parent])
                child_values.append(q_child @ rotations[child])
            translations[child] = (
                np.asarray(parent_values).mean(0) - np.asarray(child_values).mean(0)
            )
            tree_parent[child] = parent
            placement_order.append(child)
            placed.add(child)
            remaining.remove(child)

    corners = np.asarray(((0, 0), (1, 0), (0, 1), (1, 1)), dtype=np.float64)
    # Reference-image sampling may overlap when a curved component unfolds
    # over itself; atlas storage may not.  The latter is already guaranteed by
    # disjoint chart_cells.  Splitting the reference development at every 2-D
    # overlap fragmented real connected surfaces (44 islands for Stump) and
    # changed motif scale/phase at artificial cuts.  Keep one developed source
    # island per genuine 3-D connected component instead.
    island_ids = np.full(chart_count, -1, dtype=np.int64)
    island_charts: list[list[int]] = []
    for component in np.unique(chart_components):
        charts = np.flatnonzero(chart_components == component)
        island = len(island_charts)
        island_ids[charts] = island
        island_charts.append(charts.astype(np.int64).tolist())

    island_bounds = []
    for charts_list in island_charts:
        developed_corners = []
        for chart in charts_list:
            physical = lows[chart] + corners * extents[chart]
            developed_corners.append(
                physical @ rotations[chart] + translations[chart]
            )
        stacked = np.concatenate(developed_corners)
        low = stacked.min(0)
        extent = np.maximum(stacked.max(0) - low, 1e-7)
        island_bounds.append((low, extent))
    island_cells, shared_scale = _pack_fixed_aspect_rectangles(
        np.asarray([value[1][0] for value in island_bounds]),
        np.asarray([value[1][1] for value in island_bounds]),
    )
    transforms = np.zeros((chart_count, 2, 3), dtype=np.float32)
    regions = np.zeros((chart_count, 4), dtype=np.float32)
    for island, charts_list in enumerate(island_charts):
        charts = np.asarray(charts_list, dtype=np.int64)
        low, _ = island_bounds[island]
        cell = island_cells[island].astype(np.float64)
        scale = np.full(2, shared_scale, dtype=np.float64)
        for chart in charts:
            linear = np.diag(extents[chart]) @ rotations[chart] @ np.diag(scale)
            offset = (
                (lows[chart] @ rotations[chart] + translations[chart] - low) * scale
                + cell[:2]
            )
            transforms[chart, :, :2] = linear.T
            transforms[chart, :, 2] = offset
            source_corners = corners @ linear + offset
            regions[chart] = np.r_[source_corners.min(0), source_corners.max(0)]
    return transforms, np.clip(regions, 0, 1), island_ids


def _adaptive_surface_graph(
    points: np.ndarray,
    normals: np.ndarray,
    neighbours: int,
):
    """Build a deterministic, surface-aware graph from mutual and radius support."""
    count = len(points)
    base_k = min(count - 1, max(2, int(neighbours)))
    candidate_k = min(count - 1, max(16, base_k * 4))
    distances, indices = cKDTree(points).query(
        points, k=candidate_k + 1, workers=-1,
    )
    distances = np.asarray(distances)[:, 1:].astype(np.float32, copy=False)
    indices = np.asarray(indices)[:, 1:].astype(np.int64, copy=False)
    local_scale = np.median(distances[:, :base_k], axis=1).astype(np.float32)
    positive = local_scale[local_scale > 0]
    global_scale = max(float(np.median(positive)) if len(positive) else 1.0, 1e-7)
    local_scale = np.clip(local_scale, global_scale * 0.25, global_scale * 8.0)

    local_dots = np.abs(
        np.einsum(
            "nij,nj->ni", normals[indices[:, :base_k]], normals,
            optimize=True,
        )
    )
    normal_reliability = np.median(local_dots, axis=1).astype(np.float32)

    source = np.repeat(np.arange(count, dtype=np.int64), candidate_k)
    target = indices.reshape(-1)
    spatial = distances.reshape(-1)
    candidate_graph = coo_matrix(
        (np.ones(len(source), dtype=np.uint8), (source, target)),
        shape=(count, count),
    ).tocsr()
    mutual = np.asarray(candidate_graph[target, source]).reshape(-1) > 0
    adaptive = spatial <= 8.0 * np.maximum(local_scale[source], local_scale[target])

    keep = mutual | adaptive
    chunk = 1_000_000
    for start in range(0, len(source), chunk):
        stop = min(start + chunk, len(source))
        ids = slice(start, stop)
        left, right = source[ids], target[ids]
        delta = points[right] - points[left]
        length = np.maximum(spatial[ids], 1e-7)
        dot = np.abs(np.einsum("ij,ij->i", normals[left], normals[right]))
        tangent = np.maximum(
            np.abs(np.einsum("ij,ij->i", delta, normals[left])),
            np.abs(np.einsum("ij,ij->i", delta, normals[right])),
        ) / length
        reliability = np.minimum(normal_reliability[left], normal_reliability[right])
        # Reliable normals reject cross-layer and perpendicular-surface shortcuts.
        # Noisy Gaussian normals receive a deliberately softer test so sparse
        # samples of one surface are not split into hundreds of micro-islands.
        normal_threshold = 0.02 + 0.08 * reliability
        tangent_threshold = 0.995 - 0.045 * reliability
        keep[ids] &= (dot >= normal_threshold) & (tangent <= tangent_threshold)

    source, target, spatial = source[keep], target[keep], spatial[keep]
    left = np.minimum(source, target)
    right = np.maximum(source, target)
    pairs = np.stack((left, right), axis=1)
    pairs, unique_ids = np.unique(pairs, axis=0, return_index=True)
    spatial = spatial[unique_ids]
    dots = np.abs((normals[pairs[:, 0]] * normals[pairs[:, 1]]).sum(1))
    delta = points[pairs[:, 1]] - points[pairs[:, 0]]
    tangent = np.maximum(
        np.abs((delta * normals[pairs[:, 0]]).sum(1)),
        np.abs((delta * normals[pairs[:, 1]]).sum(1)),
    ) / np.maximum(spatial, 1e-7)
    costs = np.maximum(
        spatial / global_scale
        # Geodesic chart assignment must strongly prefer a consistent local
        # tangent plane.  The former 2/2 weights let one chart wrap around
        # the stump and span several bulldozer parts, which then collapsed
        # under a single PCA projection.
        + 8.0 * (1.0 - np.clip(dots, 0, 1))
        + 4.0 * np.clip(tangent, 0, 1),
        1e-6,
    )
    graph = coo_matrix(
        (
            np.r_[costs, costs],
            (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]]),
        ),
        shape=(count, count),
    ).tocsr()
    component_count, component_ids = connected_components(graph, directed=False)
    return graph, pairs, spatial, component_count, component_ids, global_scale


@dataclass
class AtlasTopology:
    """Deterministic surface charts and their packed atlas coordinates."""

    chart_ids: torch.Tensor
    component_ids: torch.Tensor
    local_uv: torch.Tensor
    atlas_uv: torch.Tensor
    chart_layout: torch.Tensor
    chart_centers: torch.Tensor
    chart_axes: torch.Tensor
    chart_low: torch.Tensor
    chart_extent: torch.Tensor
    reference_regions: torch.Tensor
    source_transforms: torch.Tensor
    source_island_ids: torch.Tensor
    edges: torch.Tensor
    edge_3d_distance: torch.Tensor
    edge_uv_per_3d: torch.Tensor
    seam_edges: torch.Tensor
    triangles: torch.Tensor
    collision_pairs: torch.Tensor
    feather_uv: torch.Tensor
    feather_weight: torch.Tensor
    chart_cells: torch.Tensor
    seam_weight: torch.Tensor

    @property
    def chart_count(self) -> int:
        return int(self.chart_layout.shape[0])

    @classmethod
    def from_checkpoint_state(
        cls, state_dict: dict[str, torch.Tensor], prefix: str = "texture_field.",
    ) -> "AtlasTopology" | None:
        names = (
            "chart_ids", "local_uv", "atlas_uv", "chart_layout", "chart_centers",
            "chart_axes", "chart_low", "chart_extent", "reference_regions", "edges",
            "edge_3d_distance", "edge_uv_per_3d",
            "seam_edges", "triangles", "collision_pairs", "feather_uv", "feather_weight",
        )
        if prefix + "chart_ids" not in state_dict:
            return None
        missing = [name for name in names if prefix + name not in state_dict]
        if missing:
            raise RuntimeError(f"incomplete atlas topology in checkpoint: {missing}")
        values = {name: state_dict[prefix + name].detach() for name in names}
        values["source_transforms"] = state_dict.get(
            prefix + "source_transforms",
            torch.stack((
                torch.stack((
                    values["reference_regions"][:, 2] - values["reference_regions"][:, 0],
                    torch.zeros_like(values["reference_regions"][:, 0]),
                    values["reference_regions"][:, 0],
                ), dim=1),
                torch.stack((
                    torch.zeros_like(values["reference_regions"][:, 0]),
                    values["reference_regions"][:, 3] - values["reference_regions"][:, 1],
                    values["reference_regions"][:, 1],
                ), dim=1),
            ), dim=1),
        ).detach()
        values["source_island_ids"] = state_dict.get(
            prefix + "source_island_ids",
            torch.arange(
                len(values["chart_layout"]), device=values["chart_layout"].device,
                dtype=torch.long,
            ),
        ).detach()
        values["component_ids"] = state_dict.get(
            prefix + "component_ids", values["chart_ids"],
        ).detach()
        values["chart_cells"] = state_dict.get(
            prefix + "chart_cells", values["chart_layout"],
        ).detach()
        values["seam_weight"] = state_dict.get(
            prefix + "seam_weight",
            torch.ones(
                len(values["seam_edges"]), 1,
                device=values["seam_edges"].device, dtype=torch.float32,
            ),
        ).detach()
        return cls(**values)

    @classmethod
    def from_surface(
        cls,
        xyz: torch.Tensor,
        normals: torch.Tensor | None,
        *,
        neighbours: int = 8,
        target_charts: int = 8,
        atlas_resolution: int = 512,
        padding: int = 4,
        feather: float = 0.15,
        max_constraint_edges: int = 131072,
    ) -> "AtlasTopology":
        points = xyz.detach().float().cpu().numpy().astype(np.float32, copy=False)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
            raise ValueError("atlas requires selected xyz with shape (N, 3), N >= 3")
        count = len(points)
        if normals is None:
            normal_values = np.zeros_like(points)
            normal_values[:, 2] = 1
        else:
            normal_values = normals.detach().float().cpu().numpy().astype(np.float32, copy=False)
            if normal_values.shape != points.shape:
                raise ValueError("atlas normals must have the same shape as xyz")
            normal_values /= np.maximum(np.linalg.norm(normal_values, axis=1, keepdims=True), 1e-7)

        graph, pairs, spatial, component_count, component_ids, scale = (
            _adaptive_surface_graph(points, normal_values, neighbours)
        )
        component_sizes = np.bincount(component_ids, minlength=component_count)
        significant_floor = max(3, int(math.ceil(count * 0.001)))
        significant_components = int(np.count_nonzero(component_sizes >= significant_floor))
        extra_splits = max(0, int(target_charts) - significant_components)
        chart_count = min(count, component_count + extra_splits)
        allocation = np.ones(component_count, dtype=np.int64)
        for _ in range(chart_count - component_count):
            eligible = allocation < component_sizes
            score = component_sizes / allocation
            score[~eligible] = -1
            allocation[int(score.argmax())] += 1

        seeds = []
        used = set()
        for component, component_chart_count in enumerate(allocation):
            ids = np.flatnonzero(component_ids == component)
            component_points = points[ids]
            low = np.quantile(component_points, 0.01, axis=0)
            high = np.quantile(component_points, 0.99, axis=0)
            extent = np.maximum(high - low, 1e-6)
            position_features = (component_points - component_points.mean(0)) / extent
            features = np.concatenate(
                # Normal orientation is as important as position for chart
                # seeds on curved or articulated selections.  Absolute
                # normals keep the result independent of normal sign.
                (position_features, 1.5 * np.abs(normal_values[ids])), axis=1,
            )
            if component_chart_count == 1:
                centroids = features.mean(0, keepdims=True)
            else:
                sample_ids = np.linspace(
                    0, len(features) - 1, min(len(features), 100000), dtype=np.int64,
                )
                centroids, _ = kmeans2(
                    features[sample_ids], int(component_chart_count), iter=20,
                    minit="++", check_finite=False, seed=0,
                )
            feature_tree = cKDTree(features)
            _, candidates = feature_tree.query(
                centroids, k=min(len(ids), max(8, int(component_chart_count))),
                workers=-1,
            )
            candidates = np.asarray(candidates).reshape(int(component_chart_count), -1)
            for row in candidates:
                seed = next(
                    (int(ids[value]) for value in row if int(ids[value]) not in used), None,
                )
                if seed is None:
                    seed = int(next(value for value in ids if int(value) not in used))
                seeds.append(seed)
                used.add(seed)
        def assign_charts(current_seeds: list[int]):
            distance, _, sources = dijkstra(
                graph, directed=False,
                indices=np.asarray(current_seeds, dtype=np.int64),
                min_only=True, return_predecessors=True,
            )
            seed_to_chart = {
                seed: chart for chart, seed in enumerate(current_seeds)
            }
            assigned = np.asarray(
                [seed_to_chart[int(seed)] for seed in sources], dtype=np.int64,
            )
            return np.asarray(distance), assigned

        distance_to_seed, chart_ids = assign_charts(seeds)
        global_centered = points - points.mean(0)
        _, _, global_v = np.linalg.svd(global_centered, full_matrices=False)
        fallback_axes = np.stack((_oriented(global_v[0]), _oriented(global_v[1])), axis=1)
        # Large planar components need one shared orientation and reference
        # coordinate system. Independent per-chart PCA can rotate or flip
        # adjacent road charts even when every chart is internally valid.
        component_frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for component in range(component_count):
            ids = np.flatnonzero(component_ids == component)
            if len(ids) < significant_floor:
                continue
            component_points = points[ids]
            component_center = component_points.mean(0)
            centered = component_points - component_center
            if len(ids) < 3 or np.linalg.matrix_rank(centered) < 2:
                continue
            _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
            thickness_ratio = float(singular[2] / max(singular[1], 1e-7))
            normal_alignment = float(
                np.median(np.abs(normal_values[ids] @ vectors[2]))
            )
            if thickness_ratio > 0.15 or normal_alignment < 0.45:
                continue
            component_axes = np.stack(
                (_oriented(vectors[0]), _oriented(vectors[1])), axis=1,
            )
            projected = centered @ component_axes
            component_low = np.quantile(projected, 0.005, axis=0).astype(np.float32)
            component_high = np.quantile(projected, 0.995, axis=0).astype(np.float32)
            component_extent = np.maximum(component_high - component_low, 1e-6)
            component_frames[component] = (
                component_center.astype(np.float32), component_axes.astype(np.float32),
                component_low, component_extent,
            )

        collision_distance_limit = scale * 8.0
        collision_uv_radius = 2.5 / max(int(atlas_resolution), 16)

        def projection_collisions(ids: np.ndarray, uv: np.ndarray):
            if len(ids) < 2:
                return set()
            query_k = min(len(ids), 5)
            uv_distances, uv_neighbours = cKDTree(uv).query(
                uv, k=query_k, workers=-1,
            )
            failures = set()
            for local_left in range(len(ids)):
                for offset in range(1, query_k):
                    if uv_distances[local_left, offset] >= collision_uv_radius:
                        continue
                    local_right = int(uv_neighbours[local_left, offset])
                    left_id, right_id = int(ids[local_left]), int(ids[local_right])
                    if np.linalg.norm(points[left_id] - points[right_id]) > collision_distance_limit:
                        failures.add((min(left_id, right_id), max(left_id, right_id)))
            return failures

        def project_charts(assigned: np.ndarray, current_count: int):
            local = np.zeros((count, 2), dtype=np.float32)
            centers_value = np.zeros((current_count, 3), dtype=np.float32)
            axes_value = np.zeros((current_count, 3, 2), dtype=np.float32)
            lows_value = np.zeros((current_count, 2), dtype=np.float32)
            extents_value = np.ones((current_count, 2), dtype=np.float32)
            for chart in range(current_count):
                ids = np.flatnonzero(assigned == chart)
                chart_points = points[ids]
                center = chart_points.mean(0)
                centered = chart_points - center
                component = int(component_ids[ids[0]])
                if component in component_frames:
                    chart_axes = component_frames[component][1]
                elif len(ids) >= 3 and np.linalg.matrix_rank(centered) >= 2:
                    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
                    chart_axes = np.stack(
                        (_oriented(vectors[0]), _oriented(vectors[1])), axis=1,
                    )
                else:
                    chart_axes = fallback_axes
                projected = centered @ chart_axes
                low = np.quantile(projected, 0.005, axis=0).astype(np.float32)
                high = np.quantile(projected, 0.995, axis=0).astype(np.float32)
                extent = np.maximum(high - low, 1e-6)
                normalized = np.clip((projected - low) / extent, 0, 1).astype(np.float32)
                local[ids] = normalized
                centers_value[chart] = center
                axes_value[chart] = chart_axes
                lows_value[chart] = low
                extents_value[chart] = extent
            return local, centers_value, axes_value, lows_value, extents_value

        local_uv, centers, axes, lows, extents = project_charts(
            chart_ids, chart_count,
        )

        # A single tangent projection cannot represent a chart that wraps
        # around a curved object or spans several articulated faces.  Detect
        # these failures before packing and add deterministic geodesic seeds
        # at the folded layer.  Voronoi regions remain graph-connected and no
        # seed can cross a disconnected component.
        adaptive_chart_limit = min(
            count,
            component_count
            + max(0, 16 * int(target_charts) - significant_components),
        )
        target_collision_rate = 0.02
        while chart_count < adaptive_chart_limit:
            chart_failures = []
            all_colliding_points: set[int] = set()
            for chart in range(chart_count):
                ids = np.flatnonzero(chart_ids == chart)
                if len(ids) < 2:
                    continue
                component = int(component_ids[ids[0]])
                if component in component_frames:
                    # This component already uses one shared affine surface
                    # frame.  Splitting it cannot unfold anything and would
                    # fragment large planar selections such as the road.
                    continue
                failures = projection_collisions(ids, local_uv[ids])
                if not failures:
                    continue
                colliding = {value for pair in failures for value in pair}
                all_colliding_points.update(colliding)
                # Prefer the most widely separated folded layers.  The point
                # farther from its current seed becomes the next seed.
                pair = max(
                    failures,
                    key=lambda value: float(np.linalg.norm(points[value[0]] - points[value[1]])),
                )
                candidate = max(pair, key=lambda value: float(distance_to_seed[value]))
                chart_failures.append((len(colliding) / len(ids), chart, int(candidate)))
            if len(all_colliding_points) / max(count, 1) <= target_collision_rate:
                break
            additions = []
            for _, _, candidate in sorted(chart_failures, reverse=True):
                if candidate in used or candidate in additions:
                    continue
                additions.append(candidate)
                if len(additions) >= min(4, adaptive_chart_limit - chart_count):
                    break
            if not additions:
                break
            seeds.extend(additions)
            used.update(additions)
            chart_count = len(seeds)
            distance_to_seed, chart_ids = assign_charts(seeds)
            local_uv, centers, axes, lows, extents = project_charts(
                chart_ids, chart_count,
            )

        pad = float(padding) / max(int(atlas_resolution), 1)
        chart_sizes = np.bincount(chart_ids, minlength=chart_count)
        minimum_cell = (2 * int(padding) + 1) / max(int(atlas_resolution), 1)
        cells = _pack_weighted_rectangles(
            # Area alone does not guarantee enough texels on both axes after
            # a guillotine split.  A modest safety factor keeps the inner
            # rectangle valid while preserving size-proportional allocation.
            chart_sizes, minimum_area_fraction=(1.5 * minimum_cell) ** 2,
        )
        layout = cells.copy()
        atlas_uv = np.zeros_like(local_uv)
        for chart in range(chart_count):
            cell = cells[chart]
            inner = cell + np.array([pad, pad, -pad, -pad], dtype=np.float32)
            if np.any(inner[2:] <= inner[:2]):
                inner = cell
            layout[chart] = inner
            ids = np.flatnonzero(chart_ids == chart)
            atlas_uv[ids] = inner[:2] + local_uv[ids] * (inner[2:] - inner[:2])

        same = chart_ids[pairs[:, 0]] == chart_ids[pairs[:, 1]]
        source_transforms, reference_regions, source_island_ids = _develop_source_transforms(
            chart_ids, component_ids, local_uv, axes, centers, lows, extents,
            pairs[~same],
        )
        same_pairs = pairs[same]
        same_spatial = spatial[same]
        edge_ids = _limited_indices(len(same_pairs), int(max_constraint_edges))
        constraint_edges = same_pairs[edge_ids]
        edge_3d_distance = np.maximum(same_spatial[edge_ids], 1e-7).astype(np.float32)
        edge_uv_distance = np.linalg.norm(
            local_uv[constraint_edges[:, 0]] - local_uv[constraint_edges[:, 1]], axis=1,
        )
        edge_uv_per_3d = (edge_uv_distance / edge_3d_distance).astype(np.float32)
        seam_edges = _limited_rows(pairs[~same], max(1024, int(max_constraint_edges) // 4))
        if len(seam_edges):
            seam_delta = points[seam_edges[:, 0]] - points[seam_edges[:, 1]]
            seam_distance = np.linalg.norm(seam_delta, axis=1)
            seam_normal = np.abs(
                (normal_values[seam_edges[:, 0]] * normal_values[seam_edges[:, 1]]).sum(1)
            )
            seam_weight = (
                np.exp(-seam_distance / max(4.0 * scale, 1e-7))
                * np.sqrt(np.clip(seam_normal, 0, 1))
            ).clip(0.05, 1.0).astype(np.float32)[:, None]
        else:
            seam_weight = np.zeros((0, 1), dtype=np.float32)
        feather_uv = atlas_uv.copy()
        feather_weight = np.zeros((count, 1), dtype=np.float32)
        for left_id, right_id in seam_edges:
            if feather_weight[left_id, 0] == 0:
                feather_uv[left_id] = atlas_uv[right_id]
                feather_weight[left_id, 0] = float(feather)
            if feather_weight[right_id, 0] == 0:
                feather_uv[right_id] = atlas_uv[left_id]
                feather_weight[right_id, 0] = float(feather)

        adjacency: dict[int, list[int]] = {}
        for left_id, right_id in constraint_edges:
            adjacency.setdefault(int(left_id), []).append(int(right_id))
            adjacency.setdefault(int(right_id), []).append(int(left_id))
        triangles = []
        for center_id, neighbours_list in adjacency.items():
            if len(neighbours_list) < 2:
                continue
            base = local_uv[center_id]
            ordered = sorted(
                neighbours_list,
                key=lambda value: math.atan2(*(local_uv[value] - base)[::-1]),
            )
            for offset in range(len(ordered) - 1):
                a, b = ordered[offset], ordered[offset + 1]
                first_vector = local_uv[a] - base
                second_vector = local_uv[b] - base
                area = first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0]
                if abs(float(area)) > 1e-7:
                    triangles.append((center_id, a, b))
                    break
            if len(triangles) >= max_constraint_edges // 4:
                break
        triangles_array = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)

        collision_candidates = []
        distance_limit = scale * 8.0
        uv_radius = 2.5 / max(int(atlas_resolution), 16)
        collision_limit = max_constraint_edges // 4
        per_chart_limit = max(1, int(math.ceil(collision_limit / chart_count)))
        for chart in range(chart_count):
            ids = np.flatnonzero(chart_ids == chart)
            if len(ids) < 2:
                continue
            chart_candidates = []
            query_k = min(len(ids), 5)
            uv_distances, uv_neighbours = cKDTree(local_uv[ids]).query(
                local_uv[ids], k=query_k, workers=-1,
            )
            local_pairs = set()
            for local_left in range(len(ids)):
                for offset in range(1, query_k):
                    if uv_distances[local_left, offset] >= uv_radius:
                        continue
                    local_right = int(uv_neighbours[local_left, offset])
                    local_pairs.add((min(local_left, local_right), max(local_left, local_right)))
            for local_left, local_right in local_pairs:
                left_id, right_id = int(ids[local_left]), int(ids[local_right])
                if np.linalg.norm(points[left_id] - points[right_id]) > distance_limit:
                    chart_candidates.append((left_id, right_id))
            chart_array = np.asarray(sorted(chart_candidates), dtype=np.int64).reshape(-1, 2)
            collision_candidates.extend(
                _limited_rows(chart_array, per_chart_limit).tolist()
            )
        collision_candidates = collision_candidates[:collision_limit]
        collision_array = np.asarray(collision_candidates, dtype=np.int64).reshape(-1, 2)

        def tensor(values, dtype=None):
            result = torch.from_numpy(np.asarray(values))
            return result if dtype is None else result.to(dtype)

        return cls(
            tensor(chart_ids, torch.long), tensor(component_ids, torch.long),
            tensor(local_uv, torch.float32),
            tensor(atlas_uv, torch.float32), tensor(layout, torch.float32),
            tensor(centers, torch.float32), tensor(axes, torch.float32),
            tensor(lows, torch.float32), tensor(extents, torch.float32),
            tensor(reference_regions, torch.float32), tensor(source_transforms, torch.float32),
            tensor(source_island_ids, torch.long),
            tensor(constraint_edges, torch.long),
            tensor(edge_3d_distance, torch.float32), tensor(edge_uv_per_3d, torch.float32),
            tensor(seam_edges, torch.long), tensor(triangles_array, torch.long),
            tensor(collision_array, torch.long), tensor(feather_uv, torch.float32),
            tensor(feather_weight, torch.float32), tensor(cells, torch.float32),
            tensor(seam_weight, torch.float32),
        )
