from dataclasses import fields

import numpy as np
import torch

from raystyle.atlas import ATLAS_VERSION, AtlasTopology, _pack_fixed_aspect_rectangles
from raystyle.style_state import StyleState
from raystyle.texture_field import AtlasTextureField


def _surface(width=10, height=8):
    x, y = torch.meshgrid(
        torch.linspace(0, 1, width), torch.linspace(0, 1, height), indexing="xy",
    )
    xyz = torch.stack((x.flatten(), y.flatten(), torch.zeros(width * height)), dim=1)
    normals = torch.tensor([0.0, 0.0, 1.0]).expand_as(xyz)
    return xyz, normals


def _curved_surface():
    angles = torch.linspace(0, 2 * torch.pi, 48)[:-1]
    heights = torch.linspace(0, 3, 24)
    theta, height = torch.meshgrid(angles, heights, indexing="ij")
    xyz = torch.stack(
        (torch.cos(theta), torch.sin(theta), height), dim=-1,
    ).reshape(-1, 3)
    normals = torch.stack(
        (torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)), dim=-1,
    ).reshape(-1, 3)
    return xyz, normals


def _atlas_state():
    xyz, normals = _surface()
    selected = torch.ones(len(xyz), dtype=torch.bool)
    return StyleState(
        torch.full((len(xyz), 3), 0.5), selected, "ours",
        selected_xyz=xyz, selected_normals=normals,
        texture_resolution=64, texture_mapping="atlas",
        atlas_charts=4, atlas_neighbours=6, atlas_padding=2,
    )


def _build_topology():
    xyz, normals = _surface()
    return AtlasTopology.from_surface(
        xyz, normals, neighbours=6, target_charts=4,
        atlas_resolution=64, padding=2,
    )


def test_chart_partition_is_deterministic():
    first = _build_topology()
    second = _build_topology()
    for field in fields(AtlasTopology):
        assert torch.equal(getattr(first, field.name), getattr(second, field.name))


def test_global_random_seeds_do_not_change_static_atlas():
    torch.manual_seed(1)
    np.random.seed(1)
    first = _build_topology()
    torch.manual_seed(9876)
    np.random.seed(9876)
    second = _build_topology()
    assert torch.equal(first.chart_ids, second.chart_ids)
    assert torch.equal(first.local_uv, second.local_uv)
    assert torch.equal(first.atlas_uv, second.atlas_uv)
    assert torch.equal(first.chart_layout, second.chart_layout)


def test_fixed_aspect_packing_uses_one_scale_without_overlap():
    widths = np.asarray([4.0, 2.0, 1.0, 3.0])
    heights = np.asarray([1.0, 3.0, 2.0, 1.5])
    cells, scale = _pack_fixed_aspect_rectangles(widths, heights)
    assert scale > 0
    assert np.all(cells >= 0) and np.all(cells <= 1)
    packed_widths = cells[:, 2] - cells[:, 0]
    packed_heights = cells[:, 3] - cells[:, 1]
    assert np.allclose(packed_widths / widths, scale, rtol=1e-5, atol=1e-6)
    assert np.allclose(packed_heights / heights, scale, rtol=1e-5, atol=1e-6)
    for first in range(len(cells)):
        for second in range(first + 1, len(cells)):
            overlap = (
                max(0.0, min(cells[first, 2], cells[second, 2]) - max(cells[first, 0], cells[second, 0]))
                * max(0.0, min(cells[first, 3], cells[second, 3]) - max(cells[first, 1], cells[second, 1]))
            )
            assert overlap == 0


def test_disconnected_surfaces_never_share_a_chart():
    first_xyz, first_normals = _surface(6, 6)
    second_xyz = first_xyz + torch.tensor([10.0, 0.0, 0.0])
    xyz = torch.cat((first_xyz, second_xyz))
    normals = torch.cat((first_normals, first_normals))
    topology = AtlasTopology.from_surface(
        xyz, normals, neighbours=4, target_charts=2,
        atlas_resolution=64, padding=2,
    )
    first_charts = set(topology.chart_ids[:len(first_xyz)].tolist())
    second_charts = set(topology.chart_ids[len(first_xyz):].tolist())
    assert first_charts
    assert second_charts
    assert first_charts.isdisjoint(second_charts)


def test_adaptive_graph_bridges_small_sampling_gap_but_not_distant_surface():
    first_xyz, first_normals = _surface(6, 6)
    second_xyz = first_xyz + torch.tensor([1.18, 0.0, 0.0])
    third_xyz = first_xyz + torch.tensor([10.0, 0.0, 0.0])
    topology = AtlasTopology.from_surface(
        torch.cat((first_xyz, second_xyz, third_xyz)),
        torch.cat((first_normals, first_normals, first_normals)),
        neighbours=4, target_charts=2, atlas_resolution=64, padding=2,
    )
    size = len(first_xyz)
    first_components = set(topology.component_ids[:size].tolist())
    second_components = set(topology.component_ids[size:2 * size].tolist())
    third_components = set(topology.component_ids[2 * size:].tolist())
    assert first_components == second_components
    assert first_components.isdisjoint(third_components)
    assert topology.chart_count == 2


def test_uv_values_are_finite_and_inside_valid_ranges():
    topology = _build_topology()
    for values in (topology.local_uv, topology.atlas_uv, topology.chart_layout):
        assert torch.isfinite(values).all()
        assert torch.all((values >= 0) & (values <= 1))
    state = _atlas_state()
    with torch.no_grad():
        state.texture_field.uv_offset_raw.normal_(mean=0, std=20)
    current = state.texture_field.current_atlas_uv()
    assert torch.isfinite(current).all()
    assert torch.all((current >= 0) & (current <= 1))
    source = state.texture_field.current_source_uv()
    assert torch.isfinite(source).all()
    assert torch.all((source >= 0) & (source <= 1))


def test_surface_graph_partitions_connected_pca_charts_and_packs_them():
    xyz, normals = _surface()
    topology = AtlasTopology.from_surface(
        xyz, normals, neighbours=6, target_charts=4,
        atlas_resolution=64, padding=2,
    )
    assert topology.chart_count == 4
    assert topology.chart_ids.shape == (len(xyz),)
    assert topology.local_uv.shape == (len(xyz), 2)
    assert torch.all((topology.local_uv >= 0) & (topology.local_uv <= 1))
    assert torch.all((topology.atlas_uv >= 0) & (topology.atlas_uv <= 1))
    assert topology.chart_axes.shape == (4, 3, 2)
    assert topology.edges.shape[1] == 2
    assert topology.seam_edges.shape[1] == 2
    assert torch.unique(topology.reference_regions, dim=0).shape[0] == 4
    for chart in range(topology.chart_count):
        members = set(torch.where(topology.chart_ids == chart)[0].tolist())
        adjacency = {member: set() for member in members}
        for left, right in topology.edges.tolist():
            if left in members and right in members:
                adjacency[left].add(right)
                adjacency[right].add(left)
        reached = {next(iter(members))}
        frontier = list(reached)
        while frontier:
            frontier.extend(adjacency[frontier.pop()] - reached)
            reached.update(frontier)
        assert reached == members


def test_curved_chart_is_split_until_static_collision_is_below_target():
    xyz, normals = _curved_surface()
    topology = AtlasTopology.from_surface(
        xyz, normals, neighbours=6, target_charts=1,
        atlas_resolution=64, padding=2,
    )
    colliding = (
        torch.unique(topology.collision_pairs).numel()
        if topology.collision_pairs.numel() else 0
    )
    assert topology.chart_count > 1
    assert colliding / len(xyz) <= 0.02
    assert torch.isfinite(topology.local_uv).all()


def test_packed_curved_source_layout_uses_disjoint_axis_aligned_cells():
    xyz, normals = _curved_surface()
    field = AtlasTextureField(
        xyz, normals, resolution=64, neighbours=6, target_charts=4,
        padding=2, source_layout="packed",
    )
    linear = field.source_transforms[:, :, :2]
    assert torch.count_nonzero(linear[:, 0, 1].abs() + linear[:, 1, 0].abs()) == 0
    source_uv = field.current_source_uv()
    assert torch.isfinite(source_uv).all()
    assert torch.all((source_uv >= 0) & (source_uv <= 1))


def test_component_source_layout_gives_disconnected_surfaces_full_reference_domains():
    first_xyz, first_normals = _surface(8, 8)
    second_xyz = first_xyz + torch.tensor([10.0, 0.0, 0.0])
    xyz = torch.cat((first_xyz, second_xyz))
    normals = torch.cat((first_normals, first_normals))
    field = AtlasTextureField(
        xyz, normals, resolution=64, neighbours=4, target_charts=4,
        padding=2, source_layout="component",
    )
    source_uv = field.current_source_uv()
    component_ids = field.component_ids
    assert torch.unique(component_ids).numel() == 2
    for component in torch.unique(component_ids):
        values = source_uv[component_ids == component]
        span = values.amax(dim=0) - values.amin(dim=0)
        assert torch.all(span > 0.9)
    assert field.diagnostics()["inter_chart_overlap_rate"] == 0


def test_adjacent_segment_atlases_have_independent_texture_parameters():
    xyz, normals = _surface(12, 8)
    left = xyz[:, 0] <= 0.5
    right = ~left
    first = AtlasTextureField(
        xyz[left], normals[left], resolution=32, neighbours=4,
        target_charts=2, padding=2,
    )
    second = AtlasTextureField(
        xyz[right], normals[right], resolution=32, neighbours=4,
        target_charts=2, padding=2,
    )
    before = {name: value.detach().clone() for name, value in second.state_dict().items()}
    optimizer = torch.optim.SGD(first.parameters(), lr=0.1)
    optimizer.zero_grad()
    first.sample().square().mean().backward()
    optimizer.step()
    for name, value in second.state_dict().items():
        assert torch.equal(value, before[name])
    first_storage = {parameter.untyped_storage().data_ptr() for parameter in first.parameters()}
    second_storage = {parameter.untyped_storage().data_ptr() for parameter in second.parameters()}
    assert first_storage.isdisjoint(second_storage)


def test_chart_source_layout_uses_local_uv_without_rotation_or_crop():
    xyz, normals = _curved_surface()
    field = AtlasTextureField(
        xyz, normals, resolution=64, neighbours=6, target_charts=4,
        padding=2, source_layout="chart",
    )
    assert torch.allclose(field.current_source_uv(), field.current_local_uv())
    expected = torch.tensor((0.0, 0.0, 1.0, 1.0)).expand(field.chart_count, -1)
    assert torch.equal(field.reference_regions.cpu(), expected)


def test_projected_source_layout_is_continuous_across_storage_charts():
    xyz, normals = _surface()
    field = AtlasTextureField(
        xyz, normals, resolution=64, neighbours=6, target_charts=4,
        padding=2, source_layout="projected",
    )
    diagnostics = field.diagnostics()
    assert diagnostics["source_uv_seam_ratio"] < 2.0
    assert diagnostics["source_texel_density_cv"] < 1e-5
    assert field.diagnostics()["inter_chart_overlap_rate"] == 0


def test_atlas_uses_distinct_reference_regions_and_feathers_seams():
    state = _atlas_state()
    reference = torch.zeros(3, 64, 64)
    reference[0] = torch.linspace(0, 1, 64).view(1, 64)
    reference[1] = torch.linspace(0, 1, 64).view(64, 1)
    state.initialize_texture(reference, strength=1.0)
    chart_means = []
    values = state.selected_albedo().detach()
    for chart in range(state.texture_field.chart_count):
        chart_means.append(values[state.texture_field.chart_ids == chart].mean(0))
    assert torch.stack(chart_means).var(0).sum() > 1e-3
    assert torch.count_nonzero(state.texture_field.feather_weight) > 0


def test_atlas_reference_repeat_changes_sampling_scale_without_changing_source_uv():
    xyz, normals = _surface()
    once = AtlasTextureField(
        xyz, normals, resolution=64, neighbours=6, target_charts=4,
        padding=2, reference_repeat=1,
    )
    repeated = AtlasTextureField(
        xyz, normals, resolution=64, neighbours=6, target_charts=4,
        padding=2, reference_repeat=3,
    )
    assert torch.equal(once.current_source_uv(), repeated.current_source_uv())
    assert torch.all((repeated.current_reference_uv() >= 0) & (repeated.current_reference_uv() < 1))
    assert not torch.equal(once.current_reference_uv(), repeated.current_reference_uv())


def test_planar_charts_share_one_affine_reference_coordinate_system():
    topology = _build_topology()
    source_uv = torch.einsum(
        "ni,nji->nj",
        torch.cat((topology.local_uv, torch.ones(len(topology.local_uv), 1)), dim=1),
        topology.source_transforms[topology.chart_ids],
    )
    xyz, _ = _surface()
    design = torch.cat((xyz[:, :2], torch.ones(len(xyz), 1)), dim=1)
    solution = torch.linalg.lstsq(design, source_uv).solution
    residual = design @ solution - source_uv
    assert residual.square().mean() < 2e-3
    assert not torch.equal(topology.reference_regions, topology.chart_cells)


def test_planar_source_uv_is_continuous_across_chart_seams():
    topology = _build_topology()
    homogeneous = torch.cat(
        (topology.local_uv, torch.ones(len(topology.local_uv), 1)), dim=1,
    )
    source_uv = torch.einsum(
        "ni,nji->nj", homogeneous, topology.source_transforms[topology.chart_ids],
    )
    seam_left, seam_right = topology.seam_edges.T
    edge_left, edge_right = topology.edges.T
    seam_distance = (source_uv[seam_left] - source_uv[seam_right]).norm(dim=1)
    edge_distance = (source_uv[edge_left] - source_uv[edge_right]).norm(dim=1)
    assert seam_distance.median() <= edge_distance.median() * 1.5


def test_disconnected_components_receive_non_overlapping_source_regions():
    first_xyz, first_normals = _surface(6, 6)
    second_xyz = first_xyz + torch.tensor([10.0, 0.0, 0.0])
    topology = AtlasTopology.from_surface(
        torch.cat((first_xyz, second_xyz)),
        torch.cat((first_normals, first_normals)),
        neighbours=4, target_charts=2, atlas_resolution=64, padding=2,
    )
    first = topology.reference_regions[topology.chart_ids[:len(first_xyz)]].amin(0), topology.reference_regions[topology.chart_ids[:len(first_xyz)]].amax(0)
    second = topology.reference_regions[topology.chart_ids[len(first_xyz):]].amin(0), topology.reference_regions[topology.chart_ids[len(first_xyz):]].amax(0)
    first_low, first_high = first[0][:2], first[1][2:]
    second_low, second_high = second[0][:2], second[1][2:]
    overlap = (
        (torch.minimum(first_high, second_high) - torch.maximum(first_low, second_low))
        .clamp_min(0).prod()
    )
    assert overlap == 0


def test_atlas_padding_is_dilated_from_inner_chart_texels():
    state = _atlas_state()
    state.initialize_texture(torch.rand(3, 64, 64), strength=1.0)
    field = state.texture_field
    grid = field.logit_grid()
    checked = False
    for layout, cell in zip(field.chart_layout, field.chart_cells):
        x0, y0, x1, y1 = field._pixel_bounds(layout)
        cx0, cy0, cx1, cy1 = field._pixel_bounds(cell)
        if cx0 < x0:
            assert torch.equal(grid[..., y0:y1, cx0:x0], grid[..., y0:y1, x0:x0 + 1].expand(-1, -1, -1, x0 - cx0))
            checked = True
            break
    assert checked


def test_seam_loss_uses_visibility_weighted_final_albedo():
    xyz, normals = _surface()
    state = StyleState(
        torch.full((len(xyz), 3), 0.5), torch.ones(len(xyz), dtype=torch.bool), "ours",
        selected_xyz=xyz, selected_normals=normals,
        selected_visibility=torch.zeros(len(xyz)),
        texture_resolution=64, texture_mapping="atlas",
        atlas_charts=4, atlas_neighbours=6, atlas_padding=2,
    )
    state.initialize_texture(torch.rand(3, 64, 64), strength=1.0)
    assert state.texture_field.seam_edges.numel()
    assert state.atlas_regularization()["chart_seam"] == 0


def test_seam_energy_measures_excess_over_local_surface_gradient():
    state = _atlas_state()
    field = state.texture_field
    source_uv = field.current_source_uv()
    smooth = torch.cat((source_uv, source_uv[:, :1]), dim=1)
    smooth_stats = state._weighted_seam_statistics(smooth)
    discontinuous = smooth.clone()
    discontinuous[field.chart_ids == 0, 0] += 0.5
    discontinuous_stats = state._weighted_seam_statistics(discontinuous)
    assert smooth_stats["raw"] > 0
    assert smooth_stats["excess"] < smooth_stats["raw"]
    assert discontinuous_stats["excess"] > smooth_stats["excess"]


def test_atlas_constraints_are_finite_differentiable_and_diagnostic():
    state = _atlas_state()
    state.initialize_texture(torch.rand(3, 48, 64), strength=1.0)
    losses = state.atlas_regularization()
    assert set(losses) == {
        "uv_continuity", "uv_distortion", "chart_seam",
        "uv_foldover", "uv_collision",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert state.texture_field.logit_grid_raw.grad is not None
    assert state.texture_field.uv_offset_raw.grad is not None
    diagnostics = state.atlas_diagnostics()
    assert set(diagnostics) == {
        "uv_collision_rate", "intra_chart_collision_rate",
        "inter_chart_overlap_rate", "uv_foldover_rate",
        "atlas_occupancy", "padding_violation_rate",
        "chart_seam_energy", "chart_seam_raw_energy",
        "surface_gradient_energy", "uv_distortion",
        "reference_texture_gradient_retention",
        "source_uv_seam_energy", "source_uv_seam_ratio",
        "source_texel_density_cv",
    }
    assert all(torch.isfinite(value) for value in diagnostics.values())
    assert diagnostics["uv_collision_rate"] == diagnostics["intra_chart_collision_rate"]
    assert 0 <= diagnostics["atlas_occupancy"] <= 1
    assert diagnostics["inter_chart_overlap_rate"] == 0
    assert diagnostics["padding_violation_rate"] == 0
    assert diagnostics["source_uv_seam_ratio"] >= 0
    assert diagnostics["source_texel_density_cv"] >= 0


def test_gradient_retention_is_zero_for_flat_reference():
    state = _atlas_state()
    with torch.no_grad():
        state.texture_field.reference_logit_grid.zero_()
        state.texture_field.logit_grid_raw.normal_(0, 0.1)
    retention = state.atlas_diagnostics()["reference_texture_gradient_retention"]
    assert torch.isfinite(retention)
    assert retention.item() == 0.0


def test_atlas_albedo_is_differentiable_with_respect_to_texture_parameters():
    state = _atlas_state()
    state.initialize_texture(torch.rand(3, 64, 64), strength=1.0)
    weights = torch.linspace(0.5, 1.5, state.selected_count).unsqueeze(1)
    (state.selected_albedo() * weights).sum().backward()
    gradient = state.texture_field.logit_grid_raw.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_uv_optimization_does_not_increase_foldover_or_chart_overlap():
    torch.manual_seed(7)
    xyz, normals = _surface(12, 10)
    field = AtlasTextureField(
        xyz, normals, resolution=64, neighbours=6, target_charts=4,
        padding=2, uv_offset_limit=0.3,
    )
    with torch.no_grad():
        field.uv_offset_raw.normal_(std=1.5)
    before = field.diagnostics()
    optimizer = torch.optim.Adam([field.uv_offset_raw], lr=0.08)
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        losses = field.geometric_losses()
        objective = losses["uv_foldover"] + losses["uv_collision"] + 0.1 * losses["uv_distortion"]
        objective.backward()
        optimizer.step()
    after = field.diagnostics()
    assert after["uv_foldover_rate"] <= before["uv_foldover_rate"] + 1e-7
    assert after["inter_chart_overlap_rate"] <= before["inter_chart_overlap_rate"] + 1e-7
    assert torch.isfinite(field.current_atlas_uv()).all()


def test_checkpoint_contains_and_restores_complete_atlas_topology():
    source = _atlas_state()
    source.initialize_texture(torch.rand(3, 64, 64), strength=0.8)
    with torch.no_grad():
        source.texture_field.uv_offset_raw.normal_(std=0.2)
        source.global_albedo_shift.copy_(torch.tensor([[0.1, -0.2, 0.05]]))
        source.roughness_logits.normal_()
        source.metallic_logits.normal_()
        source.sh_residual.normal_(std=0.1)
    state_dict = source.state_dict()
    topology = AtlasTopology.from_checkpoint_state(state_dict)
    assert topology is not None
    restored = StyleState(
        source.base_albedo, torch.ones(source.selected_count, dtype=torch.bool), "ours",
        selected_xyz=torch.zeros(source.selected_count, 3),
        selected_normals=torch.tensor([0.0, 0.0, 1.0]).expand(source.selected_count, 3),
        texture_resolution=64, texture_mapping="atlas", atlas_topology=topology,
    )
    restored.load_checkpoint_state(state_dict)
    for key, value in source.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)
    assert torch.equal(restored.selected_albedo(), source.selected_albedo())
    assert torch.equal(restored.selected_detail(), source.selected_detail())
    assert torch.equal(restored.residual(), source.residual())
    assert restored.texture_field.edge_3d_distance.shape == restored.texture_field.edges.shape[:1]
    assert restored.texture_field.edge_uv_per_3d.shape == restored.texture_field.edges.shape[:1]
    metadata = restored.checkpoint_metadata()
    assert metadata["atlas_version"] == ATLAS_VERSION
    assert metadata["atlas_state_keys"]["chart_id"] == "texture_field.chart_ids"
    assert metadata["atlas_reference_repeat"] == 1


def test_atlas_v1_checkpoint_without_component_ids_still_loads():
    source = _atlas_state()
    state_dict = dict(source.state_dict())
    state_dict.pop("texture_field.component_ids")
    state_dict.pop("texture_field.chart_cells")
    state_dict.pop("texture_field.seam_weight")
    topology = AtlasTopology.from_checkpoint_state(state_dict)
    assert topology is not None
    assert torch.equal(topology.component_ids, topology.chart_ids)
    restored = StyleState(
        source.base_albedo, torch.ones(source.selected_count, dtype=torch.bool), "ours",
        selected_xyz=torch.zeros(source.selected_count, 3),
        selected_normals=torch.tensor([0.0, 0.0, 1.0]).expand(source.selected_count, 3),
        texture_resolution=64, texture_mapping="atlas", atlas_topology=topology,
    )
    restored.load_checkpoint_state(state_dict)
    assert torch.equal(restored.texture_field.component_ids, topology.chart_ids)


def test_atlas_v6_checkpoint_without_source_transforms_still_loads():
    source = _atlas_state()
    state_dict = dict(source.state_dict())
    state_dict.pop("texture_field.source_transforms")
    state_dict.pop("texture_field.source_island_ids")
    topology = AtlasTopology.from_checkpoint_state(state_dict)
    assert topology is not None
    restored = StyleState(
        source.base_albedo, torch.ones(source.selected_count, dtype=torch.bool), "ours",
        selected_xyz=torch.zeros(source.selected_count, 3),
        selected_normals=torch.tensor([0.0, 0.0, 1.0]).expand(source.selected_count, 3),
        texture_resolution=64, texture_mapping="atlas", atlas_topology=topology,
    )
    restored.load_checkpoint_state(state_dict)
    assert torch.isfinite(restored.texture_field.current_source_uv()).all()
