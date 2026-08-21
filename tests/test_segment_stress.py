import torch

from raystyle.segment_stress import build_segment_stress_masks, stress_mask_diagnostics


def _surface(count=100):
    x = torch.linspace(-1, 1, count)
    xyz = torch.stack((x, x.square(), torch.zeros_like(x)), dim=1)
    normals = torch.stack((-2 * x, torch.ones_like(x), torch.zeros_like(x)), dim=1)
    normals = torch.nn.functional.normalize(normals, dim=1)
    selected = torch.ones(count, dtype=torch.bool)
    return xyz, normals, selected


def test_stress_masks_are_deterministic_and_well_formed():
    xyz, normals, selected = _surface()
    first = build_segment_stress_masks(xyz, normals, selected)
    second = build_segment_stress_masks(xyz, normals, selected)
    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)
    diagnostics = stress_mask_diagnostics(selected, first)
    assert diagnostics["all_subsets_of_source"]
    assert diagnostics["adjacent_pair_disjoint"]
    assert diagnostics["adjacent_pair_covers_source"]
    assert diagnostics["distant_pair_disjoint"]


def test_stress_mask_counts_follow_requested_fractions():
    xyz, normals, selected = _surface(200)
    masks = build_segment_stress_masks(
        xyz, normals, selected,
        thin_fraction=0.1, curved_fraction=0.2, distant_fraction=0.15,
    )
    assert int(masks["thin_band"].sum()) == 20
    assert int(masks["nonplanar"].sum()) == 40
    assert int(masks["distant_a"].sum()) == 30
    assert int(masks["distant_b"].sum()) == 30
