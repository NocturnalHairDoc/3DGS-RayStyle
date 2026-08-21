import torch

from raystyle.multisegment import composite_independent_edits, region_change_metrics


def test_independent_edit_composition_adds_deltas_from_one_original():
    original = torch.full((3, 2, 2), 0.5)
    first = original.clone()
    second = original.clone()
    first[:, 0, 0] = 0.7
    second[:, 1, 1] = 0.2
    result = composite_independent_edits(original, [first, second])
    assert torch.allclose(result[:, 0, 0], torch.full((3,), 0.7))
    assert torch.allclose(result[:, 1, 1], torch.full((3,), 0.2))
    assert torch.allclose(result[:, 0, 1], torch.full((3,), 0.5))


def test_region_change_metrics_separate_own_cross_and_outside_pixels():
    before = torch.zeros(3, 2, 3)
    after = before.clone()
    after[:, 0, 0] = 0.6
    after[:, 0, 1] = 0.06
    own = torch.tensor([[[1.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])
    other = torch.tensor([[[0.0, 1.0, 0.0], [0.1, 0.0, 0.0]]])
    metrics = region_change_metrics(before, after, own, other)
    assert metrics["own_core_pixels"] == 1
    assert metrics["other_core_pixels"] == 1
    assert metrics["shared_boundary_pixels"] == 1
    assert metrics["outside_pixels"] == 3
    assert abs(metrics["cross_to_own_change_ratio"] - 0.1) < 1e-6


def test_region_change_metrics_split_near_boundary_from_far_outside():
    before = torch.zeros(3, 9, 9)
    after = before.clone()
    after[:, 4, 5] = 0.25
    after[:, 8, 8] = 0.5
    own = torch.zeros(1, 9, 9)
    other = torch.zeros(1, 9, 9)
    own[:, 4, 4] = 1
    other[:, 1, 1] = 1

    metrics = region_change_metrics(
        before, after, own, other, boundary_radius=1,
    )

    assert metrics["near_boundary_change_mean"] > 0
    assert metrics["far_outside_change_mean"] > 0
    assert metrics["near_boundary_pixels"] == 16
    assert metrics["far_outside_pixels"] == 63


def test_region_change_metrics_reject_negative_boundary_radius():
    image = torch.zeros(3, 2, 2)
    mask = torch.zeros(1, 2, 2)
    try:
        region_change_metrics(image, image, mask, mask, boundary_radius=-1)
    except ValueError as error:
        assert "boundary_radius" in str(error)
    else:
        raise AssertionError("negative boundary radius should fail")
