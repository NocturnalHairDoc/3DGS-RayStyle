import torch

from raystyle.features import (
    adjacent_patch_distance, corresponded_patch_distance, masked_patch_tokens,
)
from raystyle.losses import (
    ReferenceStyleLoss, _nearest_patch_loss, boundary_outside_preservation_loss,
    illumination_consistency_loss, masked_gradient_retention,
    masked_lab_mean_distance, srgb_to_lab,
)


def test_adjacent_view_patch_metric_is_symmetric_and_identity_is_zero():
    features = torch.nn.functional.normalize(torch.randn(1, 8, 4, 4), dim=1)
    mask = torch.ones(1, 4, 4)
    tokens = masked_patch_tokens(features, mask, maximum=8)
    assert len(tokens) == 8
    assert float(adjacent_patch_distance(tokens, tokens)) < 1e-6
    other = torch.nn.functional.normalize(torch.randn_like(tokens), dim=1)
    assert torch.allclose(
        adjacent_patch_distance(tokens, other),
        adjacent_patch_distance(other, tokens),
    )


def test_corresponded_patch_metric_uses_only_shared_sorted_gaussian_ids():
    first_ids = torch.tensor([1, 4, 7, 9])
    second_ids = torch.tensor([2, 4, 7, 11])
    first = torch.eye(4)
    second = torch.stack((torch.ones(4), first[1], first[2], -torch.ones(4)))
    distance = corresponded_patch_distance(first_ids, first, second_ids, second)
    assert distance is not None
    assert float(distance) < 1e-6
    assert corresponded_patch_distance(
        torch.tensor([1]), first[:1], torch.tensor([2]), second[:1],
    ) is None


def test_multiscale_patch_loss_is_finite_and_backpropagates():
    torch.manual_seed(4)
    reference = torch.rand(3, 224, 224)
    reference_features = torch.rand(1, 32, 28, 28)
    objective = ReferenceStyleLoss(reference_features, reference)

    image = torch.rand(1, 3, 224, 224, requires_grad=True)
    features = torch.rand(1, 32, 28, 28, requires_grad=True)
    mask = torch.zeros(1, 224, 224)
    mask[:, 40:190, 30:205] = 1
    loss = objective.patch_loss(image, features, mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert float(image.grad.abs().sum()) > 0
    assert float(features.grad.abs().sum()) > 0


def test_patch_matching_penalizes_missing_reference_modes():
    reference = torch.eye(3)
    query_one_mode = reference[:1].repeat(3, 1)
    query_all_modes = reference
    incomplete = _nearest_patch_loss(query_one_mode, reference)
    complete = _nearest_patch_loss(query_all_modes, reference)
    assert complete < 1e-6
    assert incomplete > 0.2


def test_hdr_consistency_ignores_global_exposure_but_detects_structure_change():
    torch.manual_seed(9)
    first = torch.rand(1, 3, 64, 64) * 0.4 + 0.1
    mask = torch.ones(1, 64, 64)
    exposure_only = illumination_consistency_loss(first, first * 1.8, mask)
    shifted = torch.roll(first, 5, dims=-1)
    structural_change = illumination_consistency_loss(first, shifted, mask)
    assert float(exposure_only) < float(structural_change) * 0.2


def test_patch_loss_is_robust_to_exposure_scaling():
    torch.manual_seed(11)
    reference = torch.rand(3, 96, 96) * 0.35 + 0.1
    features = torch.rand(1, 16, 12, 12)
    objective = ReferenceStyleLoss(features, reference)
    mask = torch.ones(1, 96, 96)
    base = objective.patch_loss(reference.unsqueeze(0), features, mask)
    exposed = objective.patch_loss(reference.unsqueeze(0) * 1.7, features, mask)
    assert abs(float(base - exposed)) < 1e-4


def test_rgb_lab_mean_loss_penalizes_wrong_intrinsic_colour():
    reference = torch.zeros(3, 32, 32)
    reference[0] = 0.72
    reference[1] = 0.63
    reference[2] = 0.33
    features = torch.rand(1, 8, 4, 4)
    objective = ReferenceStyleLoss(features, reference)
    correct = reference.clone().requires_grad_()
    wrong = torch.full_like(reference, 0.62).requires_grad_()
    correct_loss = objective.color_mean_loss(correct)
    wrong_loss = objective.color_mean_loss(wrong)
    wrong_loss.backward()
    assert float(correct_loss.detach()) < 1e-6
    assert float(wrong_loss.detach()) > 0.1
    assert wrong.grad is not None and float(wrong.grad.abs().sum()) > 0


def test_rendered_colour_loss_uses_only_post_pbr_segment_pixels():
    reference = torch.tensor([0.72, 0.63, 0.33]).view(3, 1, 1).expand(3, 32, 32)
    objective = ReferenceStyleLoss(torch.rand(1, 8, 4, 4), reference)
    mask = torch.zeros(1, 32, 32)
    mask[:, 8:24, 8:24] = 1
    correct = torch.zeros(1, 3, 32, 32, requires_grad=True)
    with torch.no_grad():
        correct[:, :, 8:24, 8:24] = reference[:, 8:24, 8:24]
    wrong = correct.detach().clone().requires_grad_()
    with torch.no_grad():
        wrong[:, :, 8:24, 8:24] *= 0.25
    correct_loss = objective.rendered_color_loss(correct, mask)
    wrong_loss = objective.rendered_color_loss(wrong, mask)
    wrong_loss.backward()
    assert float(correct_loss.detach()) < 1e-6
    assert float(wrong_loss.detach()) > 0.2
    assert wrong.grad is not None and float(wrong.grad.abs().sum()) > 0


def test_rendered_colour_loss_compensates_known_exposure():
    reference = torch.tensor([0.4, 0.3, 0.2]).view(3, 1, 1).expand(3, 16, 16)
    objective = ReferenceStyleLoss(torch.rand(1, 8, 2, 2), reference)
    mask = torch.ones(1, 16, 16)
    exposed = reference.unsqueeze(0) * 2
    loss = objective.rendered_color_loss(exposed, mask, exposure_stops=1.0)
    assert float(loss) < 1e-6


def test_roi_patch_loss_ignores_distant_background_changes():
    torch.manual_seed(17)
    reference = torch.rand(3, 64, 64)
    features = torch.rand(1, 12, 8, 8)
    objective = ReferenceStyleLoss(features, reference)
    mask = torch.zeros(1, 64, 64)
    mask[:, 24:40, 24:40] = 1
    first = torch.zeros(1, 3, 64, 64)
    first[:, :, 20:44, 20:44] = reference[:, 20:44, 20:44]
    second = first.clone()
    second[:, :, :12, :] = torch.rand_like(second[:, :, :12, :])
    second[:, :, 52:, :] = torch.rand_like(second[:, :, 52:, :])
    first_loss = objective.patch_loss(first, features, mask)
    second_loss = objective.patch_loss(second, features, mask)
    assert torch.allclose(first_loss, second_loss, atol=1e-6)


def test_srgb_to_lab_is_finite_and_differentiable():
    image = torch.rand(1, 3, 8, 8, requires_grad=True)
    lab = srgb_to_lab(image)
    lab.mean().backward()
    assert lab.shape == image.shape
    assert torch.isfinite(lab).all()
    assert image.grad is not None and torch.isfinite(image.grad).all()


def test_boundary_outside_loss_ignores_far_background_and_penalizes_edge_leak():
    original = torch.zeros(1, 3, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., 24:40, 24:40] = 1
    far = original.clone()
    far[..., :5, :5] = 1
    near = original.clone().requires_grad_()
    with torch.no_grad():
        near[..., 20:24, 24:40] = 1
    far_loss = boundary_outside_preservation_loss(far, original, mask, radius=6)
    near_loss = boundary_outside_preservation_loss(near, original, mask, radius=6)
    near_loss.backward()
    assert far_loss == 0
    assert near_loss > 0.05
    assert near.grad is not None and near.grad[..., 20:24, 24:40].abs().sum() > 0


def test_screen_gradient_retention_is_one_for_identity_and_drops_when_blurred():
    image = torch.zeros(1, 3, 32, 32)
    image[..., ::4, :] = 1
    mask = torch.ones(1, 1, 32, 32)
    identity = masked_gradient_retention(image, image, mask)
    blurred = torch.nn.functional.avg_pool2d(image, 5, stride=1, padding=2)
    softened = masked_gradient_retention(blurred, image, mask)
    assert torch.allclose(identity, torch.ones_like(identity), atol=1e-6)
    assert softened < 0.5


def test_masked_lab_mean_distance_tracks_only_visible_colour_shift():
    reference = torch.full((1, 3, 16, 16), 0.4)
    mask = torch.zeros(1, 1, 16, 16)
    mask[..., 4:12, 4:12] = 1
    outside_only = reference.clone()
    outside_only[..., :4, :] = 1
    shifted = reference.clone()
    shifted[..., 4:12, 4:12] = torch.tensor([0.8, 0.2, 0.1]).view(1, 3, 1, 1)
    assert masked_lab_mean_distance(outside_only, reference, mask) < 1e-7
    assert masked_lab_mean_distance(shifted, reference, mask) > 0.05
