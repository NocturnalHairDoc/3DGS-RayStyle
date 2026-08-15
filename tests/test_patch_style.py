import torch

from raystyle.losses import ReferenceStyleLoss, illumination_consistency_loss, srgb_to_lab


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
