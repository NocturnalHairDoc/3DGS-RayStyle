import torch

from raystyle.backend import calibrated_tone_map
from raystyle.environments import EnvironmentMap, procedural_environment


def test_environment_sampling_shape_and_gradient():
    pixels = procedural_environment(0, 16, 32)
    environment = EnvironmentMap("test", pixels)
    directions = torch.randn(4, 5, 3, requires_grad=True)
    result = environment.sample(directions, torch.full((4, 5, 1), 0.2))
    assert result.shape == (4, 5, 3)
    result.mean().backward()
    assert directions.grad is not None


def test_diffuse_environment_is_white_balanced_and_achromatic():
    pixels = procedural_environment(1, 16, 32)
    environment = EnvironmentMap("warm", pixels)
    directions = torch.randn(6, 3)
    diffuse = environment.diffuse_sample(directions)
    assert torch.allclose(diffuse[:, 0], diffuse[:, 1])
    assert torch.allclose(diffuse[:, 1], diffuse[:, 2])
    assert float(diffuse.std()) > 0


def test_specular_environment_retains_color():
    pixels = torch.zeros(8, 16, 3)
    pixels[..., 0] = 2.0
    pixels[..., 1] = 0.5
    pixels[..., 2] = 0.1
    environment = EnvironmentMap("colored", pixels)
    directions = torch.randn(5, 3)
    specular = environment.sample(directions)
    assert float((specular[:, 0] - specular[:, 2]).abs().mean()) > 0.5


def test_diffuse_white_calibration_preserves_exposure_stops():
    pixels = torch.tensor([0.2, 0.4, 0.6]).view(1, 1, 3).expand(8, 16, 3)
    directions = torch.randn(9, 3)
    calibrated = EnvironmentMap("constant", pixels).diffuse_sample(
        directions, target_luminance=1.0,
    )
    exposed = EnvironmentMap("constant", pixels, exposure=1.0).diffuse_sample(
        directions, target_luminance=1.0,
    )
    assert torch.allclose(calibrated, torch.ones_like(calibrated), atol=1e-5)
    assert torch.allclose(exposed, torch.full_like(exposed, 2.0), atol=1e-5)


def test_calibrated_tone_map_has_explicit_white_and_exposure():
    values = torch.tensor([0.25, 0.5, 1.0])
    mapped = calibrated_tone_map(values, exposure_stops=0.0, white_point=1.0)
    exposed = calibrated_tone_map(values, exposure_stops=1.0, white_point=1.0)
    assert torch.allclose(mapped, values, atol=1e-6)
    assert torch.allclose(exposed, (values * 2).clamp_max(1), atol=1e-6)
