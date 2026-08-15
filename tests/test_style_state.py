import torch

from raystyle.style_state import StyleState


def _state(
    method, base_value=0.5, albedo_mode="replacement", texture_mapping="triplanar",
):
    albedo = torch.full((10, 3), base_value)
    selected = torch.tensor([True, True] + [False] * 8)
    xyz = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0],
        [2.0, 2.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0],
        [3.0, 3.0, 0.0],
    ])
    # The tiny unit test needs at least three points to define a PCA plane.
    selected = torch.tensor([True, True, True] + [False] * 7)
    return StyleState(
        albedo, selected, method,
        selected_xyz=xyz[selected] if method == "ours" else None,
        texture_resolution=16,
        texture_mapping=texture_mapping,
        albedo_mode=albedo_mode,
    )


def test_ours_trains_material_and_low_order_residual():
    state = _state("ours")
    assert not state.albedo_logits.requires_grad
    assert state.global_albedo_shift.requires_grad
    assert state.texture_field.logit_grid_raw.requires_grad
    assert state.roughness_logits.requires_grad
    assert state.metallic_logits.requires_grad
    assert state.sh_residual.shape == (3, 4, 3)
    assert state.texture_mapping == "triplanar"
    assert state.texture_field.logit_grid_raw.shape == (3, 3, 16, 16)
    assert state.texture_field.blend_weights.shape == (3, 3)


def test_ours_texture_initialization_is_spatial_and_differentiable():
    state = _state("ours")
    reference = torch.zeros(3, 16, 16)
    reference[:, :, 8:] = torch.tensor([0.1, 0.4, 0.9]).view(3, 1, 1)
    state.initialize_texture(reference, strength=0.8)
    preview = state.texture_preview()
    assert preview.shape == (3, 16, 16)
    assert float(preview.detach().var()) > 1e-4
    state.selected_albedo().sum().backward()
    assert state.texture_field.logit_grid_raw.grad is not None
    assert torch.count_nonzero(state.texture_field.logit_grid_raw.grad) > 0


def test_replacement_texture_is_absolute_and_ignores_original_albedo():
    dark = _state("ours", base_value=0.1)
    bright = _state("ours", base_value=0.9)
    reference = torch.full((3, 16, 16), 0.2)
    reference[0] = 0.8
    dark.initialize_texture(reference, strength=1.0)
    bright.initialize_texture(reference, strength=1.0)
    assert torch.allclose(dark.texture_preview(), reference, atol=2e-4)
    assert torch.allclose(bright.texture_preview(), reference, atol=2e-4)
    assert torch.allclose(dark.selected_albedo(), bright.selected_albedo(), atol=1e-6)


def test_additive_mode_remains_available_for_legacy_checkpoints():
    dark = _state("ours", base_value=0.1, albedo_mode="additive")
    bright = _state("ours", base_value=0.9, albedo_mode="additive")
    with torch.no_grad():
        dark.texture_field.logit_grid_raw.zero_()
        bright.texture_field.logit_grid_raw.zero_()
    assert not torch.allclose(dark.selected_albedo(), bright.selected_albedo())


def test_global_shift_and_detail_residual_are_bounded():
    state = _state("ours")
    with torch.no_grad():
        state.global_albedo_shift.fill_(100)
        state.texture_field.logit_grid_raw.normal_()
    assert float(state.bounded_global_shift().detach().abs().max()) <= 0.70001
    detail = state.selected_detail()
    assert float(detail.detach().abs().max()) <= 0.08001


def test_texture_delta_regularization_anchors_reference_without_smoothing_it():
    state = _state("ours")
    reference = torch.rand(3, 16, 16)
    state.initialize_texture(reference)
    anchor, delta_tv = state.texture_regularization()
    assert float(anchor.detach()) < 1e-6
    assert float(delta_tv.detach()) < 1e-6
    with torch.no_grad():
        state.texture_field.logit_grid_raw[..., 5, 5].add_(0.5)
    anchor, delta_tv = state.texture_regularization()
    assert float(anchor.detach()) > 0
    assert float(delta_tv.detach()) > 0


def test_old_uv_checkpoint_without_reference_anchor_still_loads():
    source = _state("ours", albedo_mode="additive")
    old_state = {
        key: value for key, value in source.state_dict().items()
        if key != "texture_field.reference_logit_grid"
    }
    restored = _state("ours", albedo_mode="additive")
    restored.load_checkpoint_state(old_state)
    assert torch.allclose(
        restored.texture_field.reference_logit_grid,
        restored.texture_field.logit_grid(),
    )


def test_legacy_planar_texture_field_remains_loadable():
    source = _state("ours", albedo_mode="additive", texture_mapping="planar")
    old_state = {
        key: value for key, value in source.state_dict().items()
        if key != "texture_field.reference_logit_grid"
    }
    restored = _state("ours", albedo_mode="additive", texture_mapping="planar")
    restored.load_checkpoint_state(old_state)
    assert restored.texture_mapping == "planar"
    assert restored.texture_field.logit_grid_raw.shape == (1, 3, 16, 16)


def test_pbr_only_has_no_sh_parameters():
    state = _state("pbr_only")
    assert state.sh_residual.numel() == 0
    assert state.graph_values().shape[0] == 3


def test_dc_only_trains_one_coefficient():
    state = _state("dc")
    assert not state.albedo_logits.requires_grad
    assert state.sh_residual.shape == (3, 1, 3)
