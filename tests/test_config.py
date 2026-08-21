from raystyle.config import ExperimentConfig


def test_nested_config_construction():
    config = ExperimentConfig.from_dict({
        "method": "pbr_only",
        "train": {"iterations": 7, "sh_degree": 1},
        "losses": {"outside": 3.0},
        "evaluation": {"max_views": 2},
    })
    assert config.method == "pbr_only"
    assert config.train.iterations == 7
    assert config.losses.outside == 3.0
    assert config.evaluation.max_views == 2
    assert config.train.albedo_mode == "replacement"
    assert config.train.texture_mapping == "atlas"
    assert config.train.reference_layout == "full"
    assert config.train.reference_saliency_patches == 4
    assert config.train.reference_focus_scale == 0.32
    assert config.train.reference_tile_count == 3
    assert config.train.reference_metric_tiles is False
    assert config.train.style_patch_reference == "layout"
    assert config.train.graph_scope == "appearance"
    assert config.train.render_mode == "pbr"
    assert config.train.pbr_diffuse_white == 1.0
    assert config.train.pbr_white_point == 1.0
    assert config.losses.color_mean == 2.0
    assert config.losses.render_color == 2.0
    assert config.losses.boundary_outside == 5.0
    assert config.train.atlas_charts == 8
    assert config.train.atlas_source_layout == "packed"
    assert config.train.atlas_reference_repeat == 1
    assert config.losses.uv_distortion == 0.02


def test_atlas_mapping_and_constraints_are_validated():
    config = ExperimentConfig.from_dict({"train": {"texture_mapping": "atlas"}})
    config.validate(require_inputs=False)
    config.train.atlas_feather = 0.75
    try:
        config.validate(require_inputs=False)
    except ValueError as error:
        assert "atlas_feather" in str(error)
    else:
        raise AssertionError("invalid atlas feather was accepted")


def test_render_mode_is_validated():
    config = ExperimentConfig.from_dict({"train": {"render_mode": "specular_only"}})
    try:
        config.validate(require_inputs=False)
    except ValueError as error:
        assert "render_mode" in str(error)
    else:
        raise AssertionError("invalid render mode was accepted")


def test_atlas_source_layout_is_validated():
    config = ExperimentConfig.from_dict({"train": {"atlas_source_layout": "developed"}})
    config.validate(require_inputs=False)
    config = ExperimentConfig.from_dict({"train": {"atlas_source_layout": "component"}})
    config.validate(require_inputs=False)
    config = ExperimentConfig.from_dict({"train": {"atlas_source_layout": "chart"}})
    config.validate(require_inputs=False)
    config = ExperimentConfig.from_dict({"train": {"atlas_source_layout": "projected"}})
    config.validate(require_inputs=False)
    config.train.atlas_source_layout = "implicit"
    try:
        config.validate(require_inputs=False)
    except ValueError as error:
        assert "atlas_source_layout" in str(error)
    else:
        raise AssertionError("invalid atlas source layout was accepted")


def test_atlas_reference_repeat_is_validated():
    config = ExperimentConfig.from_dict({"train": {"atlas_reference_repeat": 3}})
    assert config.train.atlas_reference_repeat == 3
    config.train.atlas_reference_repeat = 0
    try:
        config.validate(require_inputs=False)
    except ValueError as error:
        assert "atlas_reference_repeat" in str(error)
    else:
        raise AssertionError("invalid atlas reference repeat was accepted")


def test_reference_layout_is_validated():
    config = ExperimentConfig.from_dict({
        "train": {"reference_layout": "saliency_grid", "reference_saliency_patches": 6},
    })
    config.validate(require_inputs=False)
    config.train.reference_saliency_patches = 0
    try:
        config.validate(require_inputs=False)
    except ValueError as error:
        assert "reference_saliency_patches" in str(error)
    else:
        raise AssertionError("invalid saliency patch count was accepted")


def test_invalid_method_is_rejected():
    config = ExperimentConfig(method="unknown")
    try:
        config.validate(require_inputs=False)
    except ValueError as error:
        assert "method" in str(error)
    else:
        raise AssertionError("invalid method was accepted")


def test_invalid_graph_scope_is_rejected():
    config = ExperimentConfig.from_dict({"train": {"graph_scope": "geometry"}})
    try:
        config.validate(require_inputs=False)
    except ValueError as error:
        assert "graph_scope" in str(error)
    else:
        raise AssertionError("invalid graph scope was accepted")
