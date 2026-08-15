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
    assert config.train.texture_mapping == "triplanar"
    assert config.train.graph_scope == "appearance"
    assert config.train.pbr_diffuse_white == 1.0
    assert config.train.pbr_white_point == 1.0
    assert config.losses.color_mean == 2.0
    assert config.losses.render_color == 2.0


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
