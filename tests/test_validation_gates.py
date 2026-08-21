from raystyle.validation_gates import acceptance_pass, style_override


def _gates(**changes):
    values = {
        "gate_fixed_style_1pct": True,
        "gate_unseen_style_1pct": True,
        "gate_fixed_multiview_no_worse": True,
        "gate_unseen_multiview_no_worse": True,
        "gate_leakage": True,
    }
    values.update(changes)
    return values


def test_visual_composition_can_replace_only_style_thresholds():
    gates = _gates(
        gate_fixed_style_1pct=False,
        gate_unseen_style_1pct=False,
    )
    assert acceptance_pass(gates, True, True)


def test_visual_composition_cannot_hide_multiview_failure():
    gates = _gates(
        gate_fixed_style_1pct=False,
        gate_unseen_style_1pct=False,
        gate_unseen_multiview_no_worse=False,
    )
    assert not acceptance_pass(gates, True, True)


def test_style_override_is_explicit_per_scenario_and_split():
    review = {
        "style_composition_improved": {
            "bulldozer_starry": {"fixed": True, "unseen": False},
        },
    }
    assert style_override(review, "bulldozer_starry", "fixed")
    assert not style_override(review, "bulldozer_starry", "unseen")
    assert not style_override(review, "stump_starry", "fixed")
