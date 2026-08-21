from raystyle.validation_gates import checkpoint_numeric_gates


def _passing_row():
    return {
        "fixed_style_improvement_pct": 1.1,
        "unseen_style_improvement_pct": 1.1,
        "fixed_leakage_ratio": 1.0,
        "unseen_leakage_ratio": 1.0,
        "hdr_structure_ratio": 1.0,
        "fixed_multiview_ratio": 1.0,
        "unseen_multiview_ratio": 1.0,
        "fixed_patch_ratio": 1.0,
        "unseen_patch_ratio": 1.0,
        "fixed_boundary_leakage_ratio": 1.0,
        "inter_chart_overlap_rate": 0.0,
        "uv_foldover_rate": 0.005,
        "padding_violation_rate": 0.02,
        "reference_gradient_retention": 0.8,
    }


def test_checkpoint_numeric_gates_include_boundary_values():
    gates = checkpoint_numeric_gates(_passing_row())
    assert all(gates.values())


def test_checkpoint_numeric_gates_reject_non_style_regression():
    row = _passing_row()
    row["fixed_multiview_ratio"] = 1.0001
    row["uv_foldover_rate"] = 0.00501
    gates = checkpoint_numeric_gates(row)
    assert not gates["fixed_multiview_no_worse"]
    assert not gates["uv_foldover_half_pct"]
