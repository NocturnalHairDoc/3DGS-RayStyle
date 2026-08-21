from scripts.preparation.prepare_method_baselines import pairing_differences
from raystyle.validation_gates import compare_ours_to_baseline


def test_method_pairing_detects_only_changed_fields():
    ours = {
        "method": "ours",
        "output_dir": "/tmp/ours",
        "train": {"iterations": 400, "seed": 42},
    }
    dc = {
        "method": "dc",
        "output_dir": "/tmp/dc",
        "train": {"iterations": 400, "seed": 42},
    }
    assert set(pairing_differences(ours, dc)) == {"method", "output_dir"}


def test_method_comparison_uses_lower_is_better_metrics():
    baseline = {
        "fixed_style_distance": 2.0,
        "unseen_style_distance": 4.0,
        "fixed_content_distance": 2.0,
        "unseen_content_distance": 2.0,
        "fixed_outside_leakage": 2.0,
        "unseen_outside_leakage": 2.0,
        "hdr_structure_distance": 2.0,
        "fixed_multiview_std": 2.0,
        "unseen_multiview_std": 2.0,
        "fixed_patch_consistency": 2.0,
        "unseen_patch_consistency": 2.0,
    }
    ours = {key: value / 2 for key, value in baseline.items()}
    result = compare_ours_to_baseline("scene", "dc", baseline, ours)
    assert result["fixed_style_improvement_pct"] == 50.0
    assert result["unseen_style_improvement_pct"] == 50.0
    assert result["fixed_content_ratio"] == 0.5
    assert result["unseen_patch_ratio"] == 0.5
