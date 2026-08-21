"""Reusable acceptance rules for reproducible RayStyle experiments."""

from __future__ import annotations


def _ratio(candidate: float, baseline: float) -> float:
    return candidate / max(abs(baseline), 1e-12)


def _improvement(baseline: float, candidate: float) -> float:
    return 100.0 * (baseline - candidate) / max(abs(baseline), 1e-12)


def compare_ours_to_baseline(
    scenario: str, baseline: str, base: dict, ours: dict,
) -> dict:
    """Return lower-is-better improvements and ratios for one method pair."""
    return {
        "scenario": scenario,
        "baseline": baseline,
        "fixed_style_improvement_pct": _improvement(
            base["fixed_style_distance"], ours["fixed_style_distance"],
        ),
        "unseen_style_improvement_pct": _improvement(
            base["unseen_style_distance"], ours["unseen_style_distance"],
        ),
        "fixed_content_ratio": _ratio(
            ours["fixed_content_distance"], base["fixed_content_distance"],
        ),
        "unseen_content_ratio": _ratio(
            ours["unseen_content_distance"], base["unseen_content_distance"],
        ),
        "fixed_leakage_ratio": _ratio(
            ours["fixed_outside_leakage"], base["fixed_outside_leakage"],
        ),
        "unseen_leakage_ratio": _ratio(
            ours["unseen_outside_leakage"], base["unseen_outside_leakage"],
        ),
        "hdr_structure_ratio": _ratio(
            ours["hdr_structure_distance"], base["hdr_structure_distance"],
        ),
        "fixed_multiview_ratio": _ratio(
            ours["fixed_multiview_std"], base["fixed_multiview_std"],
        ),
        "unseen_multiview_ratio": _ratio(
            ours["unseen_multiview_std"], base["unseen_multiview_std"],
        ),
        "fixed_patch_ratio": _ratio(
            ours["fixed_patch_consistency"], base["fixed_patch_consistency"],
        ),
        "unseen_patch_ratio": _ratio(
            ours["unseen_patch_consistency"], base["unseen_patch_consistency"],
        ),
    }


def checkpoint_numeric_gates(row: dict) -> dict[str, bool]:
    """Evaluate the numeric checkpoint-selection thresholds."""
    boundary = row["fixed_boundary_leakage_ratio"]
    return {
        "fixed_style_1pct": row["fixed_style_improvement_pct"] >= 1.0,
        "unseen_style_1pct": row["unseen_style_improvement_pct"] >= 1.0,
        "fixed_leakage_10pct": row["fixed_leakage_ratio"] <= 1.10,
        "unseen_leakage_10pct": row["unseen_leakage_ratio"] <= 1.10,
        "hdr_structure_5pct": row["hdr_structure_ratio"] <= 1.05,
        "fixed_multiview_no_worse": row["fixed_multiview_ratio"] <= 1.0,
        "unseen_multiview_no_worse": row["unseen_multiview_ratio"] <= 1.0,
        "fixed_patch_5pct": row["fixed_patch_ratio"] <= 1.05,
        "unseen_patch_5pct": row["unseen_patch_ratio"] <= 1.05,
        "fixed_boundary_leakage_10pct": boundary is None or boundary <= 1.10,
        "inter_chart_overlap_zero": (
            row["inter_chart_overlap_rate"] is not None
            and row["inter_chart_overlap_rate"] <= 1e-8
        ),
        "uv_foldover_half_pct": (
            row["uv_foldover_rate"] is not None
            and row["uv_foldover_rate"] <= 0.005
        ),
        "padding_violation_2pct": (
            row["padding_violation_rate"] is not None
            and row["padding_violation_rate"] <= 0.02
        ),
        "reference_gradient_80pct": (
            row["reference_gradient_retention"] is not None
            and row["reference_gradient_retention"] >= 0.8
        ),
    }


def style_override(visual_gates: dict, scenario: str, split: str) -> bool:
    """Return an explicit per-scenario visual style override."""
    review = visual_gates.get("style_composition_improved", {}).get(scenario, {})
    return review.get(split) is True


def acceptance_pass(
    gates: dict[str, bool], fixed_style_override: bool,
    unseen_style_override: bool,
) -> bool:
    """Allow visual overrides only for the two style thresholds."""
    non_style_gates = {
        name: passed for name, passed in gates.items()
        if name not in {"gate_fixed_style_1pct", "gate_unseen_style_1pct"}
    }
    return (
        (gates["gate_fixed_style_1pct"] or fixed_style_override)
        and (gates["gate_unseen_style_1pct"] or unseen_style_override)
        and all(non_style_gates.values())
    )
