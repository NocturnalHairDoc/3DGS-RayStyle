#!/usr/bin/env python3
"""Prepare Atlas representation tests for thin and non-planar segments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raystyle.atlas import ATLAS_VERSION
from raystyle.config import load_config
from scripts.preparation.prepare_atlas_representation_validation import (
    write_synthetic_references,
)


CASES = {
    "thin_checker": {
        "base": "configs/road_starry_graph_appearance.yaml",
        "mask": "thin_band.pt", "texture": "checker",
        "layout": "packed", "repeat": 1, "seam": 0.5,
    },
    "thin_orientation": {
        "base": "configs/road_starry_graph_appearance.yaml",
        "mask": "thin_band.pt", "texture": "orientation",
        "layout": "packed", "repeat": 1, "seam": 2.0,
    },
    "nonplanar_checker": {
        "base": "configs/stump_starry_test.yaml",
        "mask": None, "texture": "checker",
        "layout": "developed", "repeat": 3, "seam": 2.0,
    },
    "nonplanar_orientation": {
        "base": "configs/stump_starry_test.yaml",
        "mask": None, "texture": "orientation",
        "layout": "developed", "repeat": 2, "seam": 2.0,
    },
}


def _source_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--segment-root", type=Path,
        default=ROOT / "segments" / "stress_bicycle_v1",
    )
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument(
        "--cases", nargs="+", choices=tuple(CASES), default=list(CASES),
    )
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    config_dir = output / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    references = write_synthetic_references(output / "references")
    segment_root = args.segment_root.expanduser().resolve()

    records = []
    for case_name in args.cases:
        settings = CASES[case_name]
        config = load_config(ROOT / settings["base"])
        if settings["mask"] is not None:
            mask = segment_root / settings["mask"]
            if not mask.is_file():
                raise FileNotFoundError(f"stress segment does not exist: {mask}")
            config.project_state = str(mask)
        else:
            mask = Path(config.project_state)
        config.reference_image = references[settings["texture"]]
        config.output_dir = str(output / case_name)
        config.train.iterations = args.iterations
        config.train.texture_stage_iterations = args.iterations
        config.train.texture_mapping = "atlas"
        config.train.atlas_source_layout = settings["layout"]
        config.train.atlas_reference_repeat = settings["repeat"]
        config.train.texture_init_strength = 1.0
        config.train.random_hdr = False
        config.train.albedo_only_render = True
        config.train.preview_every = min(50, args.iterations)
        config.train.checkpoint_every = args.iterations
        config.train.min_segment_coverage = 0.001
        config.losses.style = 0.5
        config.losses.patch_style = 2.0
        config.losses.content = 0.0
        config.losses.graph = 0.02
        config.losses.material_prior = 0.0
        config.losses.hdr_consistency = 0.0
        config.losses.texture_anchor = 0.5
        config.losses.texture_delta_tv = 0.01
        config.losses.color_mean = 1.0
        config.losses.render_color = 2.0
        config.losses.uv_continuity = 0.02
        config.losses.uv_distortion = 0.05
        config.losses.chart_seam = settings["seam"]
        config.losses.uv_foldover = 0.2
        config.losses.uv_collision = 0.2
        config_path = config_dir / f"{case_name}.yaml"
        config_path.write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8",
        )
        records.append({
            "case": case_name,
            "scene": config.scene,
            "texture": settings["texture"],
            "segment": str(mask),
            "config": str(config_path),
            "output": config.output_dir,
            "atlas_source_layout": settings["layout"],
            "atlas_reference_repeat": settings["repeat"],
            "chart_seam_weight": settings["seam"],
        })

    fingerprint_paths = list((ROOT / "raystyle").glob("*.py")) + [
        ROOT / "scripts" / "preparation" / "prepare_atlas_representation_validation.py",
        ROOT / "scripts" / "preparation" / "prepare_segment_stress_validation.py",
        ROOT / "scripts" / "training" / "run_segment_stress_validation.py",
    ]
    manifest = {
        "purpose": "Thin and non-planar Atlas representation stress validation",
        "atlas_version": ATLAS_VERSION,
        "iterations": args.iterations,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "source_fingerprint": _source_fingerprint(fingerprint_paths),
        "cases": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
