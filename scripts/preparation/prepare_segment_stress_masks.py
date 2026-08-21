#!/usr/bin/env python3
"""Derive deterministic segment masks for Atlas robustness experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raystyle.backend import LegacyGaussianBackend
from raystyle.config import load_config
from raystyle.project_state import load_segment
from raystyle.segment_stress import build_segment_stress_masks, stress_mask_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--thin-fraction", type=float, default=0.10)
    parser.add_argument("--curved-fraction", type=float, default=0.20)
    parser.add_argument("--distant-fraction", type=float, default=0.15)
    args = parser.parse_args()

    config = load_config(args.config)
    backend = LegacyGaussianBackend(
        config.legacy_root, config.model_path, config.source_path,
        config.images, config.resolution, config.white_background,
    )
    selected, metadata = load_segment(
        config.project_state, config.segment_id, backend.point_count,
    )
    selected = selected.cuda()
    ids = torch.where(selected)[0]
    normals = torch.zeros_like(backend.xyz)
    normals[ids] = backend.canonical_normals(ids)
    masks = build_segment_stress_masks(
        backend.xyz, normals, selected,
        thin_fraction=args.thin_fraction,
        curved_fraction=args.curved_fraction,
        distant_fraction=args.distant_fraction,
    )
    diagnostics = stress_mask_diagnostics(selected, masks)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = {}
    for name, mask in masks.items():
        path = output / f"{name}.pt"
        torch.save(mask, path)
        paths[name] = str(path)
    manifest = {
        "source_config": str(args.config.expanduser().resolve()),
        "source_segment": metadata,
        "fractions": {
            "thin": args.thin_fraction,
            "curved": args.curved_fraction,
            "distant": args.distant_fraction,
        },
        "diagnostics": diagnostics,
        "masks": paths,
        "intended_tests": {
            "adjacent_a+adjacent_b": "two touching segments with independent atlases",
            "thin_band": "thin-region UV stability",
            "nonplanar": "high-normal-variation parameterization",
            "distant_a+distant_b": "inter-atlas contamination and remote-surface collision",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
