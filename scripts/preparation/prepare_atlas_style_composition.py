#!/usr/bin/env python3
"""Prepare short Atlas style-composition ablations without PBR or HDR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from raystyle.config import load_config


REFERENCES = {
    "starry": "configs/road_starry_graph_appearance.yaml",
    "sunflowers": "configs/road_sunflowers.yaml",
}

VARIANTS = (
    "full",
    "full_repeat4",
    "saliency_grid",
    "saliency_focus",
    "saliency_focus_repeat4",
    "saliency_focus_repeat6",
    "saliency_focus_512_repeat4",
    "saliency_tile",
    "saliency_tile_repeat2",
    "saliency_tile_metric1",
    "saliency_motifs_metric1",
    "no_init_anchor",
    "no_patch",
    "no_uv_seam",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--styles", nargs="+", choices=tuple(REFERENCES), default=None)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=None)
    parser.add_argument("--focus-scale", type=float, default=0.32)
    parser.add_argument("--tile-count", type=int, default=3)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    output = args.output.expanduser().resolve()
    config_dir = output / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    runs = []

    selected_styles = args.styles or list(REFERENCES)
    selected_variants = args.variants or list(VARIANTS)
    for style_name in selected_styles:
        relative_config = REFERENCES[style_name]
        base = load_config(root / relative_config)
        for variant in selected_variants:
            config = type(base).from_dict(base.to_dict())
            config.output_dir = str(output / style_name / variant)
            config.train.iterations = int(args.iterations)
            config.train.texture_stage_iterations = int(args.iterations)
            config.train.texture_mapping = "atlas"
            config.train.texture_init_strength = 1.0
            config.train.random_hdr = False
            config.train.albedo_only_render = True
            config.train.preview_every = 40
            config.train.checkpoint_every = int(args.iterations)

            # The full composition retains content and region-preservation terms,
            # while material/HDR terms are disabled for this representation check.
            config.losses.style = 1.0
            config.losses.patch_style = 2.0
            config.losses.content = 0.25
            config.losses.graph = 0.05
            config.losses.outside = 10.0
            config.losses.material_prior = 0.0
            config.losses.hdr_consistency = 0.0
            config.losses.texture_anchor = 0.2
            config.losses.texture_delta_tv = 0.05
            config.losses.color_mean = 2.0
            config.losses.render_color = 2.0
            config.losses.uv_continuity = 0.02
            config.losses.uv_distortion = 0.05
            config.losses.chart_seam = 0.2
            config.losses.uv_foldover = 0.2
            config.losses.uv_collision = 0.2

            if variant == "no_init_anchor":
                config.train.texture_init_strength = 0.0
                config.losses.texture_anchor = 0.0
            elif variant == "saliency_grid":
                config.train.reference_layout = "saliency_grid"
                config.train.reference_saliency_patches = 4
            elif variant == "saliency_focus":
                config.train.reference_layout = "saliency_focus"
                config.train.reference_focus_scale = float(args.focus_scale)
            elif variant in {
                "saliency_focus_repeat4", "saliency_focus_repeat6",
                "saliency_focus_512_repeat4",
            }:
                config.train.reference_layout = "saliency_focus"
                config.train.reference_focus_scale = float(args.focus_scale)
                config.train.atlas_reference_repeat = (
                    6 if variant == "saliency_focus_repeat6" else 4
                )
                if variant == "saliency_focus_512_repeat4":
                    config.train.texture_resolution = 512
            elif variant == "saliency_tile":
                config.train.reference_layout = "saliency_tile"
                config.train.reference_focus_scale = float(args.focus_scale)
                config.train.reference_tile_count = int(args.tile_count)
                config.train.reference_metric_tiles = True
                config.train.style_patch_reference = "original"
            elif variant == "saliency_tile_repeat2":
                config.train.reference_layout = "saliency_tile"
                config.train.reference_focus_scale = float(args.focus_scale)
                config.train.reference_tile_count = int(args.tile_count)
                config.train.reference_metric_tiles = True
                config.train.style_patch_reference = "original"
                config.train.atlas_reference_repeat = 2
            elif variant == "saliency_tile_metric1":
                config.train.reference_layout = "saliency_tile"
                config.train.reference_focus_scale = float(args.focus_scale)
                config.train.reference_tile_count = 1
                config.train.reference_metric_tiles = True
                config.train.style_patch_reference = "original"
                config.train.atlas_reference_repeat = 1
            elif variant == "saliency_motifs_metric1":
                config.train.reference_layout = "saliency_motifs"
                config.train.reference_saliency_patches = 4
                config.train.reference_focus_scale = float(args.focus_scale)
                config.train.reference_tile_count = 1
                config.train.reference_metric_tiles = True
                config.train.style_patch_reference = "original"
                config.train.atlas_reference_repeat = 1
            elif variant == "full_repeat4":
                config.train.atlas_reference_repeat = 4
            elif variant == "no_patch":
                config.losses.patch_style = 0.0
            elif variant == "no_uv_seam":
                config.losses.uv_continuity = 0.0
                config.losses.uv_distortion = 0.0
                config.losses.chart_seam = 0.0
                config.losses.uv_foldover = 0.0
                config.losses.uv_collision = 0.0

            name = f"{style_name}_{variant}"
            path = config_dir / f"{name}.yaml"
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
            runs.append({
                "style": style_name,
                "variant": variant,
                "config": str(path),
                "output": config.output_dir,
                "reference": config.reference_image,
            })

    manifest = {
        "purpose": "Atlas reference/patch/seam-UV composition ablation",
        "iterations": int(args.iterations),
        "variants": list(selected_variants),
        "focus_scale": float(args.focus_scale),
        "tile_count": int(args.tile_count),
        "runs": runs,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
