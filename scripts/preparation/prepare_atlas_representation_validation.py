#!/usr/bin/env python3
"""Prepare synthetic-texture Atlas validation configs with PBR/HDR disabled."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont
import yaml

from raystyle.config import load_config
from raystyle.atlas import ATLAS_VERSION


CASES = {
    "bicycle_checker": ("configs/road_starry_graph_appearance.yaml", "checker"),
    "bicycle_gradient": ("configs/road_starry_graph_appearance.yaml", "gradient"),
    "bicycle_orientation": ("configs/road_starry_graph_appearance.yaml", "orientation"),
    "bicycle_sunflower": ("configs/road_starry_graph_appearance.yaml", "sunflower"),
    "stump_checker": ("configs/stump_starry_test.yaml", "checker"),
    "bulldozer_gradient": ("configs/kitchen_bulldozer_starry_test.yaml", "gradient"),
}

# Synthetic references need a case-specific physical scale. These values are
# fixed from the short representation ablation and are intentionally not
# inferred from the validation renders.
CASE_ATLAS_SETTINGS = {
    "bicycle_checker": {"reference_repeat": 3, "chart_seam": 0.5},
    "bicycle_gradient": {"reference_repeat": 1, "chart_seam": 2.0},
    "bicycle_orientation": {"reference_repeat": 3, "chart_seam": 2.0},
    "bicycle_sunflower": {"reference_repeat": 4, "chart_seam": 0.2},
    "stump_checker": {"reference_repeat": 3, "chart_seam": 2.0},
    "bulldozer_gradient": {"reference_repeat": 1, "chart_seam": 2.0},
}


def _source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((root / "raystyle").glob("*.py"))
    paths += [
        root / "scripts" / "preparation" / "prepare_atlas_representation_validation.py",
    ]
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _checker(size: int):
    image = Image.new("RGB", (size, size), "#132a56")
    draw = ImageDraw.Draw(image)
    cells = 8
    for y in range(cells):
        for x in range(cells):
            color = "#f4b942" if (x + y) % 2 else "#173f8a"
            draw.rectangle(
                (x * size // cells, y * size // cells, (x + 1) * size // cells, (y + 1) * size // cells),
                fill=color,
            )
    return image


def _gradient(size: int):
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            pixels[x, y] = (
                round(255 * x / (size - 1)),
                round(255 * y / (size - 1)),
                round(255 * (1 - x / (size - 1))),
            )
    return image


def _orientation_pattern(size: int, *, cells: int, stroke_fraction: float):
    image = Image.new("RGB", (size, size), "#f5f1df")
    draw = ImageDraw.Draw(image)
    cell = size / cells
    font = _font(max(8, round(cell * 0.22)))
    for row in range(cells):
        for column in range(cells):
            x0, y0 = column * cell, row * cell
            cy = y0 + cell * 0.55
            left, tip = x0 + cell * 0.18, x0 + cell * 0.82
            draw.line(
                (left, cy, tip, cy), fill="#d2232a",
                width=max(2, round(cell * stroke_fraction)),
            )
            draw.polygon(
                ((tip, cy), (x0 + cell * 0.61, y0 + cell * 0.34),
                 (x0 + cell * 0.61, y0 + cell * 0.76)),
                fill="#d2232a",
            )
            # Blue-left/green-top markers make mirroring and 90-degree flips
            # unambiguous even when only part of a tile is visible.
            radius = cell * 0.055
            draw.ellipse(
                (left - radius, cy - radius, left + radius, cy + radius),
                fill="#123c73",
            )
            draw.text(
                (x0 + cell * 0.08, y0 + cell * 0.05), "R",
                fill="#17854b", font=font,
            )
    return image


def _orientation(size: int):
    return _orientation_pattern(size, cells=8, stroke_fraction=0.10)


def _sunflower(size: int):
    image = Image.new("RGB", (size, size), "#327447")
    draw = ImageDraw.Draw(image)
    flowers = ((0.27, 0.35, 0.16), (0.68, 0.30, 0.19), (0.51, 0.72, 0.22))
    for fx, fy, radius in flowers:
        cx, cy, r = fx * size, fy * size, radius * size
        for step in range(16):
            angle = step * 2 * math.pi / 16
            px = cx + math.cos(angle) * r * 0.75
            py = cy + math.sin(angle) * r * 0.75
            pr = r * 0.38
            draw.ellipse((px - pr, py - pr * 0.55, px + pr, py + pr * 0.55), fill="#f7c928")
        draw.ellipse((cx - r * 0.42, cy - r * 0.42, cx + r * 0.42, cy + r * 0.42), fill="#633617")
        draw.ellipse((cx - r * 0.22, cy - r * 0.22, cx + r * 0.22, cy + r * 0.22), fill="#29170c")
    return image


def write_synthetic_references(reference_dir: Path, size: int = 512) -> dict[str, str]:
    """Write the deterministic representation-test textures."""
    reference_dir.mkdir(parents=True, exist_ok=True)
    generators = {
        "checker": _checker,
        "gradient": _gradient,
        "orientation": _orientation,
        "sunflower": _sunflower,
    }
    references = {}
    for name, generator in generators.items():
        path = reference_dir / f"{name}.png"
        generator(size).save(path)
        references[name] = str(path)
    return references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument(
        "--cases", nargs="+", choices=tuple(CASES), default=list(CASES),
        help="representation cases to prepare (default: all)",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.expanduser().resolve()
    config_dir = output / "configs"
    reference_dir = output / "references"
    config_dir.mkdir(parents=True, exist_ok=False)
    references = write_synthetic_references(reference_dir)

    records = []
    for case in args.cases:
        base_path, texture = CASES[case]
        atlas_settings = CASE_ATLAS_SETTINGS[case]
        config = load_config(root / base_path)
        config.reference_image = references[texture]
        config.output_dir = str(output / case)
        config.train.iterations = int(args.iterations)
        config.train.texture_stage_iterations = int(args.iterations)
        config.train.texture_mapping = "atlas"
        config.train.atlas_source_layout = (
            "developed" if case == "stump_checker"
            else "projected" if case == "bulldozer_gradient"
            else "packed"
        )
        config.train.atlas_reference_repeat = atlas_settings["reference_repeat"]
        config.train.texture_init_strength = 1.0
        config.train.random_hdr = False
        config.train.albedo_only_render = True
        config.train.preview_every = 40
        config.train.checkpoint_every = int(args.iterations)
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
        config.losses.chart_seam = atlas_settings["chart_seam"]
        config.losses.uv_foldover = 0.2
        config.losses.uv_collision = 0.2
        path = config_dir / f"{case}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
        records.append({
            "case": case,
            "texture": texture,
            "config": str(path),
            "output": config.output_dir,
            "atlas_source_layout": config.train.atlas_source_layout,
            "atlas_reference_repeat": config.train.atlas_reference_repeat,
            "chart_seam_weight": config.losses.chart_seam,
        })
    manifest = {
        "purpose": "Atlas representation validation without PBR/HDR",
        "atlas_version": ATLAS_VERSION,
        "iterations": int(args.iterations),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        ).strip(),
        "source_fingerprint": _source_fingerprint(root),
        "cases": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
