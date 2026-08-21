from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from raystyle.config import load_config


EXPERIMENTS = {
    "bicycle_starry": "configs/road_starry_graph_appearance.yaml",
    "bicycle_sunflowers": "configs/road_sunflowers.yaml",
    "stump_starry": "configs/stump_starry_test.yaml",
    "bulldozer_starry": "configs/kitchen_bulldozer_starry_test.yaml",
}


ALLOWED_PAIR_DIFFERENCES = {
    "output_dir",
    "train.texture_mapping",
    "train.reference_layout",
    "train.reference_metric_tiles",
    "losses.uv_continuity",
    "losses.uv_distortion",
    "losses.chart_seam",
    "losses.uv_foldover",
    "losses.uv_collision",
}


def _flatten(value, prefix: str = "") -> dict[str, object]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten(child, path))
    return flattened


def pairing_differences(triplanar: dict, atlas: dict) -> dict[str, dict[str, object]]:
    left = _flatten(triplanar)
    right = _flatten(atlas)
    return {
        key: {"triplanar": left.get(key), "atlas": right.get(key)}
        for key in sorted(left.keys() | right.keys())
        if left.get(key) != right.get(key)
    }


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((root / "raystyle").glob("*.py"))
    paths += [
        root / "scripts" / "preparation" / "prepare_atlas_validation.py",
        root / "scripts" / "training" / "run_atlas_validation.py",
    ]
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--texture-stage", type=int, default=150)
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--experiments", nargs="+", choices=tuple(EXPERIMENTS),
        default=list(EXPERIMENTS),
        help="paired scenarios to include (default: all)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    output = Path(args.output).expanduser().resolve()
    config_dir = output / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    records = []
    input_paths = set()
    pair_audits = {}
    for experiment in args.experiments:
        relative_config = EXPERIMENTS[experiment]
        base = load_config(root / relative_config)
        base.train.iterations = int(args.iterations)
        base.train.texture_stage_iterations = int(args.texture_stage)
        # Short matched sweeps found that both representations converge more
        # reliably at this texture budget.  Keeping these values identical is
        # part of the strict pairing contract.
        base.train.texture_learning_rate = 0.004
        base.train.texture_resolution = 512
        base.train.preview_every = 100
        # Formal 2k runs retain the requested 500/1000/1500/2000 snapshots;
        # shorter validation runs still keep a mid-run checkpoint.
        base.train.checkpoint_every = 500 if args.iterations >= 2000 else 200
        # Both representations see exactly the same untouched style image.
        # Atlas reference composition is only an initialization strategy.
        base.train.style_patch_reference = "original"
        base.train.reference_layout = "full"
        base.train.reference_metric_tiles = False
        # Repeating the complete painting hierarchy retained flower heads in
        # Sunflowers and restored star/whorl density across multi-chart objects.
        # Tri-planar stores the same value for an exact paired configuration;
        # only Atlas consumes it during source-atlas initialization.
        base.train.atlas_reference_repeat = 4
        # The multi-part Bulldozer segment needs slightly stronger spatial
        # coupling and less view-dependent capacity.  A paired 150/400-step
        # ablation showed that this removes its only remaining multiview-std
        # regression while retaining the Atlas composition advantage.  The
        # values are applied to both representations, preserving pairing.
        if experiment == "bulldozer_starry":
            base.losses.graph = 0.1
            base.train.residual_limit = 0.04
        paired_configs = {}
        for mapping in ("triplanar", "atlas"):
            config = type(base).from_dict(base.to_dict())
            config.train.texture_mapping = mapping
            if mapping == "atlas":
                # Values validated by the short reference/patch/geometry
                # composition ablation before the paired 400-iteration gate.
                config.losses.uv_continuity = 0.02
                config.losses.uv_distortion = 0.05
                config.losses.chart_seam = 0.2
                config.losses.uv_foldover = 0.2
                config.losses.uv_collision = 0.2
            config.output_dir = str(output / experiment / mapping)
            config_path = config_dir / f"{experiment}_{mapping}.yaml"
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
            paired_configs[mapping] = config.to_dict()
            records.append({
                "experiment": experiment,
                "mapping": mapping,
                "config": str(config_path),
                "output": config.output_dir,
                "reference_initialization": config.train.reference_layout,
                "atlas_reference_repeat": config.train.atlas_reference_repeat,
                "style_patch_reference": config.train.style_patch_reference,
            })
        differences = pairing_differences(
            paired_configs["triplanar"], paired_configs["atlas"],
        )
        unexpected = sorted(set(differences) - ALLOWED_PAIR_DIFFERENCES)
        if unexpected:
            raise ValueError(
                f"unpaired settings in {experiment}: {', '.join(unexpected)}"
            )
        pair_audits[experiment] = {
            "passed": True,
            "allowed_difference_paths": sorted(differences),
            "differences": differences,
        }
        input_paths.update((
            base.project_state, base.reference_image, base.dino_checkpoint,
        ))

    manifest = {
        "iterations": int(args.iterations),
        "texture_stage_iterations": int(args.texture_stage),
        "tag": args.tag,
        "experiments": list(args.experiments),
        "seed": 42,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True,
        ).splitlines(),
        "source_fingerprint": source_fingerprint(root),
        "inputs": {
            str(Path(path).resolve()): sha256(path) for path in sorted(input_paths)
        },
        "runs": records,
        "pairing_audits": pair_audits,
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
