from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from raystyle.config import load_config


VARIANTS = {
    "baseline": {},
    "anchor_005": {"losses.texture_anchor": 0.05},
    "seam_010": {"losses.chart_seam": 0.1},
    "texture_lr_003": {"train.texture_learning_rate": 0.003},
    "texture_lr_004": {"train.texture_learning_rate": 0.004},
    "texture_512": {"train.texture_resolution": 512},
    "texture_lr_004_512": {
        "train.texture_learning_rate": 0.004,
        "train.texture_resolution": 512,
    },
    "texture_lr_004_512_repeat2": {
        "train.texture_learning_rate": 0.004,
        "train.texture_resolution": 512,
        "train.atlas_reference_repeat": 2,
    },
    "texture_lr_004_512_repeat4": {
        "train.texture_learning_rate": 0.004,
        "train.texture_resolution": 512,
        "train.atlas_reference_repeat": 4,
    },
    "residual_limit_004": {"train.residual_limit": 0.04},
    "sh_dc": {"train.sh_degree": 0},
    "graph_010": {"losses.graph": 0.1},
    "graph_010_residual_limit_004": {
        "losses.graph": 0.1,
        "train.residual_limit": 0.04,
    },
}


def _assign(config, path: str, value):
    section, name = path.split(".", 1)
    setattr(getattr(config, section), name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment", default="bulldozer_starry")
    parser.add_argument("--mapping", choices=("atlas", "triplanar"), default="atlas")
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--texture-stage", type=int, default=60)
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS),
        default=list(VARIANTS),
    )
    args = parser.parse_args()

    validation = Path(args.validation).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    config_dir = output / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    source = validation / "configs" / f"{args.experiment}_{args.mapping}.yaml"
    if not source.is_file():
        raise FileNotFoundError(source)
    runs = []
    for variant in args.variants:
        changes = VARIANTS[variant]
        config = load_config(source)
        config.train.iterations = args.iterations
        config.train.texture_stage_iterations = args.texture_stage
        config.train.preview_every = 50
        config.train.checkpoint_every = args.iterations
        for path, value in changes.items():
            _assign(config, path, value)
        config.output_dir = str(output / args.experiment / variant)
        config_path = config_dir / f"{args.experiment}_{variant}.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
        runs.append({
            "experiment": args.experiment,
            "variant": variant,
            "changes": changes,
            "config": str(config_path),
            "checkpoint": str(Path(config.output_dir) / "checkpoint_latest.pt"),
            "output": config.output_dir,
        })
    manifest = {
        "source_validation": str(validation),
        "mapping": args.mapping,
        "iterations": args.iterations,
        "texture_stage_iterations": args.texture_stage,
        "runs": runs,
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
