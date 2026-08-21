from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from raystyle.config import load_config


MODES = ("albedo", "diffuse_only", "pbr")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare matched render-mode evaluations from trained Atlas checkpoints.",
    )
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--experiments", nargs="+",
        default=("bicycle_starry", "stump_starry", "bulldozer_starry"),
    )
    args = parser.parse_args()

    validation = Path(args.validation).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    config_dir = output / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    runs = []
    for experiment in args.experiments:
        source_config = validation / "configs" / f"{experiment}_atlas.yaml"
        checkpoint = validation / experiment / "atlas" / "checkpoint_latest.pt"
        if not source_config.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"missing Atlas source for {experiment}")
        for mode in MODES:
            config = load_config(source_config)
            config.train.render_mode = mode
            config.train.albedo_only_render = False
            config.output_dir = str(output / experiment / mode)
            config_path = config_dir / f"{experiment}_{mode}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
            runs.append({
                "experiment": experiment,
                "render_mode": mode,
                "config": str(config_path),
                "checkpoint": str(checkpoint),
                "output": config.output_dir,
            })
    manifest = {
        "source_validation": str(validation),
        "modes": list(MODES),
        "runs": runs,
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
