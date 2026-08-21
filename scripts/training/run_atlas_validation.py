#!/usr/bin/env python3
"""Prepare, train, and evaluate the reproducible paired Atlas runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


EXPERIMENTS = (
    "bicycle_starry",
    "bicycle_sunflowers",
    "stump_starry",
    "bulldozer_starry",
)


def _run(command: list[str], root: Path):
    print("+", " ".join(command), flush=True)
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(root) if not existing else f"{root}{os.pathsep}{existing}"
    subprocess.run(command, cwd=root, check=True, env=environment)


def _load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare, train, and evaluate paired Atlas experiments.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--texture-stage", type=int, default=150)
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--experiments", nargs="+", choices=EXPERIMENTS, default=None,
        help="scenario subset; a new run defaults to all four",
    )
    parser.add_argument(
        "--reuse", action="store_true",
        help="reuse existing checkpoints and evaluation outputs",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    output = args.output.expanduser().resolve()
    python = sys.executable
    started = time.time()
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        command = [
            python, "-m", "scripts.preparation.prepare_atlas_validation",
            "--output", str(output),
            "--iterations", str(args.iterations),
            "--texture-stage", str(args.texture_stage),
        ]
        if args.experiments:
            command.extend(("--experiments", *args.experiments))
        if args.tag:
            command.extend(("--tag", args.tag))
        _run(command, root)
    elif not args.reuse:
        raise FileExistsError(
            f"validation output already exists: {output}; pass --reuse to verify it",
        )

    manifest = _load(manifest_path)
    expected = (int(args.iterations), int(args.texture_stage))
    actual = (manifest["iterations"], manifest["texture_stage_iterations"])
    if actual != expected:
        raise ValueError(f"manifest schedule {actual} does not match requested {expected}")
    if args.experiments:
        manifest_experiments = manifest.get(
            "experiments",
            list(dict.fromkeys(run["experiment"] for run in manifest["runs"])),
        )
        if list(args.experiments) != manifest_experiments:
            raise ValueError(
                f"manifest experiments {manifest_experiments} do not match "
                f"requested {list(args.experiments)}"
            )

    completed = []
    for run in manifest["runs"]:
        config = run["config"]
        output_dir = Path(run["output"])
        checkpoint = output_dir / "checkpoint_latest.pt"
        summary = output_dir / "evaluation" / "summary.json"
        if not (args.reuse and checkpoint.is_file()):
            if checkpoint.exists():
                raise FileExistsError(f"refusing to overwrite checkpoint: {checkpoint}")
            _run([python, "-m", "raystyle", "train", "--config", config], root)
        if not (args.reuse and summary.is_file()):
            _run([
                python, "-m", "raystyle", "evaluate", "--config", config,
                "--checkpoint", str(checkpoint),
            ], root)
        completed.append({
            "experiment": run["experiment"],
            "mapping": run["mapping"],
            "checkpoint": str(checkpoint),
            "summary": str(summary),
        })

    status = {
        "tag": manifest.get("tag"),
        "completed": True,
        "elapsed_seconds": time.time() - started,
        "runs": completed,
    }
    with (output / "pipeline_status.json").open("w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
