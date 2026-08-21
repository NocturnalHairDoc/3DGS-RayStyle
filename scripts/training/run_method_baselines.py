#!/usr/bin/env python3
"""Run the reproducible four-method RayStyle baseline pipeline."""

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


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run(command: list[str], root: Path) -> None:
    print("+", " ".join(command), flush=True)
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(root) if not existing else f"{root}{os.pathsep}{existing}"
    subprocess.run(command, cwd=root, check=True, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--texture-stage", type=int, default=150)
    parser.add_argument("--max-views", type=int, default=12)
    parser.add_argument("--heldout-env-count", type=int, default=3)
    parser.add_argument("--tag")
    parser.add_argument(
        "--reuse-ours-from-source", action="store_true",
        help="reuse the accepted Atlas checkpoint/evaluation from source-validation",
    )
    parser.add_argument(
        "--experiments", nargs="+", choices=EXPERIMENTS, default=None,
    )
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    output = args.output.expanduser().resolve()
    manifest_path = output / "manifest.json"
    python = sys.executable
    started = time.time()
    if not manifest_path.is_file():
        command = [
            python, "-m", "scripts.preparation.prepare_method_baselines",
            "--source-validation", str(args.source_validation.expanduser().resolve()),
            "--output", str(output),
            "--iterations", str(args.iterations),
            "--texture-stage", str(args.texture_stage),
            "--max-views", str(args.max_views),
            "--heldout-env-count", str(args.heldout_env_count),
        ]
        if args.tag:
            command.extend(("--tag", args.tag))
        if args.reuse_ours_from_source:
            command.append("--reuse-ours-from-source")
        if args.experiments:
            command.extend(("--experiments", *args.experiments))
        _run(command, root)
    elif not args.reuse:
        raise FileExistsError(f"baseline output already exists: {output}; pass --reuse")

    manifest = _load(manifest_path)
    expected = (
        int(args.iterations), min(int(args.texture_stage), int(args.iterations)),
        int(args.max_views), int(args.heldout_env_count),
    )
    actual = (
        manifest["iterations"], manifest["texture_stage_iterations"],
        manifest["max_views"], manifest["heldout_env_count"],
    )
    if actual != expected:
        raise ValueError(f"manifest protocol {actual} does not match requested {expected}")
    if args.experiments and list(args.experiments) != manifest["experiments"]:
        raise ValueError("requested experiments do not match the existing manifest")
    if bool(args.reuse_ours_from_source) != bool(
        manifest.get("reuse_ours_from_source", False)
    ):
        raise ValueError("reuse-ours setting does not match the existing manifest")

    completed = []
    for run in manifest["runs"]:
        output_dir = Path(run["output"])
        reuse_from = run.get("reuse_from")
        if reuse_from:
            source_dir = Path(reuse_from)
            # Early versions linked the whole accepted output directory, which
            # also reused an evaluation protocol that could have fewer views.
            # Import only the immutable checkpoint and evaluate it locally.
            if output_dir.is_symlink():
                if output_dir.resolve() != source_dir.resolve():
                    raise ValueError(f"unexpected reuse symlink target: {output_dir}")
                output_dir.unlink()
            output_dir.mkdir(parents=True, exist_ok=True)
            imported_checkpoint = output_dir / "checkpoint_latest.pt"
            if not imported_checkpoint.exists():
                imported_checkpoint.symlink_to(source_dir / "checkpoint_latest.pt")
            source_texture = source_dir / "texture_latest.png"
            imported_texture = output_dir / "texture_latest.png"
            if source_texture.is_file() and not imported_texture.exists():
                imported_texture.symlink_to(source_texture)
        checkpoint = output_dir / "checkpoint_latest.pt"
        summary = output_dir / "evaluation" / "summary.json"
        if not ((args.reuse or bool(reuse_from)) and checkpoint.is_file()):
            _run([python, "-m", "raystyle", "train", "--config", run["config"]], root)
        if not (args.reuse and summary.is_file()):
            _run([
                python, "-m", "raystyle", "evaluate", "--config", run["config"],
                "--checkpoint", str(checkpoint),
            ], root)
        completed.append({
            "experiment": run["experiment"],
            "method": run["method"],
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
