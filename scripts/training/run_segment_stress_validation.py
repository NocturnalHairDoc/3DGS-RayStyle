#!/usr/bin/env python3
"""Prepare, train, and evaluate Atlas segment stress tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _run(command: list[str], root: Path) -> None:
    print("+", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(root) if not existing else f"{root}{os.pathsep}{existing}"
    subprocess.run(command, cwd=root, check=True, env=environment)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--segment-root", type=Path, default=None)
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    output = args.output.expanduser().resolve()
    python = sys.executable
    started = time.time()
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        command = [
            python, "-m", "scripts.preparation.prepare_segment_stress_validation",
            "--output", str(output), "--iterations", str(args.iterations),
        ]
        if args.segment_root is not None:
            command.extend(("--segment-root", str(args.segment_root)))
        if args.cases:
            command.extend(("--cases", *args.cases))
        _run(command, root)
    elif not args.reuse:
        raise FileExistsError(f"output exists: {output}; use --reuse to resume")

    manifest = _load(manifest_path)
    if manifest["iterations"] != args.iterations:
        raise ValueError(
            f"manifest iterations {manifest['iterations']} != requested {args.iterations}"
        )
    completed = []
    for case in manifest["cases"]:
        config = case["config"]
        case_output = Path(case["output"])
        checkpoint = case_output / "checkpoint_latest.pt"
        summary = case_output / "evaluation" / "summary.json"
        if not checkpoint.is_file():
            _run([python, "-m", "raystyle", "train", "--config", config], root)
        if not summary.is_file():
            _run([
                python, "-m", "raystyle", "evaluate", "--config", config,
                "--checkpoint", str(checkpoint),
            ], root)
        completed.append({
            "case": case["case"],
            "checkpoint": str(checkpoint),
            "summary": str(summary),
        })

    status = {
        "completed": True,
        "elapsed_seconds": time.time() - started,
        "cases": completed,
    }
    (output / "pipeline_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
