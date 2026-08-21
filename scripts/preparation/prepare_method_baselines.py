#!/usr/bin/env python3
"""Prepare strictly paired DC, full-SH, PBR-only and Atlas configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from raystyle.config import load_config


EXPERIMENTS = (
    "bicycle_starry",
    "bicycle_sunflowers",
    "stump_starry",
    "bulldozer_starry",
)
METHODS = ("dc", "full_sh", "pbr_only", "ours")
ALLOWED_DIFFERENCES = {"method", "output_dir"}


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _flatten(value, prefix: str = "") -> dict[str, object]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten(child, path))
    return flattened


def pairing_differences(left: dict, right: dict) -> dict[str, dict[str, object]]:
    baseline = _flatten(left)
    candidate = _flatten(right)
    return {
        key: {"reference": baseline.get(key), "candidate": candidate.get(key)}
        for key in sorted(baseline.keys() | candidate.keys())
        if baseline.get(key) != candidate.get(key)
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((root / "raystyle").glob("*.py"))
    paths += [
        root / "scripts" / "preparation" / "prepare_method_baselines.py",
        root / "scripts" / "training" / "run_method_baselines.py",
    ]
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
        "--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    source_validation = args.source_validation.expanduser().resolve()
    output = args.output.expanduser().resolve()
    config_dir = output / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    source_manifest = _load(source_validation / "manifest.json")
    records = []
    audits = {}
    input_paths = set()

    for experiment in args.experiments:
        source_config = source_validation / "configs" / f"{experiment}_atlas.yaml"
        if not source_config.is_file():
            raise FileNotFoundError(f"missing accepted Atlas config: {source_config}")
        base = load_config(source_config)
        base.train.iterations = int(args.iterations)
        base.train.texture_stage_iterations = min(
            int(args.texture_stage), int(args.iterations),
        )
        base.train.preview_every = min(100, int(args.iterations))
        base.train.checkpoint_every = (
            500 if args.iterations >= 2000 else max(1, min(200, args.iterations))
        )
        base.evaluation.max_views = int(args.max_views)
        base.evaluation.heldout_env_count = int(args.heldout_env_count)
        configs = {}
        for method in METHODS:
            config = type(base).from_dict(base.to_dict())
            config.method = method
            config.output_dir = str(output / experiment / method)
            config_path = config_dir / f"{experiment}_{method}.yaml"
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
            configs[method] = config.to_dict()
            records.append({
                "experiment": experiment,
                "method": method,
                "config": str(config_path),
                "output": config.output_dir,
                "reuse_from": (
                    str((source_validation / experiment / "atlas").resolve())
                    if method == "ours" and args.reuse_ours_from_source else None
                ),
            })
        reference = configs["ours"]
        audits[experiment] = {}
        for method in METHODS:
            differences = pairing_differences(reference, configs[method])
            unexpected = sorted(set(differences) - ALLOWED_DIFFERENCES)
            if unexpected:
                raise ValueError(
                    f"unpaired settings in {experiment}/{method}: "
                    f"{', '.join(unexpected)}"
                )
            audits[experiment][method] = {
                "passed": True,
                "allowed_difference_paths": sorted(differences),
                "differences": differences,
            }
        input_paths.update((
            base.project_state, base.reference_image, base.dino_checkpoint,
        ))

    manifest = {
        "tag": args.tag,
        "iterations": int(args.iterations),
        "texture_stage_iterations": min(int(args.texture_stage), int(args.iterations)),
        "max_views": int(args.max_views),
        "heldout_env_count": int(args.heldout_env_count),
        "experiments": list(args.experiments),
        "methods": list(METHODS),
        "seed": 42,
        "source_validation": str(source_validation),
        "source_validation_tag": source_manifest.get("tag"),
        "reuse_ours_from_source": bool(args.reuse_ours_from_source),
        "protocol_note": "All serialized settings are identical except method and output_dir. Ours alone consumes the configured texture stage because the other methods have no texture field.",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True,
        ).splitlines(),
        "source_fingerprint": _source_fingerprint(root),
        "inputs": {
            str(Path(path).resolve()): _sha256(path) for path in sorted(input_paths)
        },
        "runs": records,
        "pairing_audits": audits,
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
