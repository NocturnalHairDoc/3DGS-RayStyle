import json
from pathlib import Path

import pytest

from raystyle.method_comparison import load_method_comparison


METHODS = ("dc", "full_sh", "pbr_only", "ours")


def _write_manifest(root: Path, experiments=("bicycle_starry",)) -> Path:
    runs = []
    for experiment in experiments:
        for method in METHODS:
            config = root / "configs" / f"{experiment}_{method}.yaml"
            checkpoint = root / experiment / method / "checkpoint_latest.pt"
            config.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            config.touch()
            checkpoint.touch()
            runs.append({
                "experiment": experiment,
                "method": method,
                "config": str(config.relative_to(root)),
                "output": str(checkpoint.parent.relative_to(root)),
            })
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"methods": list(METHODS), "runs": runs}))
    return manifest


def test_method_comparison_resolves_complete_experiment(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    comparison = load_method_comparison(manifest)
    assert comparison.experiment == "bicycle_starry"
    assert [entry.method for entry in comparison.entries] == list(METHODS)
    assert [entry.label for entry in comparison.entries] == [
        "DC-only", "All SH", "PBR-only", "Atlas Ours",
    ]
    assert all(entry.checkpoint_path.is_file() for entry in comparison.entries)


def test_method_comparison_requires_experiment_for_multi_scene_manifest(tmp_path: Path):
    manifest = _write_manifest(tmp_path, ("bicycle_starry", "stump_starry"))
    with pytest.raises(ValueError, match="--experiment is required"):
        load_method_comparison(manifest)
    selected = load_method_comparison(manifest, "stump_starry")
    assert selected.experiment == "stump_starry"


def test_method_comparison_rejects_missing_method(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["runs"] = [row for row in payload["runs"] if row["method"] != "ours"]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing methods: ours"):
        load_method_comparison(manifest)


def test_method_comparison_rejects_duplicate_method(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["runs"].append(dict(payload["runs"][0]))
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate method: dc"):
        load_method_comparison(manifest)
