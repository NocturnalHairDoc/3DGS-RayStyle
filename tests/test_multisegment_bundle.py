from pathlib import Path

import pytest

from raystyle.multisegment_bundle import load_segment_bundle


def test_bundle_resolves_relative_paths_and_preserves_order(tmp_path: Path):
    for name in ("a.yaml", "a.pt", "b.yaml", "b.pt"):
        (tmp_path / name).touch()
    manifest = tmp_path / "bundle.yaml"
    manifest.write_text(
        "version: 1\nsegments:\n"
        "  - name: road-left\n    config: a.yaml\n    checkpoint: a.pt\n"
        "  - name: road-right\n    config: b.yaml\n    checkpoint: b.pt\n",
        encoding="utf-8",
    )
    bundle = load_segment_bundle(manifest)
    assert [entry.name for entry in bundle.entries] == ["road-left", "road-right"]
    assert bundle.entries[0].config_path == (tmp_path / "a.yaml").resolve()


def test_bundle_rejects_duplicate_segment_names(tmp_path: Path):
    manifest = tmp_path / "bundle.yaml"
    manifest.write_text(
        "version: 1\nsegments:\n"
        "  - {name: road, config: a, checkpoint: b}\n"
        "  - {name: road, config: c, checkpoint: d}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_segment_bundle(manifest, require_files=False)


def test_bundle_requires_two_segments(tmp_path: Path):
    manifest = tmp_path / "bundle.yaml"
    manifest.write_text(
        "version: 1\nsegments:\n"
        "  - {name: road, config: a, checkpoint: b}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least two"):
        load_segment_bundle(manifest, require_files=False)
