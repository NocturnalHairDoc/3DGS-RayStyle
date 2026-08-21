from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SegmentBundleEntry:
    name: str
    config_path: Path
    checkpoint_path: Path


@dataclass(frozen=True)
class SegmentBundle:
    source: Path
    entries: tuple[SegmentBundleEntry, ...]


def load_segment_bundle(
    path: str | Path, *, require_files: bool = True,
) -> SegmentBundle:
    """Load a bundle of independently trained segment checkpoints."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"multi-segment bundle not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("multi-segment bundle must be a mapping with version: 1")
    rows = payload.get("segments")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("multi-segment bundle requires at least two segments")

    entries = []
    names = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"bundle segment {index} must be a mapping")
        name = str(row.get("name", "")).strip()
        if not name:
            raise ValueError(f"bundle segment {index} has no name")
        if name in names:
            raise ValueError(f"duplicate bundle segment name: {name}")
        names.add(name)
        resolved = {}
        for key in ("config", "checkpoint"):
            value = str(row.get(key, "")).strip()
            if not value:
                raise ValueError(f"bundle segment {name!r} has no {key}")
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = source.parent / candidate
            candidate = candidate.resolve()
            if require_files and not candidate.is_file():
                raise FileNotFoundError(
                    f"bundle segment {name!r} {key} does not exist: {candidate}"
                )
            resolved[key] = candidate
        entries.append(SegmentBundleEntry(
            name=name,
            config_path=resolved["config"],
            checkpoint_path=resolved["checkpoint"],
        ))
    return SegmentBundle(source=source, entries=tuple(entries))
