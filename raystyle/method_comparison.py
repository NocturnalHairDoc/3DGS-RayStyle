from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .config import METHODS


METHOD_LABELS = {
    "dc": "DC-only",
    "full_sh": "All SH",
    "pbr_only": "PBR-only",
    "ours": "Atlas Ours",
}


@dataclass(frozen=True)
class MethodComparisonEntry:
    method: str
    label: str
    config_path: Path
    checkpoint_path: Path


@dataclass(frozen=True)
class MethodComparison:
    source: Path
    experiment: str
    entries: tuple[MethodComparisonEntry, ...]


def _resolve(source: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    return path.resolve()


def load_method_comparison(
    manifest_path: str | Path,
    experiment: str | None = None,
    *,
    require_files: bool = True,
) -> MethodComparison:
    """Resolve one strict four-method experiment from a baseline manifest."""
    source = Path(manifest_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"method baseline manifest not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        raise ValueError("method baseline manifest must contain a runs list")

    experiments = tuple(dict.fromkeys(
        str(row.get("experiment", ""))
        for row in payload["runs"] if isinstance(row, dict) and row.get("experiment")
    ))
    if experiment is None:
        if len(experiments) != 1:
            choices = ", ".join(experiments) or "none"
            raise ValueError(
                "--experiment is required when the manifest contains multiple "
                f"experiments; available: {choices}"
            )
        experiment = experiments[0]
    if experiment not in experiments:
        raise ValueError(
            f"unknown experiment {experiment!r}; available: {', '.join(experiments)}"
        )

    experiment_rows = [
        row for row in payload["runs"]
        if isinstance(row, dict) and row.get("experiment") == experiment
    ]
    rows = {}
    for row in experiment_rows:
        method = str(row.get("method", ""))
        if method not in METHODS:
            raise ValueError(f"{experiment!r} contains unknown method: {method!r}")
        if method in rows:
            raise ValueError(f"{experiment!r} contains duplicate method: {method}")
        rows[method] = row
    missing = [method for method in METHODS if method not in rows]
    if missing:
        raise ValueError(
            f"experiment {experiment!r} is missing methods: {', '.join(missing)}"
        )

    declared = payload.get("methods", METHODS)
    ordered_methods = [method for method in declared if method in METHODS]
    ordered_methods.extend(method for method in METHODS if method not in ordered_methods)
    entries = []
    for method in ordered_methods:
        row = rows[method]
        config_value = str(row.get("config", "")).strip()
        output_value = str(row.get("output", "")).strip()
        checkpoint_value = str(row.get("checkpoint", "")).strip()
        if not config_value:
            raise ValueError(f"{experiment}/{method} has no config path")
        if not checkpoint_value:
            if not output_value:
                raise ValueError(f"{experiment}/{method} has no output or checkpoint path")
            checkpoint_value = str(Path(output_value) / "checkpoint_latest.pt")
        config_path = _resolve(source, config_value)
        checkpoint_path = _resolve(source, checkpoint_value)
        if require_files and not config_path.is_file():
            raise FileNotFoundError(
                f"{experiment}/{method} config does not exist: {config_path}"
            )
        if require_files and not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"{experiment}/{method} checkpoint does not exist: {checkpoint_path}"
            )
        entries.append(MethodComparisonEntry(
            method=method,
            label=METHOD_LABELS[method],
            config_path=config_path,
            checkpoint_path=checkpoint_path,
        ))
    return MethodComparison(source, experiment, tuple(entries))
