from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneRecord:
    scene_id: str
    version: str
    name: str
    model_path: str
    source_path: str
    images: str
    resolution: int
    white_background: bool
    latest_iteration: int
    source_exists: bool

    def to_dict(self):
        return asdict(self)


def parse_cfg_args(path: str | Path) -> dict:
    """Parse the repository's Namespace(...) cfg without executing it."""
    source = Path(path)
    expression = ast.parse(source.read_text(encoding="utf-8").strip(), mode="eval").body
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ValueError(f"unsupported cfg_args expression in {source}")
    if expression.func.id != "Namespace" or expression.args:
        raise ValueError(f"expected Namespace(keyword=...) in {source}")
    result = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ValueError(f"expanded keywords are not allowed in {source}")
        result[keyword.arg] = ast.literal_eval(keyword.value)
    return result


def _latest_iteration(model_path: Path) -> int:
    iterations = []
    for ply in model_path.glob("point_cloud/iteration_*/scene_point_cloud.ply"):
        try:
            iterations.append(int(ply.parent.name.removeprefix("iteration_")))
        except ValueError:
            continue
    return max(iterations, default=-1)


def discover_scenes(workspace_root: str | Path) -> list[SceneRecord]:
    root = Path(workspace_root).expanduser().resolve()
    records = []
    for cfg_path in sorted(root.glob("3DGS-RTMaterial*/output*/**/cfg_args")):
        if "submodules" in cfg_path.parts:
            continue
        model_path = cfg_path.parent
        latest = _latest_iteration(model_path)
        if latest < 0:
            continue
        try:
            cfg = parse_cfg_args(cfg_path)
        except (SyntaxError, ValueError):
            continue
        relative = model_path.relative_to(root)
        version = relative.parts[0]
        name = model_path.name
        source_path = str(cfg.get("source_path", ""))
        records.append(SceneRecord(
            scene_id=f"{version}:{name}",
            version=version,
            name=name,
            model_path=str(model_path.resolve()),
            source_path=source_path,
            images=str(cfg.get("images", "images")),
            resolution=int(cfg.get("resolution", -1)),
            white_background=bool(cfg.get("white_background", False)),
            latest_iteration=latest,
            source_exists=bool(source_path and Path(source_path).is_dir()),
        ))
    return records


def resolve_scene(workspace_root: str | Path, selector: str) -> SceneRecord:
    records = discover_scenes(workspace_root)
    exact = [record for record in records if record.scene_id == selector]
    if exact:
        record = exact[0]
    else:
        matches = [record for record in records if record.name == selector]
        if not matches:
            available = ", ".join(record.scene_id for record in records)
            raise ValueError(f"scene {selector!r} was not found; available: {available}")
        matches.sort(
            key=lambda item: (
                item.source_exists,
                item.name != "smoke_scene",
                item.latest_iteration,
                item.version == "3DGS-RTMaterial",
            ),
            reverse=True,
        )
        record = matches[0]
    if not record.source_exists:
        raise FileNotFoundError(
            f"scene {record.scene_id} exists, but its cfg_args source_path is missing: "
            f"{record.source_path}"
        )
    return record

