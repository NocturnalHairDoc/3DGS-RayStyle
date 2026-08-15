from pathlib import Path

from raystyle.scene_catalog import discover_scenes, parse_cfg_args, resolve_scene


def _make_scene(root: Path, version="3DGS-RTMaterial-V1", name="room", iteration=30000):
    source = root / "data" / name
    source.mkdir(parents=True)
    model = root / version / "output" / name
    ply = model / "point_cloud" / f"iteration_{iteration}" / "scene_point_cloud.ply"
    ply.parent.mkdir(parents=True)
    ply.write_bytes(b"ply")
    (model / "cfg_args").write_text(
        "Namespace(source_path=%r, images='images_4', resolution=4, "
        "white_background=True)" % str(source),
        encoding="utf-8",
    )
    return model, source


def test_cfg_parser_does_not_need_eval(tmp_path):
    model, source = _make_scene(tmp_path)
    values = parse_cfg_args(model / "cfg_args")
    assert values["source_path"] == str(source)
    assert values["white_background"] is True


def test_discover_and_resolve_sibling_scene(tmp_path):
    model, source = _make_scene(tmp_path)
    records = discover_scenes(tmp_path)
    assert len(records) == 1
    record = resolve_scene(tmp_path, "3DGS-RTMaterial-V1:room")
    assert record.model_path == str(model)
    assert record.source_path == str(source)
    assert record.resolution == 4

