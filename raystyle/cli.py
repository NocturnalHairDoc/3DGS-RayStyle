from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import METHODS, load_config


def _overridden_config(args):
    config = load_config(args.config)
    if getattr(args, "method", None):
        config.method = args.method
    if getattr(args, "output_dir", None):
        config.output_dir = str(Path(args.output_dir).expanduser().resolve())
    config.validate()
    return config


def _train(args):
    from .trainer import Trainer

    config = _overridden_config(args)
    if args.iterations is not None:
        config.train.iterations = args.iterations
        config.validate()
    checkpoint = Trainer(config, resume_checkpoint=args.resume).train()
    print(f"checkpoint: {checkpoint}")


def _evaluate(args):
    from .evaluation import evaluate

    config = _overridden_config(args)
    summary = evaluate(config, args.checkpoint)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _view(args):
    from .viewer import view

    config = _overridden_config(args)
    view(config, args.checkpoint, scale=args.scale)


def _view_bundle(args):
    from .viewer import view_bundle

    view_bundle(args.bundle, scale=args.scale)


def _view_methods(args):
    from .viewer import view_methods

    view_methods(args.manifest, args.experiment, scale=args.scale)


def _validate_orbit(args):
    from .orbit_validation import validate_orbit

    config = _overridden_config(args)
    output = args.orbit_output or (Path(config.output_dir) / "orbit_validation")
    summary = validate_orbit(
        config, args.checkpoint, output,
        frame_count=args.frames, fps=args.fps,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _inspect(args):
    from .project_state import segment_inventory

    rows = segment_inventory(args.project_state)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


def _compare(args):
    summaries = []
    for path in args.summaries:
        with Path(path).open("r", encoding="utf-8") as handle:
            summaries.append(json.load(handle))
    keys = sorted({key for row in summaries for key in row if "/" in key})
    header = ["method", *keys]
    print(",".join(header))
    for row in summaries:
        print(",".join(str(row.get(key, "")) for key in header))


def _list_scenes(args):
    from .scene_catalog import discover_scenes

    rows = [record.to_dict() for record in discover_scenes(args.workspace_root)]
    if not args.include_missing_source:
        rows = [row for row in rows if row["source_exists"]]
    print(json.dumps(rows, indent=2, ensure_ascii=False))


def _validate_scene(args):
    import torch

    from .backend import LegacyGaussianBackend
    from .io_utils import save_image
    from .scene_catalog import resolve_scene

    record = resolve_scene(args.workspace_root, args.scene)
    backend = LegacyGaussianBackend(
        args.legacy_root, record.model_path, record.source_path,
        record.images, record.resolution, record.white_background,
    )
    image = backend.render_original(backend.train_cameras[0]).detach()
    if args.output:
        save_image(args.output, image)
    result = {
        **record.to_dict(),
        "points": backend.point_count,
        "train_cameras": len(backend.train_cameras),
        "test_cameras": len(backend.test_cameras),
        "image_shape": list(image.shape),
        "finite": bool(torch.isfinite(image).all()),
        "output": str(Path(args.output).resolve()) if args.output else None,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _feature_segment(args):
    from .feature_segment import segment_from_feature_clicks

    config = load_config(args.config, require_inputs=False)
    result = segment_from_feature_clicks(
        config, [tuple(value) for value in args.click], args.output,
        negative_clicks=[tuple(value) for value in args.negative_click],
        camera_index=args.camera_index, threshold=args.threshold,
        negative_margin=args.negative_margin,
        scale=args.feature_scale, preview_path=args.preview,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _sam_segment(args):
    from .feature_segment import segment_from_sam_masks

    config = load_config(args.config, require_inputs=False)
    result = segment_from_sam_masks(
        config, args.mask_index, args.output, camera_index=args.camera_index,
        dilation=args.dilation, preview_path=args.preview,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser(prog="raystyle")
    subcommands = parser.add_subparsers(dest="command", required=True)

    train = subcommands.add_parser("train", help="train one method")
    train.add_argument("--config", required=True)
    train.add_argument("--method", choices=METHODS)
    train.add_argument("--output-dir")
    train.add_argument("--resume", help="resume appearance state from a RayStyle checkpoint")
    train.add_argument(
        "--iterations", type=int, help="temporarily override the configured iteration count",
    )
    train.set_defaults(function=_train)

    evaluate = subcommands.add_parser("evaluate", help="evaluate a trained checkpoint")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--method", choices=METHODS)
    evaluate.add_argument("--output-dir")
    evaluate.set_defaults(function=_evaluate)

    viewer = subcommands.add_parser("view", help="interactively view a RayStyle checkpoint")
    viewer.add_argument("--config", required=True)
    viewer.add_argument("--checkpoint", required=True)
    viewer.add_argument("--method", choices=METHODS)
    viewer.add_argument("--output-dir")
    viewer.add_argument("--scale", type=float, default=2.0,
                        help="render resolution divisor (default: 2)")
    viewer.set_defaults(function=_view)

    bundle_viewer = subcommands.add_parser(
        "view-bundle", help="view multiple independent segment checkpoints together",
    )
    bundle_viewer.add_argument("--bundle", required=True)
    bundle_viewer.add_argument(
        "--scale", type=float, default=2.0,
        help="render resolution divisor (default: 2)",
    )
    bundle_viewer.set_defaults(function=_view_bundle)

    method_viewer = subcommands.add_parser(
        "view-methods", help="compare four paired baseline methods in one viewer",
    )
    method_viewer.add_argument("--manifest", required=True)
    method_viewer.add_argument(
        "--experiment",
        help="experiment name; required when the manifest contains more than one",
    )
    method_viewer.add_argument(
        "--scale", type=float, default=2.0,
        help="render resolution divisor (default: 2)",
    )
    method_viewer.set_defaults(function=_view_methods)

    orbit = subcommands.add_parser(
        "validate-orbit", help="render and measure a closed checkpoint camera path",
    )
    orbit.add_argument("--config", required=True)
    orbit.add_argument("--checkpoint", required=True)
    orbit.add_argument("--method", choices=METHODS)
    orbit.add_argument("--output-dir")
    orbit.add_argument("--orbit-output")
    orbit.add_argument("--frames", type=int, default=48)
    orbit.add_argument("--fps", type=int, default=12)
    orbit.set_defaults(function=_validate_orbit)

    inspect = subcommands.add_parser("inspect-segments", help="list segment ids in a GUI project state")
    inspect.add_argument("--project-state", required=True)
    inspect.set_defaults(function=_inspect)

    scenes = subcommands.add_parser("list-scenes", help="discover reusable scenes in sibling project versions")
    scenes.add_argument("--workspace-root", default="..")
    scenes.add_argument("--include-missing-source", action="store_true")
    scenes.set_defaults(function=_list_scenes)

    validate_scene = subcommands.add_parser("validate-scene", help="load and render one sibling scene")
    validate_scene.add_argument("--workspace-root", default="..")
    validate_scene.add_argument("--scene", required=True)
    validate_scene.add_argument("--legacy-root", required=True)
    validate_scene.add_argument("--output")
    validate_scene.set_defaults(function=_validate_scene)

    feature_segment = subcommands.add_parser(
        "feature-segment", help="create a point mask from calibrated-view feature clicks",
    )
    feature_segment.add_argument("--config", required=True)
    feature_segment.add_argument("--camera-index", type=int, default=0)
    feature_segment.add_argument(
        "--click", type=int, nargs=2, action="append", required=True, metavar=("X", "Y"),
    )
    feature_segment.add_argument(
        "--negative-click", type=int, nargs=2, action="append", default=[],
        metavar=("X", "Y"), help="feature seed that must be excluded",
    )
    feature_segment.add_argument("--threshold", type=float, default=0.82)
    feature_segment.add_argument(
        "--negative-margin", type=float, default=0.03,
        help="required positive-vs-negative similarity margin",
    )
    feature_segment.add_argument("--feature-scale", type=float, default=0.5)
    feature_segment.add_argument("--output", required=True)
    feature_segment.add_argument("--preview")
    feature_segment.set_defaults(function=_feature_segment)

    sam_segment = subcommands.add_parser(
        "sam-segment", help="lift selected precomputed SAM masks into a point mask",
    )
    sam_segment.add_argument("--config", required=True)
    sam_segment.add_argument("--camera-index", type=int, default=0)
    sam_segment.add_argument("--mask-index", type=int, action="append", required=True)
    sam_segment.add_argument("--dilation", type=int, default=1)
    sam_segment.add_argument("--output", required=True)
    sam_segment.add_argument("--preview")
    sam_segment.set_defaults(function=_sam_segment)

    compare = subcommands.add_parser("compare", help="print CSV from evaluation summary files")
    compare.add_argument("summaries", nargs="+")
    compare.set_defaults(function=_compare)
    return parser


def main():
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
