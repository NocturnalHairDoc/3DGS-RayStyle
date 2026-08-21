from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

from .config import ExperimentConfig
from .io_utils import save_image, write_json
from .losses import boundary_outside_preservation_loss, outside_preservation_loss
from .viewer import RayStyleViewerScene


@dataclass(frozen=True)
class CameraPose:
    rotation: np.ndarray
    translation: np.ndarray
    fov_x: float
    fov_y: float


def camera_center(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Return the world-space center for the legacy 3DGS R/T convention."""
    return -np.asarray(rotation, dtype=np.float64) @ np.asarray(
        translation, dtype=np.float64,
    )


def translation_from_center(rotation: np.ndarray, center: np.ndarray) -> np.ndarray:
    return -np.asarray(rotation, dtype=np.float64).T @ np.asarray(center, dtype=np.float64)


def _stable_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    pivot = int(np.argmax(np.abs(axis)))
    return -axis if axis[pivot] < 0 else axis


def ordered_camera_indices(poses: list[CameraPose], target: np.ndarray) -> list[int]:
    """Order calibrated cameras around a target on their dominant orbit plane."""
    if len(poses) < 2:
        return list(range(len(poses)))
    centers = np.stack([
        camera_center(pose.rotation, pose.translation) for pose in poses
    ])
    offsets = centers - np.asarray(target, dtype=np.float64)
    _, _, axes = np.linalg.svd(offsets, full_matrices=False)
    first = _stable_axis(axes[0])
    normal = _stable_axis(axes[-1])
    second = np.cross(normal, first)
    second /= max(np.linalg.norm(second), 1e-12)
    angles = np.arctan2(offsets @ second, offsets @ first)
    return np.argsort(angles, kind="stable").tolist()


def closed_camera_path(
    poses: list[CameraPose], target: np.ndarray, frame_count: int,
) -> list[CameraPose]:
    """Interpolate a closed path through calibrated camera anchors."""
    if len(poses) < 2:
        raise ValueError("orbit validation requires at least two calibrated cameras")
    if frame_count < len(poses):
        raise ValueError("frame_count must be at least the number of camera anchors")
    ordered = [poses[index] for index in ordered_camera_indices(poses, target)]
    result: list[CameraPose] = []
    for frame_index in range(frame_count):
        progress = frame_index * len(ordered) / frame_count
        first_index = int(math.floor(progress)) % len(ordered)
        alpha = progress - math.floor(progress)
        second_index = (first_index + 1) % len(ordered)
        first = ordered[first_index]
        second = ordered[second_index]
        rotations = Rotation.from_matrix(np.stack((first.rotation, second.rotation)))
        rotation = Slerp([0.0, 1.0], rotations)([alpha]).as_matrix()[0]
        first_center = camera_center(first.rotation, first.translation)
        second_center = camera_center(second.rotation, second.translation)
        center = (1 - alpha) * first_center + alpha * second_center
        result.append(CameraPose(
            rotation=rotation.astype(np.float32),
            translation=translation_from_center(rotation, center).astype(np.float32),
            fov_x=float((1 - alpha) * first.fov_x + alpha * second.fov_x),
            fov_y=float((1 - alpha) * first.fov_y + alpha * second.fov_y),
        ))
    return result


def orbit_gates(summary: dict) -> dict[str, bool]:
    """Representation-stage thresholds for albedo-only temporal validation."""
    numeric = [
        value for value in summary.values() if isinstance(value, (float, int))
    ]
    return {
        "finite_metrics": all(math.isfinite(float(value)) for value in numeric),
        "reliable_pair_coverage_ge_70pct": summary["evaluated_pair_fraction"] >= 0.70,
        "mean_corresponded_rgb_l1_le_12pct": summary["mean_corresponded_rgb_l1"] <= 0.12,
        "p95_corresponded_rgb_l1_le_20pct": summary["p95_corresponded_rgb_l1"] <= 0.20,
        "mean_temporal_excess_le_4pct": summary["mean_temporal_excess_l1"] <= 0.04,
        "p95_temporal_excess_le_8pct": summary["p95_temporal_excess_l1"] <= 0.08,
        "loop_closure_rgb_l1_le_15pct": summary["loop_closure_rgb_l1"] <= 0.15,
        "outside_leakage_le_2pct": summary["maximum_outside_leakage"] <= 0.02,
        "boundary_leakage_le_6pct": summary["maximum_boundary_leakage"] <= 0.06,
    }


def _corresponded_l1(first, second):
    first_ids, first_values = first
    second_ids, second_values = second
    if not len(first_ids) or not len(second_ids):
        return 0, float("nan")
    positions = torch.searchsorted(second_ids, first_ids)
    valid = positions < len(second_ids)
    valid &= second_ids[positions.clamp_max(len(second_ids) - 1)] == first_ids
    if not torch.any(valid):
        return 0, float("nan")
    distance = (first_values[valid] - second_values[positions[valid]]).abs().mean()
    return int(valid.sum()), float(distance)


def _contact_sheet(frame_paths: list[Path], output: Path):
    selected = [frame_paths[index] for index in np.linspace(
        0, len(frame_paths) - 1, min(12, len(frame_paths)), dtype=int,
    )]
    columns = 4
    rows = math.ceil(len(selected) / columns)
    cell_w, cell_h, label_h = 420, 280, 24
    canvas = Image.new("RGB", (columns * cell_w, rows * (cell_h + label_h)), "#181818")
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(selected):
        image = Image.open(path).convert("RGB")
        image.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        x = (index % columns) * cell_w
        y = (index // columns) * (cell_h + label_h)
        canvas.paste(image, (x + (cell_w - image.width) // 2, y))
        draw.text((x + 7, y + cell_h + 3), path.stem, fill="white")
    canvas.save(output, quality=94)


def _encode_video(frame_dir: Path, output: Path, fps: int) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            return False
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%04d.png"), "-c:v", "libx264",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt", "yuv420p", str(output),
    ], check=True)
    return True


@torch.inference_mode()
def validate_orbit(
    config: ExperimentConfig, checkpoint_path: str, output_dir: str | Path,
    *, frame_count: int = 48, fps: int = 12,
) -> dict:
    output = Path(output_dir).expanduser().resolve()
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    scene = RayStyleViewerScene(config, checkpoint_path)
    source_cameras = scene.backend.test_cameras or scene.backend.train_cameras
    poses = [CameraPose(
        np.asarray(camera.R), np.asarray(camera.T),
        float(camera.FoVx), float(camera.FoVy),
    ) for camera in source_cameras]
    target = scene.backend.xyz[scene.state.selected_ids].mean(0).detach().cpu().numpy()
    path = closed_camera_path(poses, target, frame_count)
    source = source_cameras[0]
    from scene.cameras import Camera

    # A small uniform subset often contains no frontmost splat in dense scenes.
    # Keep enough candidates to obtain stable cross-view correspondences while
    # still bounding projection memory for the large Bicycle road segment.
    sample_ids = scene.state.selected_ids
    if len(sample_ids) > 65536:
        sample_ids = sample_ids[torch.linspace(
            0, len(sample_ids) - 1, 65536, device=sample_ids.device,
        ).long()]
    samples = []
    rows = []
    frame_paths = []
    selected_albedo = scene.state.selected_albedo()
    patch_kernel = max(
        9, 2 * int(min(int(source.image_height), int(source.image_width)) * 0.03 / 2) + 1,
    )
    for frame_index, pose in enumerate(path):
        camera = Camera(
            colmap_id=frame_index, R=pose.rotation, T=pose.translation,
            FoVx=pose.fov_x, FoVy=pose.fov_y,
            image=torch.zeros(3, int(source.image_height), int(source.image_width)),
            gt_alpha_mask=None, image_name=f"orbit_{frame_index:04d}", uid=frame_index,
        )
        original = scene.backend.render_original(camera)
        mask = scene.backend.segment_mask(camera, scene.state.selected_ids)
        divisor = mask.permute(1, 2, 0)
        pure_albedo = scene.backend._sparse_map(
            camera, scene.state.selected_ids, selected_albedo, divisor,
        ).clamp(0, 1).permute(2, 0, 1)
        image = original * (1 - mask) + pure_albedo * mask
        visible_ids, grid = scene.backend.projected_visible_samples(
            camera, scene.state.selected_ids, sample_ids, mask,
        )
        # Compare the intrinsic albedo field, not boundary pixels composited
        # with the original scene. This isolates UV/texture stability from
        # expected view-dependent changes in Gaussian alpha coverage.
        patch_albedo = torch.nn.functional.avg_pool2d(
            pure_albedo.unsqueeze(0), patch_kernel, stride=1, padding=patch_kernel // 2,
        )
        patch_original = torch.nn.functional.avg_pool2d(
            original.unsqueeze(0), patch_kernel, stride=1, padding=patch_kernel // 2,
        )
        raw_values = scene.backend.sample_projected_features(pure_albedo.unsqueeze(0), grid)
        values = scene.backend.sample_projected_features(patch_albedo, grid)
        original_values = scene.backend.sample_projected_features(patch_original, grid)
        local_ids = torch.searchsorted(scene.state.selected_ids, visible_ids)
        expected_values = selected_albedo[local_ids]
        reliable = (raw_values - expected_values).abs().mean(1) <= 0.12
        projected_count = int(len(visible_ids))
        visible_ids = visible_ids[reliable]
        values = values[reliable]
        original_values = original_values[reliable]
        samples.append((visible_ids, values, original_values))
        frame_path = frames_dir / f"frame_{frame_index:04d}.png"
        save_image(frame_path, image)
        frame_paths.append(frame_path)
        rows.append({
            "frame": frame_index,
            "visible_samples": int(len(visible_ids)),
            "projected_visible_samples": projected_count,
            "reliable_sample_fraction": float(reliable.float().mean()) if len(reliable) else 0.0,
            "outside_leakage": float(outside_preservation_loss(image, original, mask)),
            "boundary_leakage": float(
                boundary_outside_preservation_loss(image, original, mask)
            ),
        })

    pair_metrics = []
    for index in range(frame_count):
        first = samples[index]
        second = samples[(index + 1) % frame_count]
        count, distance = _corresponded_l1(
            (first[0], first[1]), (second[0], second[1]),
        )
        _, original_distance = _corresponded_l1(
            (first[0], first[2]), (second[0], second[2]),
        )
        pair_metrics.append({
            "first_frame": index,
            "second_frame": (index + 1) % frame_count,
            "correspondence_count": count,
            "corresponded_rgb_l1": distance,
            "original_corresponded_rgb_l1": original_distance,
            "temporal_excess_l1": max(distance - original_distance, 0.0),
        })
    reliable_pairs = [
        row for row in pair_metrics if row["correspondence_count"] >= 64
    ]
    distances = np.asarray([
        row["corresponded_rgb_l1"] for row in reliable_pairs
    ], dtype=np.float64)
    original_distances = np.asarray([
        row["original_corresponded_rgb_l1"] for row in reliable_pairs
    ], dtype=np.float64)
    excess_distances = np.asarray([
        row["temporal_excess_l1"] for row in reliable_pairs
    ], dtype=np.float64)
    counts = [row["correspondence_count"] for row in pair_metrics]
    summary = {
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "frame_count": frame_count,
        "fps": fps,
        "camera_anchor_count": len(poses),
        "patch_kernel": patch_kernel,
        "minimum_correspondence_count": min(counts),
        "evaluated_pair_count": len(reliable_pairs),
        "evaluated_pair_fraction": len(reliable_pairs) / len(pair_metrics),
        "mean_corresponded_rgb_l1": float(np.nanmean(distances)),
        "p95_corresponded_rgb_l1": float(np.nanpercentile(distances, 95)),
        "mean_original_corresponded_rgb_l1": float(np.nanmean(original_distances)),
        "mean_temporal_excess_l1": float(np.nanmean(excess_distances)),
        "p95_temporal_excess_l1": float(np.nanpercentile(excess_distances, 95)),
        "loop_closure_rgb_l1": pair_metrics[-1]["corresponded_rgb_l1"],
        "maximum_outside_leakage": max(row["outside_leakage"] for row in rows),
        "maximum_boundary_leakage": max(row["boundary_leakage"] for row in rows),
    }
    summary["gates"] = orbit_gates(summary)
    summary["automatic_gates_pass"] = all(summary["gates"].values())
    write_json(output / "per_frame.json", rows)
    write_json(output / "adjacent_pairs.json", pair_metrics)
    _contact_sheet(frame_paths, output / "orbit_contact_sheet.jpg")
    summary["video_created"] = _encode_video(frames_dir, output / "orbit.mp4", fps)
    write_json(output / "orbit_summary.json", summary)
    return summary
