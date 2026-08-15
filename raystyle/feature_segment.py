from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .backend import LegacyGaussianBackend
from .config import ExperimentConfig
from .io_utils import save_image


@torch.inference_mode()
def segment_from_feature_clicks(
    config: ExperimentConfig,
    clicks: list[tuple[int, int]],
    output_path: str,
    negative_clicks: list[tuple[int, int]] | None = None,
    camera_index: int = 0,
    threshold: float = 0.82,
    negative_margin: float = 0.03,
    scale: float = 0.5,
    feature_iteration: int = 10000,
    preview_path: str | None = None,
):
    """Create a SAGA feature mask from one or more calibrated-view clicks."""
    if not clicks:
        raise ValueError("at least one --click X Y is required")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    backend = LegacyGaussianBackend(
        config.legacy_root, config.model_path, config.source_path,
        config.images, config.resolution, config.white_background,
    )
    if not 0 <= camera_index < len(backend.train_cameras):
        raise IndexError(
            f"camera index {camera_index} outside [0, {len(backend.train_cameras) - 1}]"
        )

    from gaussian_renderer import render_contrastive_feature
    from scene import FeatureGaussianModel

    model_root = Path(config.model_path)
    feature_path = (
        model_root / "point_cloud" / f"iteration_{feature_iteration}" /
        "contrastive_feature_point_cloud.ply"
    )
    gate_path = (
        model_root / "point_cloud" / f"iteration_{feature_iteration}" / "scale_gate.pt"
    )
    if not feature_path.is_file() or not gate_path.is_file():
        raise FileNotFoundError(
            f"feature model missing: {feature_path} or {gate_path}"
        )
    feature_model = FeatureGaussianModel(32)
    feature_model.load_ply(str(feature_path))
    if feature_model.get_xyz.shape[0] != backend.point_count:
        raise ValueError(
            "feature and scene Gaussian counts differ; cannot create a point mask"
        )
    gate = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Sigmoid()).cuda()
    gate.load_state_dict(torch.load(gate_path, map_location="cuda", weights_only=False))
    gate.eval()

    camera = backend.train_cameras[camera_index]
    camera.feature_width = camera.image_width
    camera.feature_height = camera.image_height
    background = torch.zeros(32, device="cuda")
    rendered = render_contrastive_feature(
        camera, feature_model, backend.pipeline, background,
    )["render"].permute(1, 2, 0)
    rendered = F.normalize(rendered, dim=-1, eps=1e-6)
    height, width = rendered.shape[:2]
    def collect_seeds(points: list[tuple[int, int]], label: str):
        result = []
        for x, y in points:
            if not 0 <= x < width or not 0 <= y < height:
                raise ValueError(
                    f"{label} click {(x, y)} outside rendered image {width}x{height}"
                )
            result.append(rendered[y, x])
        return torch.stack(result, dim=1) if result else None

    seeds = collect_seeds(clicks, "positive")
    negative_seeds = collect_seeds(negative_clicks or [], "negative")

    gate_values = gate(torch.tensor([float(scale)], device="cuda"))
    point_features = feature_model.get_point_features.squeeze() * gate_values.unsqueeze(0)
    point_features = F.normalize(point_features, dim=-1, eps=1e-6)
    score = ((point_features @ seeds + 1) * 0.5).amax(dim=1)
    mask = score > float(threshold)
    if negative_seeds is not None:
        negative_score = ((point_features @ negative_seeds + 1) * 0.5).amax(dim=1)
        mask &= score > negative_score + float(negative_margin)
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mask.cpu(), target)

    if preview_path:
        soft_mask = backend.segment_mask(camera, torch.where(mask)[0])
        original = backend.render_original(camera)
        red = original.new_tensor([1.0, 0.03, 0.03]).view(3, 1, 1)
        preview = original * (1 - soft_mask) + red * soft_mask
        save_image(preview_path, preview)
    return {
        "output": str(target),
        "selected": int(mask.sum()),
        "total": backend.point_count,
        "fraction": float(mask.float().mean()),
        "camera_index": int(camera_index),
        "image_size": [int(width), int(height)],
        "threshold": float(threshold),
        "negative_margin": float(negative_margin),
        "negative_clicks": len(negative_clicks or []),
        "scale": float(scale),
        "preview": str(Path(preview_path).resolve()) if preview_path else None,
    }


@torch.inference_mode()
def segment_from_sam_masks(
    config: ExperimentConfig,
    mask_indices: list[int],
    output_path: str,
    camera_index: int = 0,
    dilation: int = 1,
    preview_path: str | None = None,
):
    """Lift selected precomputed SAM masks from one calibrated view into 3D."""
    if not mask_indices:
        raise ValueError("at least one --mask-index is required")
    backend = LegacyGaussianBackend(
        config.legacy_root, config.model_path, config.source_path,
        config.images, config.resolution, config.white_background,
    )
    if not 0 <= camera_index < len(backend.train_cameras):
        raise IndexError(
            f"camera index {camera_index} outside [0, {len(backend.train_cameras) - 1}]"
        )
    camera = backend.train_cameras[camera_index]
    image_name = Path(camera.image_name).stem
    sam_path = Path(config.source_path) / "sam_masks" / f"{image_name}.pt"
    if not sam_path.is_file():
        raise FileNotFoundError(f"precomputed SAM masks not found: {sam_path}")
    masks = torch.load(sam_path, map_location="cuda", weights_only=False).bool()
    invalid = [index for index in mask_indices if not 0 <= index < len(masks)]
    if invalid:
        raise IndexError(f"SAM mask indices outside [0, {len(masks) - 1}]: {invalid}")
    union = masks[mask_indices].any(dim=0).float()[None, None]
    union = F.interpolate(
        union, size=(camera.image_height, camera.image_width), mode="nearest",
    )
    if dilation > 0:
        kernel = 2 * int(dilation) + 1
        union = F.max_pool2d(union, kernel, stride=1, padding=int(dilation))
    union = union[0, 0].bool()

    from segmentation.utils import nearest_visible_points

    xyz = backend.gaussians.get_xyz
    height, width = union.shape
    ones = torch.ones((len(xyz), 1), device=xyz.device, dtype=xyz.dtype)
    # Gaussian Splatting stores transposed transforms and therefore applies
    # homogeneous points as row vectors (xyz_h @ full_proj_transform).
    clip = torch.cat((xyz, ones), dim=1) @ camera.full_proj_transform.to(xyz.device)
    raw_w = clip[:, 3]
    safe_w = raw_w.clamp_min(1e-6)
    ndc = clip[:, :3] / safe_w[:, None]
    u_float = (ndc[:, 0] + 1.0) * 0.5 * (width - 1)
    # The legacy camera basis already uses image-down y, so no OpenGL-style
    # vertical flip is needed here.
    v_float = (ndc[:, 1] + 1.0) * 0.5 * (height - 1)
    valid = (
        (raw_w > 1e-6) & (ndc[:, 2] > 0) &
        (u_float >= 0) & (u_float < width) &
        (v_float >= 0) & (v_float < height)
    )
    u = u_float.long().clamp(0, width - 1)
    v = v_float.long().clamp(0, height - 1)
    depth = raw_w
    valid_ids = torch.nonzero(valid, as_tuple=False).flatten()
    pixels = v[valid_ids] * width + u[valid_ids]
    front = nearest_visible_points(pixels, depth[valid_ids], width * height)
    visible_ids = valid_ids[front]
    selected_ids = visible_ids[union[v[visible_ids], u[visible_ids]]]
    mask = torch.zeros(backend.point_count, dtype=torch.bool, device=xyz.device)
    mask[selected_ids] = True

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mask.cpu(), target)
    if preview_path:
        soft_mask = backend.segment_mask(camera, selected_ids)
        original = backend.render_original(camera)
        red = original.new_tensor([1.0, 0.03, 0.03]).view(3, 1, 1)
        save_image(preview_path, original * (1 - soft_mask) + red * soft_mask)
    return {
        "output": str(target),
        "selected": int(mask.sum()),
        "total": backend.point_count,
        "fraction": float(mask.float().mean()),
        "camera_index": int(camera_index),
        "image_name": image_name,
        "mask_indices": [int(index) for index in mask_indices],
        "dilation": int(dilation),
        "preview": str(Path(preview_path).resolve()) if preview_path else None,
    }
