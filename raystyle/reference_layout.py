from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ReferenceLayout:
    canvas: torch.Tensor
    saliency: torch.Tensor
    selection_preview: torch.Tensor
    regions: tuple[tuple[float, float, float, float], ...]
    scales: tuple[float, ...]
    tile_grid: tuple[int, int]


def metric_tile_grid(texture_field, selected_xyz: torch.Tensor, base_count=3):
    """Choose tile counts that make motifs approximately isotropic in 3D."""
    base_count = int(base_count)
    if texture_field is None or not hasattr(texture_field, "component_ids"):
        return base_count, base_count
    components = texture_field.component_ids
    component = int(torch.argmax(torch.bincount(components)).item())
    ids = torch.where(components == component)[0]
    if len(ids) < 3:
        return base_count, base_count
    chart = int(texture_field.chart_ids[ids[0]].item())
    axes = texture_field.chart_axes[chart]
    charts = torch.unique(texture_field.chart_ids[ids])
    alignment = torch.abs(
        torch.einsum("cdi,di->ci", texture_field.chart_axes[charts], axes)
    ).mean()
    if float(alignment) < 0.85:
        return base_count, base_count
    points = selected_xyz.detach().float()[ids]
    projected = (points - points.mean(0)) @ axes
    extent = (
        torch.quantile(projected, 0.995, dim=0)
        - torch.quantile(projected, 0.005, dim=0)
    ).clamp_min(1e-6)
    relative = extent / extent.min()
    columns = max(1, min(16, int(round(base_count * float(relative[0])))))
    rows = max(1, min(16, int(round(base_count * float(relative[1])))))
    return columns, rows


def _unit_quantile(values: torch.Tensor, quantile=0.95):
    scale = torch.quantile(values.flatten(), float(quantile)).clamp_min(1e-6)
    return (values / scale).clamp(0, 1)


def reference_saliency(reference_chw: torch.Tensor, analysis_size=192):
    """Deterministic image-only structure score used for reference crop selection."""
    if reference_chw.ndim != 3 or reference_chw.shape[0] != 3:
        raise ValueError("reference image must have shape (3, H, W)")
    image = F.interpolate(
        reference_chw.unsqueeze(0).float(),
        (int(analysis_size), int(analysis_size)),
        mode="bilinear", align_corners=False,
    ).clamp(0, 1)
    luminance = (
        0.2126 * image[:, 0:1]
        + 0.7152 * image[:, 1:2]
        + 0.0722 * image[:, 2:3]
    )
    dx = F.pad(luminance[..., 1:] - luminance[..., :-1], (0, 1, 0, 0))
    dy = F.pad(luminance[..., 1:, :] - luminance[..., :-1, :], (0, 0, 0, 1))
    gradient = torch.sqrt(dx.square() + dy.square() + 1e-12)
    local_mean = F.avg_pool2d(luminance, 11, stride=1, padding=5)
    contrast = (luminance - local_mean).abs()
    saturation = image.amax(1, keepdim=True) - image.amin(1, keepdim=True)
    score = (
        0.45 * _unit_quantile(gradient)
        + 0.35 * _unit_quantile(contrast)
        + 0.20 * _unit_quantile(saturation)
    )
    return F.avg_pool2d(score, 7, stride=1, padding=3)[0]


def _best_window(saliency: torch.Tensor, scale: float):
    size = saliency.shape[-1]
    window = max(8, min(size, int(round(size * float(scale)))))
    stride = max(1, size // 48)
    pooled = F.avg_pool2d(
        saliency.unsqueeze(0), window, stride=stride,
    )[0, 0]
    # Flattened argmax is deterministic and resolves ties in raster order.
    index = int(torch.argmax(pooled).item())
    columns = pooled.shape[1]
    y0 = (index // columns) * stride
    x0 = (index % columns) * stride
    y0 = min(y0, size - window)
    x0 = min(x0, size - window)
    return (
        x0 / size, y0 / size, (x0 + window) / size, (y0 + window) / size,
    )


def _region_iou(first, second):
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-8)


def _motif_windows(saliency: torch.Tensor, count: int, focus_scale: float):
    """Select distinct, centred high-detail crops without a learned detector."""
    size = saliency.shape[-1]
    base = float(focus_scale)
    scales = tuple(min(0.72, base * factor) for factor in (1.0, 1.25, 1.55, 1.9))
    candidates = []
    for scale_index, scale in enumerate(scales):
        window = max(8, min(size, int(round(size * scale))))
        stride = max(1, size // 64)
        field = saliency.unsqueeze(0)
        mean = F.avg_pool2d(field, window, stride=stride)[0, 0]
        inner = max(4, int(round(window * 0.58)))
        inner_mean = F.avg_pool2d(field, inner, stride=stride)[0, 0]
        inset = max(0, (inner_mean.shape[0] - mean.shape[0]) // 2)
        inner_mean = inner_mean[
            inset:inset + mean.shape[0], inset:inset + mean.shape[1]
        ]
        # Prefer a structured centre over a crop whose strongest edge is cut by
        # the window boundary. The small scale term prevents tiny detail crops
        # from occupying every slot.
        score = 0.62 * mean + 0.38 * inner_mean + 0.035 * scale
        limit = min(score.numel(), max(256, count * 128))
        order = torch.argsort(score.flatten(), descending=True, stable=True)[:limit]
        columns = score.shape[1]
        for index in order.tolist():
            y0 = min((index // columns) * stride, size - window)
            x0 = min((index % columns) * stride, size - window)
            region = (
                x0 / size, y0 / size,
                (x0 + window) / size, (y0 + window) / size,
            )
            candidates.append((float(score.flatten()[index]), scale_index, region))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2][1], item[2][0]))
    selected = []
    for _, _, region in candidates:
        if all(_region_iou(region, previous) <= 0.30 for previous in selected):
            selected.append(region)
            if len(selected) == count:
                break
    if len(selected) < count:
        for _, _, region in candidates:
            if region not in selected:
                selected.append(region)
                if len(selected) == count:
                    break
    return tuple(selected)


def _crop_normalized(image: torch.Tensor, region):
    _, height, width = image.shape
    x0 = max(0, min(width - 1, int(math.floor(region[0] * width))))
    y0 = max(0, min(height - 1, int(math.floor(region[1] * height))))
    x1 = max(x0 + 1, min(width, int(math.ceil(region[2] * width))))
    y1 = max(y0 + 1, min(height, int(math.ceil(region[3] * height))))
    return image[:, y0:y1, x0:x1]


def _match_channel_statistics(canvas: torch.Tensor, reference: torch.Tensor):
    source_mean = canvas.mean((1, 2), keepdim=True)
    source_std = canvas.std((1, 2), keepdim=True).clamp_min(1e-4)
    target_mean = reference.mean((1, 2), keepdim=True)
    target_std = reference.std((1, 2), keepdim=True).clamp_min(1e-4)
    # A full covariance transfer can suppress the very salient colours this
    # layout is meant to expose. Correct the mean, but limit contrast changes.
    scale = (target_std / source_std).clamp(0.9, 1.1)
    result = ((canvas - source_mean) * scale + target_mean).clamp(0, 1)
    # Clipping can shift the mean for strongly saturated references. A few
    # deterministic projections restore it without flattening local contrast.
    for _ in range(4):
        correction = target_mean - result.mean((1, 2), keepdim=True)
        result = (result + correction).clamp(0, 1)
    return result


def _selection_preview(reference: torch.Tensor, regions):
    preview = reference.detach().clone()
    _, height, width = preview.shape
    colours = preview.new_tensor(
        ((1.0, 0.15, 0.1), (0.1, 1.0, 0.2), (0.15, 0.45, 1.0), (1.0, 0.85, 0.1)),
    )
    thickness = max(1, round(min(height, width) / 180))
    for index, region in enumerate(regions):
        x0 = max(0, min(width - 1, int(round(region[0] * (width - 1)))))
        y0 = max(0, min(height - 1, int(round(region[1] * (height - 1)))))
        x1 = max(x0 + 1, min(width, int(round(region[2] * width))))
        y1 = max(y0 + 1, min(height, int(round(region[3] * height))))
        colour = colours[index % len(colours)][:, None]
        preview[:, y0:min(y0 + thickness, height), x0:x1] = colour[:, :, None]
        preview[:, max(y1 - thickness, 0):y1, x0:x1] = colour[:, :, None]
        preview[:, y0:y1, x0:min(x0 + thickness, width)] = colour[:, None, :]
        preview[:, y0:y1, max(x1 - thickness, 0):x1] = colour[:, None, :]
    return preview


def build_reference_layout(
    reference_chw: torch.Tensor,
    mode="full",
    patch_count=4,
    canvas_size=512,
    focus_scale=0.32,
    tile_count=3,
    tile_grid: tuple[int, int] | None = None,
):
    """Build the texture/style source while preserving original colour statistics.

    ``saliency_grid`` repeats the strongest structures at several physical scales.
    It intentionally uses no semantic or generative model, so construction stays
    deterministic and cheap enough to run as part of every experiment.
    """
    reference = reference_chw.detach().float().clamp(0, 1)
    saliency = reference_saliency(reference)
    if mode == "full":
        return ReferenceLayout(
            canvas=reference,
            saliency=saliency,
            selection_preview=reference.clone(),
            regions=((0.0, 0.0, 1.0, 1.0),),
            scales=(1.0,),
            tile_grid=(1, 1),
        )
    if mode in {"saliency_focus", "saliency_tile"}:
        if not 0.15 <= float(focus_scale) <= 1.0:
            raise ValueError("reference focus scale must be in [0.15, 1.0]")
        region = _best_window(saliency, float(focus_scale))
        crop = _crop_normalized(reference, region)
        size = int(canvas_size)
        if mode == "saliency_focus":
            canvas = F.interpolate(
                crop.unsqueeze(0), (size, size),
                mode="bilinear", align_corners=False,
            )[0]
        else:
            tile_count = int(tile_count)
            if not 1 <= tile_count <= 8:
                raise ValueError("reference tile count must be in [1, 8]")
            columns, rows = tile_grid or (tile_count, tile_count)
            columns, rows = int(columns), int(rows)
            if not (1 <= columns <= 16 and 1 <= rows <= 16):
                raise ValueError("reference tile grid dimensions must be in [1, 16]")
            canvas = reference.new_zeros(3, size, size)
            for row in range(rows):
                for column in range(columns):
                    x0 = round(column * size / columns)
                    x1 = round((column + 1) * size / columns)
                    y0 = round(row * size / rows)
                    y1 = round((row + 1) * size / rows)
                    canvas[:, y0:y1, x0:x1] = F.interpolate(
                        crop.unsqueeze(0), (y1 - y0, x1 - x0),
                        mode="bilinear", align_corners=False,
                    )[0]
        canvas = _match_channel_statistics(canvas, reference)
        return ReferenceLayout(
            canvas=canvas,
            saliency=saliency,
            selection_preview=_selection_preview(reference, (region,)),
            regions=(region,),
            scales=(float(focus_scale),),
            tile_grid=(1, 1) if mode == "saliency_focus" else (columns, rows),
        )
    if mode == "saliency_motifs":
        patch_count = int(patch_count)
        if not 1 <= patch_count <= 16:
            raise ValueError("reference saliency patch count must be in [1, 16]")
        if not 0.15 <= float(focus_scale) <= 0.5:
            raise ValueError("motif focus scale must be in [0.15, 0.5]")
        regions = _motif_windows(saliency, patch_count, focus_scale)
        columns, rows = tile_grid or (
            int(math.ceil(math.sqrt(patch_count))),
            int(math.ceil(patch_count / math.ceil(math.sqrt(patch_count)))),
        )
        columns, rows = int(columns), int(rows)
        if not (1 <= columns <= 16 and 1 <= rows <= 16):
            raise ValueError("reference motif grid dimensions must be in [1, 16]")
        size = int(canvas_size)
        canvas = reference.new_zeros(3, size, size)
        for index in range(columns * rows):
            row, column = divmod(index, columns)
            x0 = round(column * size / columns)
            x1 = round((column + 1) * size / columns)
            y0 = round(row * size / rows)
            y1 = round((row + 1) * size / rows)
            crop = _crop_normalized(reference, regions[index % len(regions)])
            canvas[:, y0:y1, x0:x1] = F.interpolate(
                crop.unsqueeze(0), (y1 - y0, x1 - x0),
                mode="bilinear", align_corners=False,
            )[0]
        canvas = _match_channel_statistics(canvas, reference)
        return ReferenceLayout(
            canvas=canvas,
            saliency=saliency,
            selection_preview=_selection_preview(reference, regions),
            regions=regions,
            scales=tuple(region[2] - region[0] for region in regions),
            tile_grid=(columns, rows),
        )
    if mode != "saliency_grid":
        raise ValueError(
            "reference layout must be 'full', 'saliency_grid', 'saliency_focus', "
            "'saliency_tile', or 'saliency_motifs'"
        )
    patch_count = int(patch_count)
    if not 1 <= patch_count <= 16:
        raise ValueError("reference saliency patch count must be in [1, 16]")

    hierarchy = (0.70, 0.48, 0.32, 0.24)
    scales = tuple(hierarchy[index % len(hierarchy)] for index in range(patch_count))
    regions = tuple(_best_window(saliency, scale) for scale in scales)
    columns = int(math.ceil(math.sqrt(patch_count)))
    rows = int(math.ceil(patch_count / columns))
    size = int(canvas_size)
    canvas = reference.new_zeros(3, size, size)
    fallback = F.interpolate(
        reference.unsqueeze(0), (size, size), mode="bilinear", align_corners=False,
    )[0]
    canvas.copy_(fallback)
    for index, region in enumerate(regions):
        row, column = divmod(index, columns)
        x0, x1 = round(column * size / columns), round((column + 1) * size / columns)
        y0, y1 = round(row * size / rows), round((row + 1) * size / rows)
        crop = _crop_normalized(reference, region)
        canvas[:, y0:y1, x0:x1] = F.interpolate(
            crop.unsqueeze(0), (y1 - y0, x1 - x0),
            mode="bilinear", align_corners=False,
        )[0]
    canvas = _match_channel_statistics(canvas, reference)
    return ReferenceLayout(
        canvas=canvas,
        saliency=saliency,
        selection_preview=_selection_preview(reference, regions),
        regions=regions,
        scales=scales,
        tile_grid=(columns, rows),
    )
