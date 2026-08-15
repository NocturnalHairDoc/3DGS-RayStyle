from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .features import masked_feature_stats


def srgb_to_lab(image: torch.Tensor):
    """Differentiable D65 CIE Lab, normalized to roughly unit channels."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    srgb = image.clamp(0, 1)
    linear = torch.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055).pow(2.4),
    )
    r, g, b = linear.unbind(1)
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883
    delta = 6.0 / 29.0

    def transform(value):
        return torch.where(
            value > delta ** 3,
            value.clamp_min(1e-8).pow(1 / 3),
            value / (3 * delta ** 2) + 4 / 29,
        )

    fx, fy, fz = transform(x), transform(y), transform(z)
    return torch.stack(
        ((116 * fy - 16) / 100, 500 * (fx - fy) / 128, 200 * (fy - fz) / 128),
        dim=1,
    )


def _limit_rows(values: torch.Tensor, maximum: int):
    if len(values) <= maximum:
        return values
    indices = torch.linspace(0, len(values) - 1, maximum, device=values.device).long()
    return values[indices]


def _patch_style_channels(image: torch.Tensor):
    """Exposure-robust chromaticity and log-luminance gradients."""
    rgb = image.clamp_min(0)
    chromaticity = rgb / rgb.sum(1, keepdim=True).clamp_min(0.03)
    luminance = (
        0.2126 * rgb[:, 0:1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 2:3]
    )
    log_luminance = torch.log(luminance.clamp_min(1e-4))
    dx = F.pad(log_luminance[..., 1:] - log_luminance[..., :-1], (0, 1, 0, 0))
    dy = F.pad(log_luminance[..., 1:, :] - log_luminance[..., :-1, :], (0, 0, 0, 1))
    return torch.cat((chromaticity, dx, dy), dim=1)


def _rgb_patch_bank(image: torch.Tensor, size: int, kernel=5, stride=3, maximum=1536):
    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = F.interpolate(image, (size, size), mode="bilinear", align_corners=False)
    channels = _patch_style_channels(image)
    patches = F.unfold(channels, kernel_size=kernel, stride=stride)
    patches = patches[0].T.reshape(-1, channels.shape[1], kernel * kernel)
    # Centered patches force local stroke/edge matching rather than allowing a
    # flat blue patch to win through global colour similarity.
    patches = patches - patches.mean(2, keepdim=True)
    patches = F.normalize(patches.flatten(1), dim=1, eps=1e-6)
    return _limit_rows(patches, maximum)


def _masked_rgb_patches(
    image, mask, size, kernel=5, stride=3, maximum=512, support_threshold=0.6,
):
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    image = F.interpolate(image, (size, size), mode="bilinear", align_corners=False)
    mask = F.interpolate(mask.float(), (size, size), mode="bilinear", align_corners=False)
    channels = _patch_style_channels(image)
    patches = F.unfold(channels, kernel_size=kernel, stride=stride)[0].T
    support = F.unfold(mask, kernel_size=kernel, stride=stride)[0].mean(0)
    patches = patches[support > float(support_threshold)].reshape(
        -1, channels.shape[1], kernel * kernel,
    )
    if not len(patches):
        return patches.flatten(1)
    patches = patches - patches.mean(2, keepdim=True)
    patches = F.normalize(patches.flatten(1), dim=1, eps=1e-6)
    return _limit_rows(patches, maximum)


def _nearest_patch_loss(query: torch.Tensor, reference: torch.Tensor):
    if not len(query):
        return reference.sum() * 0
    similarity = query @ reference.T
    return (1 - similarity.max(dim=1).values).mean()


def _segment_roi(image: torch.Tensor, mask: torch.Tensor, padding=0.12):
    """Crop an image/mask pair around visible segment support.

    Bounds are selected from the non-differentiable visibility mask; gradients
    still flow normally through the cropped image.
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    height, width = image.shape[-2:]
    support = mask[0, 0] > 0.15
    positions = torch.nonzero(support, as_tuple=False)
    if not len(positions):
        return image, mask, (0, height, 0, width)
    y0, x0 = positions.amin(0).tolist()
    y1, x1 = (positions.amax(0) + 1).tolist()
    pad_y = max(2, int(round((y1 - y0) * float(padding))))
    pad_x = max(2, int(round((x1 - x0) * float(padding))))
    y0, y1 = max(0, y0 - pad_y), min(height, y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(width, x1 + pad_x)
    return image[..., y0:y1, x0:x1], mask[..., y0:y1, x0:x1], (y0, y1, x0, x1)


def _crop_scaled(values: torch.Tensor, bounds, source_shape):
    y0, y1, x0, x1 = bounds
    source_h, source_w = source_shape
    height, width = values.shape[-2:]
    fy0 = max(0, min(height - 1, int(y0 * height / source_h)))
    fy1 = max(fy0 + 1, min(height, math.ceil(y1 * height / source_h)))
    fx0 = max(0, min(width - 1, int(x0 * width / source_w)))
    fx1 = max(fx0 + 1, min(width, math.ceil(x1 * width / source_w)))
    return values[..., fy0:fy1, fx0:fx1]


def rgb_style_stats(image: torch.Tensor, mask: torch.Tensor | None = None):
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if mask is None:
        weights = torch.ones_like(image[:, :1])
    else:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        weights = F.interpolate(mask.float(), image.shape[-2:], mode="bilinear", align_corners=False)
    denom = weights.sum((2, 3)).clamp_min(1e-4)
    mean = (image * weights).sum((2, 3)) / denom
    centered = image - mean[:, :, None, None]
    flat = centered.flatten(2)
    flat_w = weights.flatten(2)
    covariance = (flat * flat_w) @ flat.transpose(1, 2) / denom.unsqueeze(-1)
    gray = image.mean(1, keepdim=True)
    dx = gray[..., :, 1:] - gray[..., :, :-1]
    dy = gray[..., 1:, :] - gray[..., :-1, :]
    gradient = torch.stack((dx.abs().mean((2, 3)), dy.abs().mean((2, 3))), dim=-1)
    return mean, covariance, gradient


class ReferenceStyleLoss:
    def __init__(self, dino_features: torch.Tensor, reference_rgb: torch.Tensor):
        with torch.no_grad():
            self.feature_targets = tuple(v.detach() for v in masked_feature_stats(dino_features))
            self.rgb_targets = tuple(v.detach() for v in rgb_style_stats(reference_rgb))
            reference_batch = reference_rgb.unsqueeze(0) if reference_rgb.ndim == 3 else reference_rgb
            self.reference_rgb_mean = reference_batch.mean((2, 3)).detach()
            self.reference_lab_mean = srgb_to_lab(reference_batch).mean((2, 3)).detach()
            self.dino_patch_bank = F.normalize(
                dino_features.detach().flatten(2).transpose(1, 2)[0], dim=1, eps=1e-6,
            )
            self.rgb_patch_banks = {
                size: _rgb_patch_bank(reference_rgb.detach(), size).detach()
                for size in (96, 160, 224)
            }

    def color_mean_loss(self, albedo_texture: torch.Tensor):
        """Match intrinsic UV albedo colour, independently of HDR lighting."""
        texture = (
            albedo_texture.unsqueeze(0)
            if albedo_texture.ndim == 3 else albedo_texture
        )
        rgb_mean = texture.mean((2, 3))
        lab_mean = srgb_to_lab(texture).mean((2, 3))
        return F.l1_loss(rgb_mean, self.reference_rgb_mean) + F.l1_loss(
            lab_mean, self.reference_lab_mean,
        )

    def rendered_color_loss(
        self, image: torch.Tensor, mask: torch.Tensor, exposure_stops: float = 0.0,
    ):
        """Constrain post-PBR colour after compensating sampled exposure."""
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = (image * (2.0 ** -float(exposure_stops))).clamp(0, 1)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        weights = F.interpolate(
            mask.float(), image.shape[-2:], mode="bilinear", align_corners=False,
        )
        denom = weights.sum((2, 3)).clamp_min(1e-4)
        rgb_mean = (image * weights).sum((2, 3)) / denom
        lab_mean = (srgb_to_lab(image) * weights).sum((2, 3)) / denom
        return F.l1_loss(rgb_mean, self.reference_rgb_mean) + F.l1_loss(
            lab_mean, self.reference_lab_mean,
        )

    def global_loss(self, image: torch.Tensor, features: torch.Tensor, mask: torch.Tensor):
        feat = masked_feature_stats(features, mask)
        rgb = rgb_style_stats(image, mask)
        feature_loss = F.l1_loss(feat[0], self.feature_targets[0]) + F.l1_loss(feat[1], self.feature_targets[1])
        rgb_loss = F.l1_loss(rgb[0], self.rgb_targets[0]) + F.l1_loss(rgb[1], self.rgb_targets[1])
        gradient_loss = F.l1_loss(rgb[2], self.rgb_targets[2])
        return feature_loss + 0.5 * rgb_loss + 0.25 * gradient_loss

    def patch_loss(self, image: torch.Tensor, features: torch.Tensor, mask: torch.Tensor):
        source_shape = image.shape[-2:]
        roi_image, roi_mask, bounds = _segment_roi(image, mask)
        roi_features = _crop_scaled(features, bounds, source_shape)
        feature_mask = roi_mask
        feature_mask = F.interpolate(
            feature_mask.float(), roi_features.shape[-2:], mode="bilinear", align_corners=False,
        )[0, 0]
        dino_queries = roi_features.flatten(2).transpose(1, 2)[0][feature_mask.flatten() > 0.5]
        dino_queries = _limit_rows(F.normalize(dino_queries, dim=1, eps=1e-6), 768)
        dino_loss = _nearest_patch_loss(dino_queries, self.dino_patch_bank)
        rgb_losses = []
        for size, reference in self.rgb_patch_banks.items():
            queries = _masked_rgb_patches(roi_image, roi_mask, size)
            rgb_losses.append(_nearest_patch_loss(queries, reference))
        return 0.5 * dino_loss + torch.stack(rgb_losses).mean()

    def __call__(self, image: torch.Tensor, features: torch.Tensor, mask: torch.Tensor):
        return self.global_loss(image, features, mask) + self.patch_loss(image, features, mask)


def outside_preservation_loss(stylized: torch.Tensor, original: torch.Tensor, mask: torch.Tensor):
    outside = (1 - mask).clamp(0, 1)
    return ((stylized - original).abs() * outside).sum() / (outside.sum() * 3).clamp_min(1e-4)


def _local_structure(image: torch.Tensor):
    if image.ndim == 3:
        image = image.unsqueeze(0)
    luminance = (
        0.2126 * image[:, 0:1] + 0.7152 * image[:, 1:2] + 0.0722 * image[:, 2:3]
    ).clamp_min(0)
    log_luminance = torch.log(luminance.clamp_min(1e-4))
    mean = F.avg_pool2d(log_luminance, 9, stride=1, padding=4)
    centered = log_luminance - mean
    variance = F.avg_pool2d(centered.square(), 9, stride=1, padding=4)
    normalized = centered / torch.sqrt(variance + 1e-3)
    dx = F.pad(normalized[..., 1:] - normalized[..., :-1], (0, 1, 0, 0))
    dy = F.pad(normalized[..., 1:, :] - normalized[..., :-1, :], (0, 0, 0, 1))
    return torch.cat((normalized, dx, dy), dim=1)


def illumination_consistency_loss(
    first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor,
):
    """Compare chroma/structure after explicit segment-luminance normalization."""
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    weights_rgb = F.interpolate(
        mask.float(), first.shape[-2:], mode="bilinear", align_corners=False,
    )

    def normalize_luminance(image):
        luminance = (
            0.2126 * image[:, 0:1] + 0.7152 * image[:, 1:2] + 0.0722 * image[:, 2:3]
        ).clamp_min(1e-4)
        mean = (luminance * weights_rgb).sum((2, 3), keepdim=True) / (
            weights_rgb.sum((2, 3), keepdim=True).clamp_min(1e-4)
        )
        return image / mean.clamp_min(1e-4)

    first_normalized = normalize_luminance(first)
    second_normalized = normalize_luminance(second)
    first_structure = _local_structure(first_normalized)
    second_structure = _local_structure(second_normalized)
    weights = F.interpolate(
        mask.float(), first_structure.shape[-2:], mode="bilinear", align_corners=False,
    )
    structure = (first_structure - second_structure).abs() * weights
    structure_loss = structure.sum() / (
        weights.sum() * first_structure.shape[1]
    ).clamp_min(1e-4)
    first_chroma = first_normalized.clamp_min(0) / first_normalized.clamp_min(0).sum(
        1, keepdim=True,
    ).clamp_min(0.03)
    second_chroma = second_normalized.clamp_min(0) / second_normalized.clamp_min(0).sum(
        1, keepdim=True,
    ).clamp_min(0.03)
    chroma_loss = ((first_chroma - second_chroma).abs() * weights_rgb).sum() / (
        weights_rgb.sum() * 3
    ).clamp_min(1e-4)
    return structure_loss + 0.25 * chroma_loss
