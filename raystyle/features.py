from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class FrozenDINO(nn.Module):
    """Legacy DINO encoder with frozen weights but gradients retained for the input image."""

    def __init__(self, legacy_root: str, checkpoint: str, image_size: int = 224):
        super().__init__()
        legacy = str(Path(legacy_root).resolve())
        if legacy not in sys.path:
            sys.path.insert(0, legacy)
        from utils.dino_utils import DINO

        wrapper = DINO(patch_size=8, device="cuda")
        wrapper.load_checkpoint(checkpoint)
        self.model = wrapper.model
        self.image_size = int(image_size)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        images = F.interpolate(images, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        images = (images.clamp(0, 1) - self.mean) / self.std
        tokens = self.model.get_intermediate_layers(images, n=1)[0][:, 1:]
        side = int(round(tokens.shape[1] ** 0.5))
        features = tokens.reshape(images.shape[0], side, side, -1).permute(0, 3, 1, 2)
        return F.normalize(features, dim=1, eps=1e-6)


def masked_feature_stats(features: torch.Tensor, mask: torch.Tensor | None = None):
    if mask is None:
        weights = torch.ones(
            features.shape[0], 1, features.shape[2], features.shape[3],
            device=features.device, dtype=features.dtype,
        )
    else:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        weights = F.interpolate(mask.float(), features.shape[-2:], mode="bilinear", align_corners=False)
    weights = weights.clamp_min(0)
    denom = weights.sum((2, 3)).clamp_min(1e-4)
    mean = (features * weights).sum((2, 3)) / denom
    variance = ((features - mean[:, :, None, None]).square() * weights).sum((2, 3)) / denom
    return mean, torch.sqrt(variance + 1e-6)


def dino_content_loss(stylized: torch.Tensor, content: torch.Tensor, mask: torch.Tensor | None = None):
    similarity = (stylized * content.detach()).sum(1, keepdim=True)
    error = 1 - similarity
    if mask is None:
        return error.mean()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    weights = F.interpolate(mask.float(), error.shape[-2:], mode="bilinear", align_corners=False)
    return (error * weights).sum() / weights.sum().clamp_min(1e-4)

