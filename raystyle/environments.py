from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F


def _normalise(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values, dim=-1, eps=1e-7)


def _luminance(values: torch.Tensor) -> torch.Tensor:
    weights = values.new_tensor([0.2126, 0.7152, 0.0722])
    return (values * weights).sum(-1, keepdim=True)


def procedural_environment(index: int = 0, height: int = 128, width: int = 256):
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij")
    palettes = (
        torch.tensor([0.20, 0.32, 0.58]),
        torch.tensor([0.65, 0.30, 0.16]),
        torch.tensor([0.18, 0.48, 0.30]),
        torch.tensor([0.48, 0.22, 0.55]),
        torch.tensor([0.52, 0.52, 0.48]),
        torch.tensor([0.16, 0.35, 0.62]),
    )
    base = palettes[index % len(palettes)]
    horizon = torch.exp(-((yy - 0.52) / 0.24).square())
    pixels = base.view(1, 1, 3) * (0.35 + 0.75 * horizon[..., None])
    sun_x = 0.13 + 0.14 * (index % 6)
    sun_y = 0.18 + 0.05 * (index % 3)
    sun = torch.exp(-(((xx - sun_x) / 0.028).square() + ((yy - sun_y) / 0.04).square()))
    pixels = pixels + sun[..., None] * torch.tensor([12.0, 10.0, 7.0])
    return pixels.float()


def load_environment(path: Path) -> torch.Tensor:
    values = np.asarray(iio.imread(path), dtype=np.float32)
    if values.ndim == 2:
        values = np.repeat(values[..., None], 3, axis=-1)
    values = values[..., :3]
    if path.suffix.lower() not in {".hdr", ".exr"} and values.max(initial=0) > 1.5:
        values /= 255.0 if values.max(initial=0) <= 255 else 65535.0
    return torch.from_numpy(values).clamp_min(0)


@dataclass
class EnvironmentMap:
    name: str
    pixels: torch.Tensor
    yaw: float = 0.0
    exposure: float = 0.0

    def to(self, device) -> "EnvironmentMap":
        return EnvironmentMap(self.name, self.pixels.to(device), self.yaw, self.exposure)

    def sample(self, directions: torch.Tensor, roughness: torch.Tensor | None = None):
        directions = _normalise(directions)
        x, y, z = directions.unbind(-1)
        u = torch.remainder(torch.atan2(x, z) / (2 * torch.pi) + 0.5 + self.yaw, 1.0)
        v = torch.acos(y.clamp(-1, 1)) / torch.pi
        grid = torch.stack((u * 2 - 1, v * 2 - 1), dim=-1)
        shape = grid.shape[:-1]
        pixels = self.pixels.to(device=directions.device, dtype=directions.dtype)
        sampled = F.grid_sample(
            pixels.permute(2, 0, 1).unsqueeze(0),
            grid.reshape(1, 1, -1, 2), mode="bilinear", padding_mode="border",
            align_corners=True,
        ).reshape(3, -1).T.reshape(*shape, 3)
        scale = 2.0 ** self.exposure
        if roughness is not None:
            blur = roughness.clamp(0, 1)
            mean = pixels.mean((0, 1))
            sampled = sampled * (1 - blur.square()) + mean * blur.square()
        return sampled * scale

    def diffuse_sample(
        self,
        directions: torch.Tensor,
        roughness: torch.Tensor | None = None,
        target_luminance: float | None = None,
    ):
        """White-balanced, achromatic irradiance for diffuse albedo shading.

        Dividing by each HDR channel mean prevents the environment palette
        from being baked into albedo. The remaining signal is converted to
        luminance, while exposure and spatial intensity variation are kept.
        Specular sampling intentionally continues to use ``sample`` so colored
        highlights remain possible.
        """
        sampled = self.sample(directions, roughness)
        pixels = self.pixels.to(device=sampled.device, dtype=sampled.dtype)
        channel_mean = pixels.mean((0, 1)).clamp_min(1e-4)
        mean_luminance = _luminance(channel_mean).clamp_min(1e-4)
        balanced = sampled * (mean_luminance / channel_mean)
        luminance = _luminance(balanced)
        if target_luminance is not None:
            # Establish a scene-referred white: an average environment at
            # exposure zero illuminates diffuse white to target_luminance.
            # The sampled exposure remains in `sampled`, so exposure stops and
            # spatial HDR variation are preserved.
            luminance = luminance * (float(target_luminance) / mean_luminance)
        return luminance.expand_as(sampled)


class EnvironmentPool:
    def __init__(self, directory: str = "", seed: int = 42):
        paths: list[Path] = []
        if directory:
            root = Path(directory).expanduser()
            if root.exists():
                paths = sorted(p for p in root.iterdir() if p.suffix.lower() in {".hdr", ".exr"})
        maps = [EnvironmentMap(path.stem, load_environment(path)) for path in paths]
        if len(maps) < 6:
            maps.extend(
                EnvironmentMap(f"procedural_{i}", procedural_environment(i))
                for i in range(len(maps), 6)
            )
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(maps)).tolist()
        maps = [maps[i] for i in order]
        split = max(1, int(round(len(maps) * 0.67)))
        self.train = maps[:split]
        self.heldout = maps[split:] or maps[-1:]
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def sample(self, training: bool, exposure_range=(-1.0, 1.0)) -> EnvironmentMap:
        candidates = self.train if training else self.heldout
        index = int(torch.randint(len(candidates), (1,), generator=self.generator).item())
        yaw = float(torch.rand((), generator=self.generator).item())
        lo, hi = exposure_range
        exposure = lo + (hi - lo) * float(torch.rand((), generator=self.generator).item())
        source = candidates[index]
        return EnvironmentMap(source.name, source.pixels, yaw, exposure)

    @property
    def fixed(self) -> EnvironmentMap:
        source = self.train[0]
        return EnvironmentMap(source.name, source.pixels, 0.0, 0.0)

    @property
    def neutral(self) -> EnvironmentMap:
        pixels = torch.full((64, 128, 3), 0.72, dtype=torch.float32)
        pixels[:32] *= torch.linspace(1.15, 0.85, 32).view(32, 1, 1)
        return EnvironmentMap("neutral_texture_stage", pixels, 0.0, 0.0)
