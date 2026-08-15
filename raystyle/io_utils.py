from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_image(path: str, device="cuda"):
    image = Image.open(path).convert("RGB")
    values = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return values.to(device)


def save_image(path: str | Path, image: torch.Tensor):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    values = image.detach().clamp(0, 1).permute(1, 2, 0).mul(255).byte().cpu().numpy()
    Image.fromarray(values).save(target)


def append_jsonl(path: str | Path, payload: dict):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

