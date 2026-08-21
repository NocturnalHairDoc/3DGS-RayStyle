#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
weights_dir="${project_dir}/weights"
destination="${weights_dir}/dino_vitbase8_pretrain.pth"
partial="${destination}.part"
url="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"

mkdir -p "${weights_dir}"

if [[ ! -f "${destination}" ]]; then
  echo "Downloading official DINO ViT-B/8 backbone..."
  curl --fail --location --continue-at - --output "${partial}" "${url}"
  mv "${partial}" "${destination}"
else
  echo "Using existing ${destination}"
fi

python - "${destination}" <<'PY'
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
state = torch.load(path, map_location="cpu", weights_only=True)
if not isinstance(state, dict):
    raise SystemExit("checkpoint is not a state dictionary")
required = {"cls_token", "pos_embed", "patch_embed.proj.weight"}
missing = sorted(required.difference(state))
if missing:
    raise SystemExit(f"checkpoint is incompatible; missing keys: {missing}")
shape = tuple(state["patch_embed.proj.weight"].shape)
if shape != (768, 3, 8, 8):
    raise SystemExit(f"expected ViT-B/8 patch embedding, got {shape}")
print({"checkpoint": str(path), "bytes": path.stat().st_size, "keys": len(state), "patch_embed": shape})
PY

