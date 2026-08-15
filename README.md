# RayStyle for 3D Gaussian Splatting

RayStyle is a research prototype for local style transfer on a pretrained 3D
Gaussian Splatting scene. It changes the appearance of a selected segment while
keeping Gaussian position, scale, rotation, and opacity fixed.

The implementation is intended for experiments on view consistency and
relighting. It does not use a diffusion model.

## Method

For the selected Gaussians, RayStyle optimizes:

- a tri-planar albedo texture field;
- roughness and metallic values;
- a bounded low-order spherical-harmonic residual.

The geometry and all Gaussians outside the segment remain frozen. During
training, the renderer samples calibrated scene cameras and HDR environments.
The objective combines reference style statistics, segment-cropped multi-scale
patch matching, DINO content preservation, an anchor-graph regularizer, an
outside-region preservation term, and luminance-normalized HDR consistency.
By default the graph covers albedo, light-independent detail, PBR material, and
the low-order SH residual. Set `train.graph_scope: material` to reproduce the
older roughness/metallic-only graph for an ablation.

Three simpler parameterizations are included for comparison:

| Method | Trainable appearance |
| --- | --- |
| `ours` | tri-planar albedo, PBR material, low-order SH residual |
| `dc` | DC colour residual only |
| `full_sh` | all native SH coefficients |
| `pbr_only` | per-Gaussian albedo, roughness, and metallic |

## Repository layout

```text
raystyle/        training, rendering, losses, evaluation, and viewer
configs/         portable example configuration
scripts/         weight download and ablation helpers
tests/           CPU-safe unit tests
docs/            experiment definition and cross-scene notes
style/           location for user-provided reference images
environment.yml  Conda dependencies used by RayStyle
```

Datasets, Gaussian checkpoints, DINO weights, segment masks, HDR maps, and
training outputs are deliberately excluded from Git.

## Requirements

- Python 3.10 or newer
- PyTorch with CUDA support
- a working 3DGS environment with the required CUDA rasterizer
- a compatible RTMaterial/SAGA checkout for scene loading and DINO utilities
- an already trained Gaussian scene

The project was developed alongside `3DGS-RTMaterial-clean-validation`. The
path to that checkout is supplied through the experiment configuration; no
upstream source is copied into this repository.

## Installation

First prepare the upstream 3DGS environment, including PyTorch, CUDA, and its
compiled rasterizers. RayStyle adds its Python dependencies to that environment:

```bash
conda env update --file environment.yml
conda activate gaussian_splatting_v2
```

Do not add `--prune` when updating an existing environment, because it may
remove packages and CUDA extensions required by the upstream renderer. PyTorch
and the CUDA toolkit are intentionally not pinned in `environment.yml`; their
versions must match the renderer installation.

Download the original DINO ViT-B/8 checkpoint:

```bash
bash scripts/download_dino_vitb8.sh
```

The script writes `weights/dino_vitbase8_pretrain.pth`. Model weights are not
tracked by Git.

## Preparing a segment

The input scene must already have a trained 3DGS model. A segment can be loaded
from either:

- a project-state `.npz` exported by the existing GUI; or
- a boolean `precomputed_mask.pt` saved by **Save current segment**.

For a project-state file, inspect the available one-based segment identifiers:

```bash
python -m raystyle inspect-segments \
  --project-state /path/to/scene_project.npz
```

A standalone boolean mask contains a single segment, so `segment_id` is ignored.
The original PLY and project state are never modified.

## Configuration

Copy the example before changing paths:

```bash
cp configs/mvp.example.yaml configs/local_scene.yaml
```

At minimum, set:

```yaml
workspace_root: /path/to/3DGS/workspace
scene: 3DGS-RTMaterial:bicycle
legacy_root: /path/to/3DGS-RTMaterial-clean-validation
project_state: /path/to/precomputed_mask.pt
segment_id: 1
reference_image: /path/to/reference.jpg
dino_checkpoint: /path/to/dino_vitbase8_pretrain.pth
environment_dir: /path/to/hdr/files
output_dir: ../outputs/bicycle_style
```

When `scene` is set, the scene catalog reads the trained model and dataset paths
from its `cfg_args`. Explicit `model_path` and `source_path` values can be used
for scenes outside the catalog.

HDR files are optional. If fewer than six `.hdr` or `.exr` files are available,
the environment pool is completed with deterministic procedural maps. Real HDR
maps are preferable when evaluating unseen illumination.

## Training

```bash
python -m raystyle train --config configs/local_scene.yaml
```

The default schedule first trains the texture under neutral illumination, then
unlocks material parameters and the SH residual while sampling HDR lighting.
Outputs include resolved configuration, JSONL loss logs, previews, texture
images, and checkpoints.

To run a short smoke test without editing the YAML:

```bash
python -m raystyle train \
  --config configs/local_scene.yaml \
  --iterations 100 \
  --output-dir outputs/smoke_test
```

## Viewer

```bash
python -m raystyle view \
  --config configs/local_scene.yaml \
  --checkpoint outputs/bicycle_style/checkpoint_latest.pt \
  --scale 2
```

The viewer provides calibrated training and test cameras, free-orbit navigation,
original/stylized comparison, segment inspection, HDR selection, yaw, exposure,
and screenshot export. In free-orbit mode, use left drag to rotate, middle drag
to pan, and the mouse wheel to zoom.

## Evaluation

```bash
python -m raystyle evaluate \
  --config configs/local_scene.yaml \
  --checkpoint outputs/bicycle_style/checkpoint_latest.pt
```

Evaluation renders fixed and held-out HDR conditions and reports style distance,
content distance, outside leakage, multi-view descriptor variation, relighting
response, and texture-structure error. Per-view measurements and rendered images
are written below the experiment's `evaluation/` directory.

Run all four methods with:

```bash
bash scripts/run_all_methods.sh configs/local_scene.yaml
```

## Current limitations

- Viewpoints are sampled from calibrated scene cameras rather than arbitrary
  interpolated poses.
- The renderer uses Gaussian normal estimates and approximate image-based PBR
  lighting; OptiX is not part of the differentiable path.
- Tri-planar mapping preserves local strokes more reliably than a single planar
  projection, but it does not preserve the full composition of a reference
  painting on small or highly irregular segments.
- Soft segment boundaries and heavy occlusion can produce visible colour leakage.
- Strong specular response can obscure high-frequency texture detail.

The loss definitions and evaluation protocol are described in
[docs/EXPERIMENT.md](docs/EXPERIMENT.md). Cross-scene observations are recorded
in [docs/OTHER_SCENES_TEST.md](docs/OTHER_SCENES_TEST.md). Results for the
appearance-graph ablation are reported in
[docs/GRAPH_SCOPE_ABLATION.md](docs/GRAPH_SCOPE_ABLATION.md).

## License note

This repository does not redistribute the upstream 3DGS/SAGA implementation,
datasets, or pretrained model weights. Their original licenses still apply.
No separate license has been selected for this prototype yet.
