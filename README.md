# RayStyle for 3D Gaussian Splatting

RayStyle is a small side project spun out of
[3DGS-RTMaterial](https://github.com/NocturnalHairDoc/3DGS-RTMaterial). It adds
local style transfer to a pretrained 3D Gaussian Splatting scene by changing the
appearance of a selected segment while keeping Gaussian position, scale,
rotation, and opacity fixed. It does not use a diffusion model.

## Demos

| Bicycle road | Stump | Kitchen bulldozer |
| :---: | :---: | :---: |
| ![Starry Night style transferred to the road in the bicycle scene](docs/assets/demo_bicycle_road_starry.jpg) | ![Starry Night style transferred to the stump](docs/assets/demo_stump_starry.jpg) | ![Starry Night style transferred to the toy bulldozer](docs/assets/demo_kitchen_bulldozer_starry.jpg) |

### Recent texture demos

| Cobblestone road | Checkerboard road | Sunflower road |
| :---: | :---: | :---: |
| ![Cobblestone texture transferred to the road in the bicycle scene](docs/assets/demo_bicycle_road_cobblestone.png) | ![Blue and yellow checkerboard transferred to the road in the bicycle scene](docs/assets/demo_bicycle_road_checker.png) | ![Sunflower pattern transferred to the road in the bicycle scene](docs/assets/demo_bicycle_road_sunflower.png) |

### Held-out HDR examples

| Bicycle sunflowers · Atlas · unseen · view 24 | Stump Starry Night · Atlas · unseen · view 11 |
| :---: | :---: |
| ![Sunflower road under held-out lighting, Atlas unseen view 24](docs/assets/demo_bicycle_sunflowers_atlas_unseen_24.webp) | ![Starry Night stump under held-out lighting, Atlas unseen view 11](docs/assets/demo_stump_starry_atlas_unseen_11.webp) |

The portable [project results dashboard](docs/project-results-dashboard/project-results-dashboard.html)
contains the complete staged comparison and its web-optimized media bundle.

## Method

For the selected Gaussians, RayStyle optimizes:

- a charted surface atlas (the earlier planar and tri-planar mappings remain available);
- roughness and metallic values;
- a bounded low-order spherical-harmonic residual.

The geometry and all Gaussians outside the segment remain frozen. During
training, the renderer samples calibrated scene cameras and HDR environments.
The objective combines reference style statistics, segment-cropped multi-scale
patch matching, DINO content preservation, an anchor-graph regularizer, an
outside-region preservation term, and luminance-normalized HDR consistency.
The atlas is built from a local kNN surface graph. Connected surface regions
are split into charts, projected with a local PCA basis, and packed into one
texture. Each chart starts from a different crop of the reference image, with
light feathering across chart boundaries. By default the graph covers albedo,
light-independent detail, PBR material, and
the low-order SH residual. Set `train.graph_scope: material` to reproduce the
older roughness/metallic-only graph for an ablation.

Three simpler parameterizations are included for comparison:

| Method | Trainable appearance |
| --- | --- |
| `ours` | atlas albedo, PBR material, low-order SH residual |
| `dc` | DC colour residual only |
| `full_sh` | all native SH coefficients |
| `pbr_only` | per-Gaussian albedo, roughness, and metallic |

## Repository layout

```text
raystyle/        training, rendering, losses, evaluation, and viewer
configs/         portable example configuration
scripts/         setup, preparation, and reproducible training entry points
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

The project uses scenes and renderer components from
[3DGS-RTMaterial](https://github.com/NocturnalHairDoc/3DGS-RTMaterial). The path
to a compatible checkout is supplied through the configuration; no upstream
source is copied into this repository.

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
bash scripts/setup/download_dino_vitb8.sh
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

train:
  texture_mapping: atlas
  texture_resolution: 512
  atlas_charts: 8
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
and screenshot export. Atlas checkpoints add projected albedo, chart boundaries,
chart-ID colors, packed texture preview, and UV-collision debug views. In
free-orbit mode, use left drag to rotate, middle drag to pan, and the mouse wheel
to zoom.

### Viewing multiple segments

Independently trained, non-overlapping segments from the same scene can be
loaded together without converting their checkpoints. Start from the example
bundle:

```bash
cp configs/multisegment.example.yaml configs/local_multisegment.yaml
python -m raystyle view-bundle \
  --bundle configs/local_multisegment.yaml \
  --scale 2
```

Each bundle entry names a segment and points to its resolved configuration and
checkpoint. All entries must use the same scene and renderer backend, and their
Gaussian masks must be disjoint. The styled view composites every segment;
Atlas diagnostic modes apply to the segment selected in the side panel. This is
a viewing and composition workflow for independent runs, not joint
multi-reference optimization.

### Comparing the four methods

A completed strict-baseline manifest can be opened directly, so every method
uses the same camera and lighting controls:

```bash
python -m raystyle view-methods \
  --manifest outputs/method_baselines_400/manifest.json \
  --experiment bicycle_starry \
  --scale 2
```

Use **Comparison method** to switch between DC-only, All SH, PBR-only, and
Atlas Ours. The normal **Original** display mode remains available. **Save
Original + all methods** renders all five images from the current camera, HDR
rotation, and exposure, then saves the individual images, a horizontal
comparison row, and `metadata.json` below the baseline's
`viewer_comparisons/` directory.

## Evaluation

```bash
python -m raystyle evaluate \
  --config configs/local_scene.yaml \
  --checkpoint outputs/bicycle_style/checkpoint_latest.pt
```

Evaluation renders fixed and held-out HDR conditions and reports style distance,
content distance, outside leakage, multi-view descriptor variation, adjacent-view
patch consistency, relighting response, and texture-structure error. Atlas runs
also report UV collision rate, chart seam energy, UV distortion, and reference
gradient retention. Held-out environments additionally report their individual
Lab mean colour shift and texture-gradient retention relative to the fixed
lighting render. Per-view measurements and rendered images are written below the
experiment's `evaluation/` directory.

The four-scene paired Atlas validation can be reproduced with one command:

```bash
python -m scripts.training.run_atlas_validation \
  --output outputs/atlas_paired_400 \
  --iterations 400 \
  --texture-stage 150 \
  --tag atlas-paired-400
```

It prepares matched tri-planar/Atlas configurations, trains all eight runs, and
evaluates fixed and held-out illumination. Each run writes its checkpoint,
rendered views, metrics, and a pipeline status file. Passing `--reuse` resumes
from completed checkpoints and evaluations. The historical
`atlas-v0-smoke-400` record in `baselines/atlas-v0-smoke-400.json` captures the
failed topology baseline before the chart, collision, and seam fixes; it should
not be used as the accepted source run below.

For a strict comparison of DC-only, all-SH, PBR-only, and Atlas parameterizations:

```bash
python -m scripts.training.run_method_baselines \
  --source-validation outputs/atlas_paired_400 \
  --output outputs/method_baselines_400 \
  --iterations 400 \
  --texture-stage 150 \
  --tag method-baselines-v1-400 \
  --reuse-ours-from-source
```

The source validation supplies the matched scene configurations and, with
`--reuse-ours-from-source`, the accepted Atlas runs. The script trains and
evaluates the other three methods under the same fixed and held-out lighting
protocol. Every method keeps its own checkpoint, rendered views, and metrics.

Atlas robustness on a narrow road strip and a continuous non-planar Stump
surface can be checked with:

```bash
python -m scripts.training.run_segment_stress_validation \
  --output outputs/atlas_segment_stress_150 \
  --iterations 150
```

This builds deterministic checker and orientation references, runs all four
stress cases, and evaluates their UV diagnostics. The thin masks can be regenerated with
`scripts/preparation/prepare_segment_stress_masks.py`.

Run all four methods with:

```bash
bash scripts/training/run_all_methods.sh configs/local_scene.yaml
```

## Validation status

The current checked experiments establish the following, rather than treating a
single training preview as sufficient evidence:

| Check | Current result |
| --- | --- |
| Four-scene Atlas validation, 400 steps | Passed the paired automatic and visual gates |
| Four-method comparison, 400 steps | Atlas improves style distance by about 7.4–16.9% over the strongest simpler baseline, with a larger DINO content distance |
| Long-run checkpoint comparison | Three scenes prefer 2,000 steps; Stump regresses and retains the 400-step checkpoint |
| Thin/non-planar representation stress test | All four 150-step checker/orientation cases pass automatic and visual review |
| Adjacent-segment isolation | Cross-core change stays below 0.2%; far-outside mean change stays below `1e-4` |

The selected checkpoint should therefore be based on validation metrics and
visual review, not assumed to be the final training iteration.

## Current limitations

- Viewpoints are sampled from calibrated scene cameras rather than arbitrary
  interpolated poses.
- The renderer uses Gaussian normal estimates and approximate image-based PBR
  lighting; OptiX is not part of the differentiable path.
- Atlas quality depends on the selected Gaussian surface graph. Sparse floating
  points and badly estimated normals can produce uneven charts or visible seams.
- A local PCA chart cannot flatten strongly curved or folded surfaces without
  some distortion; the UV constraints limit this rather than removing it.
- Narrow segments can have uneven texel density even when collision, fold-over,
  and seam diagnostics pass.
- Soft Gaussian footprints can change a small ring immediately outside an
  adjacent segment boundary. The isolation diagnostic separates this local
  footprint spread from remote surface contamination.
- Strong specular response can obscure high-frequency texture detail.
- Stronger style matching can increase DINO content distance. The style/content
  weights remain scene-dependent rather than universally optimal.
- More iterations are not always better: the Stump validation regressed between
  400 and 2,000 steps, so checkpoint selection or early stopping is required.
- Multi-segment viewing currently combines independently trained checkpoints;
  joint multi-segment optimization and per-segment enable/disable controls are
  not implemented yet.

The loss definitions and evaluation protocol are described in
[docs/EXPERIMENT.md](docs/EXPERIMENT.md). Cross-scene observations are recorded
in [docs/OTHER_SCENES_TEST.md](docs/OTHER_SCENES_TEST.md). Results for the
appearance-graph ablation are reported in
[docs/GRAPH_SCOPE_ABLATION.md](docs/GRAPH_SCOPE_ABLATION.md).

## License note

This repository does not redistribute the upstream 3DGS/SAGA implementation,
datasets, or pretrained model weights. Their original licenses still apply.
No separate license has been selected for this project yet.
