# MVP experiment contract

## Hypothesis

For a fixed 3D Gaussian geometry and a user-selected semantic region,
optimizing style in an explicit material space plus a low-order artistic
radiance residual yields better appearance under held-out illumination than
optimizing only view-dependent radiance.

## Frozen and trainable state

The scene's Gaussian centers, log scales, quaternions, and opacity logits are
never placed in an optimizer. A numerical fingerprint of all four tensors is
recorded before training and checked after the final iteration.

Scene geometry and calibrated cameras may be resolved from any compatible
sibling `3DGS-RTMaterial*` version through its saved `cfg_args`; the clean V3
implementation remains the renderer dependency.

PCA of the selected Gaussian centers defines a deterministic local frame.
Three projections use the PCA-axis pairs (YZ, XZ, XY), robustly normalized
with 0.5% and 99.5% bounds. Absolute Gaussian normal components in that frame,
sharpened to the fourth power, blend the three chart samples. For selected
Gaussian `i`, the proposed model learns

```text
albedo_i = sigmoid(delta_global + sum_k w_i,k bilinear(T_abs,k, uv_i,k))
roughness_i = 0.04 + 0.96 sigmoid(r_i)
metallic_i = sigmoid(m_i)
residual_i(v) = sum_(l<=1,m) c_i,lm Y_lm(v)
```

`T_abs` is a bounded absolute RGB logit texture initialized from the style
reference. It replaces the old Gaussian albedo instead of being added to it.
The first three quantities enter image-based PBR shading. The final
term is bounded and added after PBR shading to represent non-physical artistic
effects.

## Objective

```text
L = ws Lglobal + wpatch Lpatch + wc Lcontent + wg Lgraph + wo Loutside + wp Lprior
    + whdr Lhdr-consistency + wcolor Lintrinsic-color + wrender Lrender-color
```

- `Lglobal`: DINO channel mean/standard deviation plus RGB mean, covariance,
  and gradient statistics against the reference image.
- `Lpatch`: first crops image, mask, and feature support to the padded visible
  segment ROI, then performs masked DINO nearest-neighbour matching plus
  centered RGB 5x5 patch matching at 96, 160, and 224 pixel scales. Patches
  require at least 60% foreground support and contain normalized chromaticity
  and log-luminance gradients rather than raw RGB.
- `Lhdr-consistency`: explicitly divides both renders by their masked segment
  mean luminance, then compares local-normalized luminance structure,
  gradients, and chromaticity under two distinct training HDRs.
- `Lintrinsic-color`: RGB and normalized CIE Lab channel-mean agreement between
  intrinsic UV albedo and the reference. It is evaluated before PBR so HDR
  colour and exposure cannot corrupt the material colour target.
- `Lrender-color`: masked RGB and CIE Lab mean agreement after PBR and tone
  mapping. The sampled environment exposure is inverted before comparison so
  exposure augmentation is not baked into intrinsic albedo.
- `Lcontent`: masked cosine distance between frozen DINO features of the
  original and stylized render.
- `Lgraph`: weighted L1 difference between material/style values aggregated on
  neighbouring spatial anchors.
- `Loutside`: RGB L1 outside the soft rendered segment mask.
- `Lprior`: weak penalty on extreme roughness and unnecessary global colour
  correction. Replacement mode has no pull toward the old albedo.

Stage 1 uses neutral lighting and freezes roughness, metallic, and SH residuals
so the UV field must explain the reference pattern. Stage 2 independently
samples a calibrated camera, environment map, environment yaw, and exposure
and optimizes all allowed appearance parameters. Reference and content encoders
remain frozen throughout.

Diffuse HDR samples are divided by their per-channel environment mean,
collapsed to luminance, and calibrated so average exposure-zero diffuse white
equals `pbr_diffuse_white` (default 1). Colored HDR is retained for specular
reflections. Output exposure and extended-Reinhard white point are explicit;
the default white point of 1 is identity below display white and avoids the
legacy fourfold darkening. The global albedo shift is bounded to ±0.7, the SH
residual to ±0.08, and a ±0.08 high-pass UV residual is added after PBR tone
mapping to preserve painterly strokes.

## Baselines

- `dc`: one view-invariant RGB radiance residual per selected Gaussian.
- `full_sh`: native degree-3 SH radiance residuals.
- `pbr_only`: the same material branch as Ours without an artistic residual.

All methods use the same cameras, segment, reference, iteration count, random
seed, content/style losses, graph construction, and outside penalty. DC and
full-SH intentionally do not respond to HDR lighting; this is a property of the
representation, not an omitted augmentation.

## Evaluation

Fixed-light and held-out-HDR results report:

- style distance: the actual lightweight training-style objective;
- content distance: masked DINO cosine distance;
- outside leakage: RGB change outside the selected segment;
- multi-view descriptor standard deviation: view variation, computed per HDR
  environment before averaging;
- relighting response: selected-region RGB change from fixed to held-out light.

Relighting response is diagnostic rather than a quality score: zero indicates
that a representation cannot react to lighting, while a large value does not
by itself prove physically correct relighting. Visual grids and, in the next
stage, a synthetic dataset with ground-truth relighting are needed for that
claim.

## Required ablations

1. Ours vs DC vs full-SH vs PBR-only.
2. Ours without random HDR augmentation.
3. Ours without graph regularization.
4. SH residual degree 0, 1, and 2.
5. At least three segment sizes and three style types: painterly, material-like,
   and colour-dominant.

Set `train.random_hdr: false` for ablation 2, set `losses.graph: 0` for
ablation 3, and change `train.sh_degree` for ablation 4.
