# Other-scene stress test

This test first records the planar/uncalibrated baseline, then repeats the same
scenes after implementing calibrated PBR, rendered colour supervision, ROI
multi-scale patches, tri-planar charts, and luminance-normalized HDR
consistency. Each run uses 400 iterations: 150 intrinsic texture iterations
followed by 250 random-HDR PBR iterations.

## Tested segments

| Scene | Segment | Style | Selected Gaussians | Mask fraction |
|---|---|---:|---:|---:|
| kitchen | yellow bulldozer body | Starry Night | 87,524 | 5.16% |
| counter | upper oven mitt | Sunflowers | 6,473 | 0.60% |
| stump | main stump and roots | Starry Night | 111,613 | 2.38% |

The masks were produced with calibrated SAGA feature clicks and visually
checked before training. They are stored in `segments/` and their red-overlay
previews are stored in `diagnostics/`.

## Evaluation

Lower is better for style distance, leakage, structure distance, and the
descriptor standard deviation. Relighting response is not a quality score by
itself; a response near zero can also indicate an overly dark, insensitive
material.

| Segment | Fixed style dist. | Unseen-HDR style dist. | Outside leakage | Multi-view std. | HDR structure dist. |
|---|---:|---:|---:|---:|---:|
| bulldozer | 0.9322 | 0.9282 | 0.099% | 0.0047 | 0.0876 |
| oven mitt | 0.8988 | 0.8953 | 0.111% | 0.0108 | 0.0148 |
| stump | 0.9538 | 0.9506 | 0.125% | 0.0061 | 0.0522 |

Locality is good, but style distances remain high. The low multi-view standard
deviation must not be read as a success in isolation: a nearly uniform dark
surface is also stable across views.

## Implemented regression

| Segment | Fixed style old → new | HDR structure old → new | Leakage old → new |
|---|---:|---:|---:|
| bulldozer | 0.9322 → 0.7565 | 0.0876 → 0.0661 | 0.099% → 0.150% |
| oven mitt | 0.8988 → 0.7397 | 0.0148 → 0.0060 | 0.111% → 0.124% |
| stump | 0.9538 → 0.7974 | 0.0522 → 0.0272 | 0.125% → 0.463% |

Style distance improves by roughly 16--19%, and held-out-HDR structure error
drops in all scenes. PBR luminance relative to the composited UV-albedo view
improves from 26.8% to 94.0% on bulldozer, 24.9% to 90.0% on the mitt, and
27.2% to 100.8% on stump. Stump's numerical leakage rises because its brighter
edit amplifies soft-mask antialiasing at the boundary; no outside Gaussian is
made trainable.

## Baseline failure modes and current status

### 1. PBR output darkness — fixed

The learned UV texture RGB mean remains within about 0.002 of its reference in
all three runs. On the most visible evaluation camera, however, fixed-light PBR
luminance retains only the following fraction of UV-albedo luminance:

| Segment | UV luminance | Styled luminance | Styled / UV |
|---|---:|---:|---:|
| bulldozer | 0.395 | 0.106 | 26.8% |
| oven mitt | 0.622 | 0.155 | 24.9% |
| stump | 0.421 | 0.115 | 27.2% |

The procedural environment has a low diffuse energy level and the renderer
then applies `surface / (1 + surface)`. There is no calibrated diffuse white
level or output exposure compensation. The current full-grid RGB/Lab loss sees
the intrinsic UV texture, so it cannot correct this post-PBR brightness loss.

The new renderer normalizes diffuse scene white, exposes explicit output
exposure/white-point controls, and adds a masked post-PBR RGB/Lab loss with
sampled-exposure compensation.

### 2. One PCA plane — replaced by PCA-aligned tri-planar charts

The ratio between the omitted PCA-axis standard deviation and the second
in-plane standard deviation is a simple non-planarity indicator (zero is a
plane, one is volumetric):

| Segment | Thickness / mid-axis |
|---|---:|
| road | 0.177 |
| oven mitt | 0.131 |
| stump | 0.338 |
| chair | 0.450 |
| bicycle | 0.574 |
| bulldozer | 0.755 |

The bulldozer folds many disconnected parts onto the same texture plane. The
stump collapses front and back surfaces together. The oven mitt is relatively
planar, but only a small visible surface samples the full painting, so the
recognizable flower composition is lost.

Tri-planar mapping avoids forcing every side surface through one projection.
It produces stable painterly strokes, but it does not preserve one global
painting composition over a complex object: oblique normals blend three copies
of the reference. Automatic semantic charts remain future work for that use.

### 3. Patch supervision dominated by context — fixed for patch terms

The selected object can occupy only a small part of a 224-pixel training view.
Global and patch descriptors therefore receive substantial signal from the
unchanged background. The optimizer can reduce its objective with a dark local
colour wash instead of reconstructing recognizable strokes inside the object.

Patch matching now crops a padded visible-segment ROI and requires at least
60% foreground support at 96, 160, and 224 pixel scales. Global content/style
statistics intentionally retain full-scene context.

### 4. Single-view segmentation is incomplete

The feature-click mask is clean in its seed view, but the oven-mitt mask covers
the upper mitt rather than the complete pair and can miss back-facing boundary
Gaussians. This produces original-colour outlines without technically leaking
the edit outside the mask. Multi-view mask union and graph propagation are
needed before judging boundary quality on unseen views.

### 5. HDR consistency dark solution — fixed

Raw-image consistency rewards surfaces that change little across environments.
Darkening albedo response and suppressing texture contrast can lower this term.
The loss should compare illumination-normalized chroma/gradients and be paired
with an explicit rendered contrast or detail-preservation constraint.

The new loss divides each render by masked segment mean luminance and compares
normalized structure, gradients, and chromaticity. Calibrated PBR and rendered
colour supervision prevent a dark constant solution.

### 6. Full-scene cost remains high for local edits

The stump contains about 4.69 million Gaussians although only 2.38% are edited.
Its 400-iteration run took about 2 minutes 13 seconds, versus roughly 53--59
seconds for the smaller indoor scenes. Rendering is still full-scene; local
editing does not yet provide proportional compute or memory savings.

## Remaining implementation order

1. Build masks from several calibrated views and propagate labels on the
   Gaussian anchor graph.
2. Add automatic semantic charts when one global painting composition must be
   preserved instead of a tri-planar painterly material.
3. Add visibility/ROI culling for large scenes such as stump.
4. Refine soft-mask boundary evaluation for bright, high-contrast edits.
5. Retrain the road and irregular-object cases with `graph_scope: appearance`
   versus `material` to measure whether albedo/detail/SH propagation improves
   multi-view coherence without erasing deliberate high-frequency strokes.
