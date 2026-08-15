# Graph-scope ablation

## Setup

This experiment compares the former material-only graph with the expanded
appearance graph on the bicycle road segment. Both runs use the same segment,
Starry Night reference, seed, cameras, procedural HDR pool, losses, and 2,000
training iterations. The only configuration difference is `train.graph_scope`.

| Run | Scope | Output |
| --- | --- | --- |
| Material | roughness, metallic | `outputs/bicycle/road_starry_graph_material` |
| Appearance | albedo, detail, roughness, metallic, SH | `outputs/bicycle/road_starry_graph_appearance` |

The material run took 10 min 12 s and the appearance run took 10 min 17 s on
an RTX 5090.

## Render-space results

Lower is better for all distances and leakage values in this table.

| Metric | Material | Appearance |
| --- | ---: | ---: |
| Fixed style distance | 0.755219 | 0.755292 |
| Fixed content distance | 0.490451 | 0.490788 |
| Fixed outside leakage | 0.002930 | 0.002933 |
| Fixed multi-view descriptor std | 0.007252 | 0.007265 |
| Unseen-HDR style distance | 0.756467 | 0.756467 |
| Unseen-HDR structure distance | 0.050259 | 0.050312 |
| Unseen-HDR multi-view descriptor std | 0.007282 | 0.007285 |
| Relighting response | 0.010871 | 0.010873 |

At graph weight 0.05, the two render-space results are effectively tied. The
mean absolute difference between matching fixed-light frames is 0.00121 in
normalized RGB (about 0.31/255). The 99th-percentile channel difference is
about 5.1/255 and is concentrated on the edited road.

## Parameter-space results

Each checkpoint was measured with the same corrected anchor graph. These
values are weighted L1 differences between neighbouring anchor means.

| Field | Material run | Appearance run | Change |
| --- | ---: | ---: | ---: |
| Albedo | 0.122516 | 0.070609 | -42.4% |
| Detail | 0.016821 | 0.012151 | -27.8% |
| SH residual | 0.010383 | 0.000426 | -95.9% |
| Combined appearance | 0.030045 | 0.016812 | -44.0% |

The expanded graph therefore changes the intended latent quantities even
though rasterization and image-space losses make the final frames look almost
identical. The mean texture preview differs by 0.00222 RGB, while its gradient
energy is unchanged (0.21476 vs 0.21484), so this setting does not measurably
blur the reference painting.

## Interpretation

The appearance graph is safe at weight 0.05: it substantially reduces
per-anchor albedo/detail/SH discontinuities without harming style, relighting,
leakage, or visible high-frequency texture. On this already coherent
tri-planar road field, that latent smoothing does not translate into a
measurable render-space improvement.

The next useful sweep is `losses.graph` in `{0.05, 0.1, 0.2}` with appearance
scope. A stronger value is only justified if it improves multi-view metrics or
visible Gaussian noise; otherwise 0.05 should remain the default.
