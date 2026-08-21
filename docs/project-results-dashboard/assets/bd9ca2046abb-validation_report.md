# Atlas 400-iteration validation

## Decision

**Do not start the 2,000-iteration runs yet.** None of the four paired scenarios passes every automatic gate, and the visual gates fail on reference-pattern recognition and chart seams.

All eight runs used seed 42, 400 total iterations, a 150-iteration texture stage, identical scene/style inputs within each pair, and differed only in `texture_mapping` and output path. The source fingerprint was `b76ab76bf4e25991862097c3ec04034e826275e0c21a4390bc4ed5d54eaf1c45`.

## Automatic gates

Positive style percentages mean Atlas is better. Ratios must be at most 1.10 for leakage, 1.05 for HDR structure, and 1.00 for multi-view standard deviation.

| Scenario | Fixed style | Unseen style | Fixed leak | Unseen leak | HDR structure | Fixed MV | Unseen MV | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Bicycle / Starry Night | +0.32% | +0.25% | 1.268 | 1.271 | 0.962 | 0.953 | 0.956 | Fail |
| Bicycle / Sunflowers | +0.68% | +0.70% | 1.051 | 1.051 | 1.140 | 1.003 | 1.009 | Fail |
| Stump / Starry Night | +1.75% | +1.84% | 1.034 | 1.034 | 1.114 | 0.952 | 0.949 | Fail |
| Bulldozer / Starry Night | -0.28% | -0.20% | 1.106 | 1.104 | 0.963 | 1.008 | 1.008 | Fail |

## Atlas diagnostics

| Scenario | UV collision rate | Seam energy | UV distortion | Reference gradient retention |
| --- | ---: | ---: | ---: | ---: |
| Bicycle / Starry Night | 0.818 | 0.236 | 0.025 | 1.087 |
| Bicycle / Sunflowers | 0.850 | 0.021 | 0.012 | 1.455 |
| Stump / Starry Night | 0.864 | 0.403 | 0.066 | 1.112 |
| Bulldozer / Starry Night | 0.878 | 0.154 | 0.053 | 1.086 |

The collision rate is unacceptably high in every scene. Stump has the highest seam energy and distortion. The Sunflowers gradient-retention value is also oversharpened relative to the reference, without retaining recognizable flower structure.

## Visual gates

| Gate | Result | Evidence |
| --- | --- | --- |
| Sunflowers contains recognizable flower heads/petals | Fail | The road is predominantly yellow/orange; no stable sunflower motif is visible. |
| No obvious chart seams | Fail | Bright rectangular blocks and chart-dependent color changes are visible, especially on Bicycle and Stump. |
| No abrupt unseen-view texture change | Not certified | The texture remains surface-attached overall, but chart discontinuities appear/disappear as different surface regions become visible. A continuous orbit test should be added after topology fixes. |

## Recommended next iteration

1. Fix topology before increasing training length. The Bicycle segment currently produces 193 charts because the selected-surface graph is highly fragmented. Use an adaptive surface graph that bridges only nearby, normal-compatible gaps while preserving the test that truly disconnected surfaces never merge.
2. Reduce UV collisions structurally. Add occupancy-aware packing and a non-neighbour collision repulsion term; report collision rate separately for intra-chart fold-over and inter-chart overlap.
3. Make seams trainable constraints. Pair boundary samples across charts, share or pad border texels, and apply the seam loss to sampled albedo rather than only Gaussian endpoints.
4. Improve motif retention after topology is stable. Use reference patch nearest-neighbour/optimal-transport matching and multi-scale crops; then consider 512px textures. More iterations alone are likely to reinforce the current yellow fill and chart blocks.
5. Repeat the same 400-iteration matrix. Start 2,000-iteration runs only after all automatic gates pass and the Sunflowers/continuous-orbit visual checks pass.

## Artifacts

- `atlas_vs_triplanar.csv`: complete numeric comparison and per-gate booleans.
- `gate_report.json`: machine-readable gate report.
- `fixed_view_comparison.jpg`: fixed-lighting paired render sheet.
- `unseen_view_comparison.jpg`: paired unseen-HDR views.
- `atlas_unseen_sequence.jpg`: six unseen views per Atlas scenario.
- `atlas_debug_comparison.jpg`: albedo, boundaries, chart IDs, texture atlas, and collision modes.
- `atlas_debug/`: full-resolution debug renders.
