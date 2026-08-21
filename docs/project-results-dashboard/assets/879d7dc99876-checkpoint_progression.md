# Checkpoint progression

Paired Atlas versus tri-planar results. Ratios at or below 1.0 are better for multiview consistency; fold-over must remain at or below 0.5%.

| Scenario | Step | Fixed style | Unseen style | Fixed MV | Unseen MV | Fold-over | Accepted |
|---|---:|---:|---:|---:|---:|---:|:---:|
| bicycle_starry | 400 | 2.776% | 2.708% | 0.95804 | 0.96584 | 0.320% | yes |
| bicycle_starry | 2000 | 0.818% | 0.825% | 0.99305 | 0.99812 | 0.427% | yes |
| bicycle_sunflowers | 400 | 5.266% | 5.269% | 0.95433 | 0.95259 | 0.363% | yes |
| bicycle_sunflowers | 2000 | 6.120% | 6.142% | 0.95411 | 0.95288 | 0.436% | yes |
| stump_starry | 400 | 1.211% | 1.167% | 0.98822 | 0.98925 | 0.473% | yes |
| stump_starry | 500 | 1.157% | 1.141% | 0.98618 | 0.98422 | 0.500% | no |
| stump_starry | 1000 | 0.916% | 0.939% | 1.00041 | 1.00176 | 0.473% | no |
| stump_starry | 1500 | 0.576% | 0.615% | 1.00343 | 1.00630 | 0.513% | no |
| stump_starry | 2000 | 0.680% | 0.703% | 1.01166 | 1.01435 | 0.519% | no |
| bulldozer_starry | 400 | 0.781% | 0.699% | 0.98658 | 0.99375 | 0.363% | yes |
| bulldozer_starry | 2000 | 0.712% | 0.749% | 0.99393 | 0.99610 | 0.482% | yes |

## Selected checkpoints

- `bicycle_starry`: iteration 2000 — latest evaluated checkpoint passes all non-style gates and the numeric or reviewed style criterion
- `bicycle_sunflowers`: iteration 2000 — latest evaluated checkpoint passes all non-style gates and the numeric or reviewed style criterion
- `stump_starry`: iteration 400 — later iteration 2000 fails: fixed_style_1pct, unseen_style_1pct, fixed_multiview_no_worse, unseen_multiview_no_worse, uv_foldover_half_pct
- `bulldozer_starry`: iteration 2000 — latest evaluated checkpoint passes all non-style gates and the numeric or reviewed style criterion
