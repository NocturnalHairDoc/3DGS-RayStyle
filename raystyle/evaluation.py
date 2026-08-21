from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .atlas import AtlasTopology
from .backend import LegacyGaussianBackend
from .config import ExperimentConfig
from .environments import EnvironmentMap, EnvironmentPool
from .features import (
    FrozenDINO, adjacent_patch_distance, corresponded_patch_distance,
    dino_content_loss, masked_feature_stats, masked_patch_tokens,
)
from .io_utils import load_image, save_image, seed_everything, write_json
from .losses import (
    ReferenceStyleLoss, boundary_outside_preservation_loss,
    illumination_consistency_loss, masked_gradient_retention,
    masked_lab_mean_distance,
    outside_preservation_loss,
)
from .project_state import load_segment
from .reference_layout import build_reference_layout, metric_tile_grid
from .style_state import StyleState
from .texture_field import AtlasTextureField


def _source_uv_stats(source_uv: torch.Tensor, bins=24):
    if not len(source_uv):
        return 0.0, 0.0, 0.0, source_uv.new_zeros(3, bins, bins)
    cells = (source_uv.clamp(0, 1) * (bins - 1)).long()
    linear = cells[:, 1] * bins + cells[:, 0]
    histogram = torch.bincount(linear, minlength=bins * bins).float().reshape(bins, bins)
    probabilities = histogram.flatten() / histogram.sum().clamp_min(1)
    occupied = probabilities > 0
    entropy = -(
        probabilities[occupied] * probabilities[occupied].log()
    ).sum() / np.log(bins * bins)
    span = (source_uv.amax(0) - source_uv.amin(0)).prod()
    heatmap = torch.log1p(histogram)
    heatmap = heatmap / heatmap.amax().clamp_min(1)
    heatmap = torch.stack((heatmap, heatmap.sqrt(), 1 - heatmap), dim=0)
    return (
        float(occupied.float().mean()), float(span), float(entropy), heatmap,
    )


@torch.no_grad()
def evaluate(config: ExperimentConfig, checkpoint_path: str):
    seed_everything(config.train.seed)
    output = Path(config.output_dir) / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    backend = LegacyGaussianBackend(
        config.legacy_root, config.model_path, config.source_path,
        config.images, config.resolution, config.white_background,
    )
    selected, _ = load_segment(config.project_state, config.segment_id, backend.point_count)
    selected = selected.cuda()
    payload = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    metadata = payload.get("state_metadata", {})
    atlas_topology = AtlasTopology.from_checkpoint_state(payload["state_dict"])
    saved_method = metadata.get("method")
    if saved_method and saved_method != config.method:
        raise ValueError(
            f"checkpoint was trained with method {saved_method!r}, "
            f"but evaluation config requests {config.method!r}"
        )
    state = StyleState(
        backend.base_albedo, selected, config.method,
        original_sh_degree=int(backend.gaussians.active_sh_degree),
        residual_degree=int(metadata.get("residual_degree", config.train.sh_degree)),
        residual_limit=float(metadata.get("residual_limit", config.train.residual_limit)),
        global_shift_limit=float(metadata.get(
            "global_shift_limit", config.train.global_shift_limit,
        )),
        detail_residual_limit=float(metadata.get(
            "detail_residual_limit", config.train.detail_residual_limit,
        )),
        selected_xyz=backend.xyz[selected] if config.method == "ours" else None,
        selected_normals=(
            backend.canonical_normals(torch.where(selected)[0])
            if config.method == "ours" else None
        ),
        selected_visibility=(
            backend.gaussians.get_opacity.detach()[selected]
            if config.method == "ours" else None
        ),
        texture_resolution=int(
            metadata.get("texture_resolution") or config.train.texture_resolution
        ),
        texture_logit_limit=config.train.texture_logit_limit,
        texture_mapping=str(metadata.get("texture_mapping", "planar")),
        atlas_charts=int(metadata.get("atlas_chart_count", config.train.atlas_charts)),
        atlas_neighbours=int(metadata.get("atlas_neighbours", config.train.atlas_neighbours)),
        atlas_padding=int(metadata.get("atlas_padding", config.train.atlas_padding)),
        atlas_feather=float(metadata.get("atlas_feather", config.train.atlas_feather)),
        atlas_uv_offset_limit=float(metadata.get(
            "atlas_uv_offset_limit", config.train.atlas_uv_offset_limit,
        )),
        atlas_source_layout=str(metadata.get(
            "atlas_source_layout", config.train.atlas_source_layout,
        )),
        atlas_reference_repeat=int(metadata.get(
            "atlas_reference_repeat", config.train.atlas_reference_repeat,
        )),
        atlas_topology=atlas_topology,
        albedo_mode=str(metadata.get("albedo_mode", "additive")),
        pbr_diffuse_white=float(metadata.get("pbr_diffuse_white", 0.0)),
        pbr_exposure=float(metadata.get("pbr_exposure", 0.0)),
        pbr_white_point=float(metadata.get("pbr_white_point", 0.0)),
    ).cuda()
    state.load_checkpoint_state(payload["state_dict"])
    state.eval()
    dino = FrozenDINO(config.legacy_root, config.dino_checkpoint, config.train.image_size).cuda()
    reference = load_image(config.reference_image)
    tile_grid = None
    if config.train.reference_metric_tiles:
        tile_grid = metric_tile_grid(
            state.texture_field, backend.xyz[selected], config.train.reference_tile_count,
        )
    reference_layout = build_reference_layout(
        reference,
        mode=config.train.reference_layout,
        patch_count=config.train.reference_saliency_patches,
        canvas_size=config.train.texture_resolution,
        focus_scale=config.train.reference_focus_scale,
        tile_count=config.train.reference_tile_count,
        tile_grid=tile_grid,
    )
    save_image(output / "reference_canvas.png", reference_layout.canvas)
    save_image(output / "reference_selection.png", reference_layout.selection_preview)
    write_json(output / "reference_layout.json", {
        "mode": config.train.reference_layout,
        "regions": [list(region) for region in reference_layout.regions],
        "scales": list(reference_layout.scales),
        "tile_grid": list(reference_layout.tile_grid),
    })
    reference_features = dino(reference)
    layout_features = dino(reference_layout.canvas)
    # Primary evaluation always uses the untouched reference so layout/method
    # comparisons share one target. The layout-specific bank is diagnostic.
    style_objective = ReferenceStyleLoss(reference_features, reference)
    layout_style_objective = ReferenceStyleLoss(
        reference_features, reference,
        patch_dino_features=layout_features,
        patch_reference_rgb=reference_layout.canvas,
    )
    environments = EnvironmentPool(config.environment_dir, config.train.seed)
    fixed = environments.fixed
    heldout = environments.heldout[:config.evaluation.heldout_env_count]
    cameras = backend.test_cameras[:config.evaluation.max_views]

    rows = []
    descriptors = defaultdict(list)
    patch_tokens = defaultdict(list)
    correspondence_tokens = defaultdict(list)
    correspondence_ids = state.selected_ids
    if len(correspondence_ids) > 4096:
        subset = torch.linspace(
            0, len(correspondence_ids) - 1, 4096,
            device=correspondence_ids.device,
        ).long()
        correspondence_ids = correspondence_ids[subset]
    fixed_images = {}
    source_uv_colors = None
    source_reference_colors = None
    if isinstance(state.texture_field, AtlasTextureField):
        source_uv_colors = torch.zeros(
            backend.point_count, 3, device=state.selected_ids.device,
        )
        source_uv = state.texture_field.current_reference_uv()
        source_uv_colors[state.selected_ids] = torch.cat(
            (source_uv, 1 - source_uv[:, :1]), dim=1,
        )
        source_reference_colors = torch.zeros_like(source_uv_colors)
        source_reference_colors[state.selected_ids] = AtlasTextureField._sample_at(
            reference_layout.canvas.unsqueeze(0), source_uv,
        )
    for view_index, camera in enumerate(cameras):
        original = backend.render_original(camera)
        original_features = dino(original)
        mask = backend.segment_mask(camera, state.selected_ids)
        visible_ids, visible_grid = backend.projected_visible_samples(
            camera, state.selected_ids, correspondence_ids, mask,
        )
        source_coverage = source_span = source_entropy = None
        if source_uv_colors is not None:
            # Render UV as colour through the same Gaussian rasterizer used by
            # appearance. Dividing by segment alpha recovers the actual source
            # coordinates at screen pixels without an approximate z-buffer.
            source_render = backend.render_colors(camera, source_uv_colors)
            support = mask[0] > 0.15
            unpremultiplied = source_render / mask.clamp_min(1e-4)
            screen_source_uv = unpremultiplied[:2, support].T.clamp(0, 1)
            source_coverage, source_span, source_entropy, source_heatmap = (
                _source_uv_stats(screen_source_uv)
            )
            save_image(output / "source_uv" / f"view_{view_index:03d}.png", source_heatmap)
            save_image(
                output / "source_uv_render" / f"view_{view_index:03d}.png",
                unpremultiplied.clamp(0, 1) * mask,
            )
            source_reference_render = backend.render_colors(
                camera, source_reference_colors,
            )
            save_image(
                output / "source_reference_render" / f"view_{view_index:03d}.png",
                source_reference_render.clamp(0, 1),
            )
        for split, env_list in (("fixed", [fixed]), ("unseen_hdr", heldout)):
            for env_index, source_env in enumerate(env_list):
                env = EnvironmentMap(source_env.name, source_env.pixels, 0.0, 0.0)
                render_mode = (
                    "albedo" if config.train.albedo_only_render
                    else config.train.render_mode
                )
                image = (
                    backend.render_albedo(camera, state)
                    if render_mode == "albedo"
                    else backend.render_stylized(
                        camera, state, env, render_mode=render_mode,
                    )
                )
                features = dino(image)
                global_style_distance = style_objective.global_loss(image, features, mask)
                patch_match_distance = style_objective.patch_loss(image, features, mask)
                layout_patch_match_distance = layout_style_objective.patch_loss(
                    image, features, mask,
                )
                lab_mean_distance, lab_covariance_distance = (
                    style_objective.lab_diagnostics(image, mask)
                )
                descriptor = torch.cat(masked_feature_stats(features, mask), dim=1).squeeze(0)
                descriptors[(split, env.name)].append(descriptor.cpu())
                patch_tokens[(split, env.name)].append(
                    masked_patch_tokens(features, mask).cpu()
                )
                correspondence_tokens[(split, env.name)].append((
                    visible_ids.cpu(),
                    backend.sample_projected_features(features, visible_grid).cpu(),
                ))
                row = {
                    "view": view_index,
                    "split": split,
                    "environment": env.name,
                    "style_distance": float(global_style_distance + patch_match_distance),
                    "patch_match_distance": float(patch_match_distance),
                    "layout_patch_match_distance": float(layout_patch_match_distance),
                    "lab_mean_distance": float(lab_mean_distance),
                    "lab_covariance_distance": float(lab_covariance_distance),
                    "content_distance": float(dino_content_loss(features, original_features, mask)),
                    "outside_leakage": float(outside_preservation_loss(image, original, mask)),
                    "boundary_outside_leakage": float(
                        boundary_outside_preservation_loss(image, original, mask)
                    ),
                }
                if source_reference_colors is not None:
                    row["screen_reference_gradient_retention"] = float(
                        masked_gradient_retention(
                            image, source_reference_render, mask,
                        )
                    )
                if source_coverage is not None:
                    row.update({
                        "source_uv_coverage": source_coverage,
                        "source_uv_span": source_span,
                        "source_uv_entropy": source_entropy,
                    })
                if split == "fixed":
                    fixed_images[view_index] = image
                    save_image(output / "fixed" / f"view_{view_index:03d}.png", image)
                else:
                    response = ((image - fixed_images[view_index]).abs() * mask).sum() / (mask.sum() * 3).clamp_min(1e-4)
                    row["relighting_response"] = float(response)
                    row["hdr_lab_mean_shift"] = float(
                        masked_lab_mean_distance(
                            image, fixed_images[view_index], mask,
                        )
                    )
                    row["hdr_gradient_retention"] = float(
                        masked_gradient_retention(
                            image, fixed_images[view_index], mask,
                        )
                    )
                    row["texture_structure_distance"] = float(
                        illumination_consistency_loss(
                            fixed_images[view_index], image, mask,
                        )
                    )
                    save_image(output / "unseen_hdr" / f"view_{view_index:03d}_{env_index:02d}.png", image)
                rows.append(row)

    summary = {"method": config.method, "checkpoint": str(Path(checkpoint_path).resolve()), "count": len(rows)}
    for name, value in state.atlas_diagnostics().items():
        summary[f"atlas/{name}"] = float(value)
    for split in ("fixed", "unseen_hdr"):
        split_rows = [row for row in rows if row["split"] == split]
        for metric in (
            "style_distance", "content_distance", "outside_leakage",
            "boundary_outside_leakage",
            "patch_match_distance", "lab_mean_distance", "lab_covariance_distance",
            "layout_patch_match_distance",
            "source_uv_coverage", "source_uv_span", "source_uv_entropy",
            "screen_reference_gradient_retention",
            "relighting_response", "texture_structure_distance",
            "hdr_lab_mean_shift", "hdr_gradient_retention",
        ):
            values = [row[metric] for row in split_rows if metric in row]
            if values:
                summary[f"{split}/{metric}_mean"] = float(np.mean(values))
                summary[f"{split}/{metric}_std"] = float(np.std(values))
        environment_stds = []
        for (descriptor_split, _), values in descriptors.items():
            if descriptor_split == split and len(values) > 1:
                environment_stds.append(float(torch.stack(values).std(0).mean()))
        if environment_stds:
            summary[f"{split}/multiview_descriptor_std"] = float(np.mean(environment_stds))
        adjacent_patch_distances = []
        for (patch_split, _), values in patch_tokens.items():
            if patch_split == split and len(values) > 1:
                for first, second in zip(values[:-1], values[1:]):
                    distance = adjacent_patch_distance(first, second)
                    if distance is not None:
                        adjacent_patch_distances.append(float(distance))
        if adjacent_patch_distances:
            summary[f"{split}/adjacent_view_patch_consistency"] = float(
                np.mean(adjacent_patch_distances)
            )
        overlap_distances = []
        for (token_split, _), values in correspondence_tokens.items():
            if token_split == split and len(values) > 1:
                for first, second in zip(values[:-1], values[1:]):
                    distance = corresponded_patch_distance(
                        first[0], first[1], second[0], second[1],
                    )
                    if distance is not None:
                        overlap_distances.append(float(distance))
        if overlap_distances:
            summary[f"{split}/overlap_aware_adjacent_patch_consistency"] = float(
                np.mean(overlap_distances)
            )
    environment_report = {}
    for environment in heldout:
        environment_rows = [
            row for row in rows
            if row["split"] == "unseen_hdr" and row["environment"] == environment.name
        ]
        environment_report[environment.name] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in environment_rows])),
                "std": float(np.std([row[metric] for row in environment_rows])),
            }
            for metric in (
                "hdr_lab_mean_shift", "hdr_gradient_retention",
                "texture_structure_distance", "style_distance",
            )
        }
    if environment_report:
        summary["unseen_hdr/by_environment"] = environment_report
    write_json(output / "per_view.json", rows)
    write_json(output / "summary.json", summary)
    return summary
