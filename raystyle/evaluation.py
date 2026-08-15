from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .backend import LegacyGaussianBackend
from .config import ExperimentConfig
from .environments import EnvironmentMap, EnvironmentPool
from .features import FrozenDINO, dino_content_loss, masked_feature_stats
from .io_utils import load_image, save_image, seed_everything, write_json
from .losses import (
    ReferenceStyleLoss, illumination_consistency_loss, outside_preservation_loss,
)
from .project_state import load_segment
from .style_state import StyleState


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
        texture_resolution=int(
            metadata.get("texture_resolution") or config.train.texture_resolution
        ),
        texture_logit_limit=config.train.texture_logit_limit,
        texture_mapping=str(metadata.get("texture_mapping", "planar")),
        albedo_mode=str(metadata.get("albedo_mode", "additive")),
        pbr_diffuse_white=float(metadata.get("pbr_diffuse_white", 0.0)),
        pbr_exposure=float(metadata.get("pbr_exposure", 0.0)),
        pbr_white_point=float(metadata.get("pbr_white_point", 0.0)),
    ).cuda()
    state.load_checkpoint_state(payload["state_dict"])
    state.eval()
    dino = FrozenDINO(config.legacy_root, config.dino_checkpoint, config.train.image_size).cuda()
    reference = load_image(config.reference_image)
    style_objective = ReferenceStyleLoss(dino(reference), reference)
    environments = EnvironmentPool(config.environment_dir, config.train.seed)
    fixed = environments.fixed
    heldout = environments.heldout[:config.evaluation.heldout_env_count]
    cameras = backend.test_cameras[:config.evaluation.max_views]

    rows = []
    descriptors = defaultdict(list)
    fixed_images = {}
    for view_index, camera in enumerate(cameras):
        original = backend.render_original(camera)
        original_features = dino(original)
        mask = backend.segment_mask(camera, state.selected_ids)
        for split, env_list in (("fixed", [fixed]), ("unseen_hdr", heldout)):
            for env_index, source_env in enumerate(env_list):
                env = EnvironmentMap(source_env.name, source_env.pixels, 0.0, 0.0)
                image = backend.render_stylized(camera, state, env)
                features = dino(image)
                descriptor = torch.cat(masked_feature_stats(features, mask), dim=1).squeeze(0)
                descriptors[(split, env.name)].append(descriptor.cpu())
                row = {
                    "view": view_index,
                    "split": split,
                    "environment": env.name,
                    "style_distance": float(style_objective(image, features, mask)),
                    "content_distance": float(dino_content_loss(features, original_features, mask)),
                    "outside_leakage": float(outside_preservation_loss(image, original, mask)),
                }
                if split == "fixed":
                    fixed_images[view_index] = image
                    save_image(output / "fixed" / f"view_{view_index:03d}.png", image)
                else:
                    response = ((image - fixed_images[view_index]).abs() * mask).sum() / (mask.sum() * 3).clamp_min(1e-4)
                    row["relighting_response"] = float(response)
                    row["texture_structure_distance"] = float(
                        illumination_consistency_loss(
                            fixed_images[view_index], image, mask,
                        )
                    )
                    save_image(output / "unseen_hdr" / f"view_{view_index:03d}_{env_index:02d}.png", image)
                rows.append(row)

    summary = {"method": config.method, "checkpoint": str(Path(checkpoint_path).resolve()), "count": len(rows)}
    for split in ("fixed", "unseen_hdr"):
        split_rows = [row for row in rows if row["split"] == split]
        for metric in (
            "style_distance", "content_distance", "outside_leakage",
            "relighting_response", "texture_structure_distance",
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
    write_json(output / "per_view.json", rows)
    write_json(output / "summary.json", summary)
    return summary
