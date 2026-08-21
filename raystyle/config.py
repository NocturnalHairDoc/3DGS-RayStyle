from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


METHODS = ("ours", "dc", "full_sh", "pbr_only")


@dataclass
class LossConfig:
    style: float = 1.0
    patch_style: float = 2.0
    content: float = 0.5
    graph: float = 0.05
    outside: float = 10.0
    boundary_outside: float = 5.0
    material_prior: float = 0.01
    hdr_consistency: float = 0.5
    texture_anchor: float = 0.2
    texture_delta_tv: float = 0.05
    color_mean: float = 2.0
    render_color: float = 2.0
    uv_continuity: float = 0.01
    uv_distortion: float = 0.02
    chart_seam: float = 0.05
    uv_foldover: float = 0.05
    uv_collision: float = 0.05


@dataclass
class TrainConfig:
    iterations: int = 2000
    learning_rate: float = 0.01
    texture_learning_rate: float = 0.002
    albedo_mode: str = "replacement"
    seed: int = 42
    sh_degree: int = 1
    residual_limit: float = 0.08
    global_shift_limit: float = 0.7
    detail_residual_limit: float = 0.08
    texture_resolution: int = 256
    texture_logit_limit: float = 4.0
    texture_mapping: str = "atlas"
    reference_layout: str = "full"
    reference_saliency_patches: int = 4
    reference_focus_scale: float = 0.32
    reference_tile_count: int = 3
    reference_metric_tiles: bool = False
    style_patch_reference: str = "layout"
    atlas_charts: int = 8
    atlas_neighbours: int = 8
    atlas_padding: int = 4
    atlas_feather: float = 0.15
    atlas_uv_offset_limit: float = 0.03
    atlas_source_layout: str = "packed"
    atlas_reference_repeat: int = 1
    texture_init_strength: float = 0.6
    texture_stage_iterations: int = 1200
    image_size: int = 224
    preview_every: int = 100
    checkpoint_every: int = 500
    graph_anchors: int = 512
    graph_neighbours: int = 8
    graph_scope: str = "appearance"
    min_segment_coverage: float = 0.005
    random_hdr: bool = True
    render_mode: str = "pbr"
    albedo_only_render: bool = False
    random_exposure_min: float = -1.0
    random_exposure_max: float = 1.0
    pbr_diffuse_white: float = 1.0
    pbr_exposure: float = 0.0
    pbr_white_point: float = 1.0


@dataclass
class EvalConfig:
    max_views: int = 12
    heldout_env_count: int = 3


@dataclass
class ExperimentConfig:
    method: str = "ours"
    workspace_root: str = "../.."
    scene: str = ""
    legacy_root: str = "../3DGS-RTMaterial-clean-validation"
    model_path: str = ""
    source_path: str = ""
    images: str = "images"
    resolution: int = -1
    white_background: bool = False
    project_state: str = ""
    segment_id: int = 1
    reference_image: str = ""
    dino_checkpoint: str = ""
    environment_dir: str = ""
    output_dir: str = "outputs/ours"
    train: TrainConfig = field(default_factory=TrainConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def validate(self, require_inputs: bool = True) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.segment_id < 1:
            raise ValueError("segment_id is the one-based GUI segment id and must be >= 1")
        if self.train.sh_degree < 0 or self.train.sh_degree > 3:
            raise ValueError("train.sh_degree must be in [0, 3]")
        if self.train.iterations < 1:
            raise ValueError("train.iterations must be positive")
        if self.train.learning_rate <= 0 or self.train.texture_learning_rate <= 0:
            raise ValueError("training learning rates must be positive")
        if self.train.albedo_mode not in {"replacement", "additive"}:
            raise ValueError("train.albedo_mode must be 'replacement' or 'additive'")
        if self.train.texture_resolution < 16:
            raise ValueError("train.texture_resolution must be at least 16")
        if self.train.texture_logit_limit <= 0:
            raise ValueError("train.texture_logit_limit must be positive")
        if self.train.texture_mapping not in {"planar", "triplanar", "atlas"}:
            raise ValueError("train.texture_mapping must be 'planar', 'triplanar', or 'atlas'")
        if self.train.reference_layout not in {
            "full", "saliency_grid", "saliency_focus", "saliency_tile",
            "saliency_motifs",
        }:
            raise ValueError(
                "train.reference_layout must be 'full', 'saliency_grid', "
                "'saliency_focus', 'saliency_tile', or 'saliency_motifs'"
            )
        if not 1 <= self.train.reference_saliency_patches <= 16:
            raise ValueError("train.reference_saliency_patches must be in [1, 16]")
        if not 0.15 <= self.train.reference_focus_scale <= 1.0:
            raise ValueError("train.reference_focus_scale must be in [0.15, 1.0]")
        if not 1 <= self.train.reference_tile_count <= 8:
            raise ValueError("train.reference_tile_count must be in [1, 8]")
        if self.train.style_patch_reference not in {"original", "layout"}:
            raise ValueError(
                "train.style_patch_reference must be 'original' or 'layout'"
            )
        if self.train.atlas_charts < 1 or self.train.atlas_neighbours < 1:
            raise ValueError("atlas chart and neighbour counts must be positive")
        if self.train.atlas_padding < 0:
            raise ValueError("train.atlas_padding cannot be negative")
        if not 0 <= self.train.atlas_feather <= 0.5:
            raise ValueError("train.atlas_feather must be in [0, 0.5]")
        if not 0 <= self.train.atlas_uv_offset_limit <= 0.25:
            raise ValueError("train.atlas_uv_offset_limit must be in [0, 0.25]")
        if self.train.atlas_source_layout not in {
            "packed", "developed", "component", "chart", "projected",
        }:
            raise ValueError(
                "train.atlas_source_layout must be 'packed', 'developed', "
                "'component', 'chart', or 'projected'"
            )
        if not 1 <= self.train.atlas_reference_repeat <= 16:
            raise ValueError("train.atlas_reference_repeat must be in [1, 16]")
        if self.train.global_shift_limit <= 0:
            raise ValueError("train.global_shift_limit must be positive")
        if self.train.detail_residual_limit < 0:
            raise ValueError("train.detail_residual_limit cannot be negative")
        if self.train.texture_stage_iterations < 0:
            raise ValueError("train.texture_stage_iterations cannot be negative")
        if self.train.graph_scope not in {"appearance", "material"}:
            raise ValueError("train.graph_scope must be 'appearance' or 'material'")
        if self.train.render_mode not in {"pbr", "diffuse_only", "albedo"}:
            raise ValueError(
                "train.render_mode must be 'pbr', 'diffuse_only', or 'albedo'"
            )
        if self.train.pbr_diffuse_white <= 0:
            raise ValueError("train.pbr_diffuse_white must be positive")
        if self.train.pbr_white_point <= 0:
            raise ValueError("train.pbr_white_point must be positive")
        if not 0 <= self.train.min_segment_coverage < 1:
            raise ValueError("train.min_segment_coverage must be in [0, 1)")
        if self.losses.patch_style < 0:
            raise ValueError("losses.patch_style cannot be negative")
        if self.losses.outside < 0 or self.losses.boundary_outside < 0:
            raise ValueError("outside preservation weights cannot be negative")
        if self.losses.hdr_consistency < 0:
            raise ValueError("losses.hdr_consistency cannot be negative")
        if self.losses.texture_anchor < 0 or self.losses.texture_delta_tv < 0:
            raise ValueError("texture regularization weights cannot be negative")
        if self.losses.color_mean < 0:
            raise ValueError("losses.color_mean cannot be negative")
        if self.losses.render_color < 0:
            raise ValueError("losses.render_color cannot be negative")
        for name in (
            "uv_continuity", "uv_distortion", "chart_seam", "uv_foldover",
            "uv_collision",
        ):
            if getattr(self.losses, name) < 0:
                raise ValueError(f"losses.{name} cannot be negative")
        if require_inputs:
            required = {
                "legacy_root": self.legacy_root,
                "model_path": self.model_path,
                "source_path": self.source_path,
                "project_state": self.project_state,
                "reference_image": self.reference_image,
                "dino_checkpoint": self.dino_checkpoint,
            }
            missing = [name for name, value in required.items() if not str(value).strip()]
            if missing:
                raise ValueError("missing required configuration: " + ", ".join(missing))
            for name in (
                "legacy_root", "model_path", "source_path", "project_state",
                "reference_image", "dino_checkpoint",
            ):
                if not Path(getattr(self, name)).exists():
                    raise FileNotFoundError(f"configured {name} does not exist: {getattr(self, name)}")
            if self.environment_dir and not Path(self.environment_dir).is_dir():
                raise FileNotFoundError(
                    f"configured environment_dir is not a directory: {self.environment_dir}"
                )

    def resolved(self, base: Path) -> "ExperimentConfig":
        result = ExperimentConfig.from_dict(asdict(self))
        for name in (
            "workspace_root", "legacy_root", "model_path", "source_path", "project_state",
            "reference_image", "dino_checkpoint", "environment_dir", "output_dir",
        ):
            value = getattr(result, name)
            if value:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = base / path
                setattr(result, name, str(path.resolve()))
        return result

    def with_catalog_scene(self) -> "ExperimentConfig":
        if not self.scene:
            return self
        from .scene_catalog import resolve_scene

        record = resolve_scene(self.workspace_root, self.scene)
        result = ExperimentConfig.from_dict(asdict(self))
        result.model_path = record.model_path
        result.source_path = record.source_path
        result.images = record.images
        result.resolution = record.resolution
        result.white_background = record.white_background
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        values = dict(values)
        values["train"] = TrainConfig(**values.get("train", {}))
        values["losses"] = LossConfig(**values.get("losses", {}))
        values["evaluation"] = EvalConfig(**values.get("evaluation", {}))
        return cls(**values)


def load_config(path: str | Path, require_inputs: bool = True) -> ExperimentConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    config = ExperimentConfig.from_dict(payload).resolved(source.parent).with_catalog_scene()
    config.validate(require_inputs=require_inputs)
    return config
