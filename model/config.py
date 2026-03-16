from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_CONFIG_SCHEMA = 1
ResumeMode = Literal["never", "latest_matching_version", "exact_path"]
CoordSpace = Literal["zero_to_one", "minus_one_to_one"]


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    token = token.strip("._-")
    return token or "unnamed"


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping.")
    return dict(value)


def _require_list(value: Any, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list.")
    return list(value)


def _require_bool(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean.")
    return value


def _require_int(value: Any, *, context: str, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer.")
    if min_value is not None and value < min_value:
        raise ValueError(f"{context} must be >= {min_value}.")
    return value


def _require_float(value: Any, *, context: str, min_value: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric.")
    result = float(value)
    if min_value is not None and result < min_value:
        raise ValueError(f"{context} must be >= {min_value}.")
    return result


def _require_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value


def _optional_string(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, context=context)


def _ensure_no_extra_keys(payload: dict[str, Any], *, context: str, allowed: set[str]) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {context}: {', '.join(unknown)}")


def _pop_required(payload: dict[str, Any], key: str, *, context: str) -> Any:
    if key not in payload:
        raise ValueError(f"Missing required key {context}.{key}")
    return payload.pop(key)


def _pop_optional(payload: dict[str, Any], key: str, default: Any = None) -> Any:
    return payload.pop(key, default)


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _normalize_image_size(value: Any, *, context: str) -> int:
    if isinstance(value, int):
        image_size = value
    elif isinstance(value, (list, tuple)):
        if len(value) != 2 or value[0] != value[1]:
            raise ValueError(f"{context} only supports square image sizes.")
        image_size = _require_int(value[0], context=context, min_value=1)
    else:
        raise ValueError(f"{context} must be an integer or [size, size].")

    if image_size <= 0:
        raise ValueError(f"{context} must be > 0.")
    return image_size


def _apply_override(payload: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Invalid override {override!r}. Expected dotted.path=value.")
    dotted_path, raw_value = override.split("=", 1)
    keys = [part for part in dotted_path.split(".") if part]
    if not keys:
        raise ValueError(f"Invalid override path in {override!r}.")

    value = yaml.safe_load(raw_value)
    target: dict[str, Any] = payload
    traversed: list[str] = []
    for key in keys[:-1]:
        traversed.append(key)
        if key not in target:
            raise ValueError(
                f"Override path {dotted_path!r} is invalid. Missing {'.'.join(traversed)!r}."
            )
        next_value = target[key]
        if not isinstance(next_value, dict):
            raise ValueError(
                f"Override path {dotted_path!r} is invalid. {'.'.join(traversed)!r} is not a mapping."
            )
        target = next_value

    leaf_key = keys[-1]
    if leaf_key not in target:
        raise ValueError(f"Override path {dotted_path!r} is invalid. Missing key {leaf_key!r}.")
    target[leaf_key] = value


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    description: str | None
    tags: tuple[str, ...]


@dataclass(frozen=True)
class PathsConfig:
    checkpoint_root: Path
    split_root: Path
    artifact_root: Path


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    processed_dir: Path
    processed_dir_input: str


@dataclass(frozen=True)
class DataConfig:
    datasets: tuple[DatasetConfig, ...]
    normalized_coords: bool
    coord_space: CoordSpace
    drop_missing: bool
    drop_quality_flagged: bool
    require_finger: bool
    require_shirt: bool
    min_shirt_confidence: float
    min_shirt_sample_count: int


@dataclass(frozen=True)
class SplitsConfig:
    val_fraction: float
    test_fraction: float
    seed: int
    persist: bool
    reuse_existing: bool


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool
    shuffle_train: bool


@dataclass(frozen=True)
class ModelConfig:
    kind: str
    image_size: int
    coord_dim: int
    base_channels: int
    latent_channels: int
    bottleneck_size: int
    latent_dim: int
    posterior_base_channels: int
    prior_hidden_dim: int


@dataclass(frozen=True)
class ResumeConfig:
    mode: ResumeMode
    path: Path | None


@dataclass(frozen=True)
class TrainingSettings:
    device: str
    epochs: int
    lr: float
    amp: bool | Literal["auto"]
    resume: ResumeConfig
    optimizer: Literal["adam", "adamw"]
    weight_decay: float
    grad_accum_steps: int
    grad_clip_norm: float | None
    recon_mse_weight: float
    kl_beta_max: float
    kl_beta_warmup_steps: int
    prior_sample_temperature: float


@dataclass(frozen=True)
class CheckpointingConfig:
    save_final: bool
    save_best: bool
    keep_history: bool
    save_every_n_epochs: int | None
    save_every_n_optimizer_steps: int | None
    filename_template: str


@dataclass(frozen=True)
class ReproducibilityConfig:
    seed: int
    enforce_determinism: bool
    capture_git_state: bool
    save_resolved_config: bool
    save_split_artifact: bool


@dataclass(frozen=True)
class NotebookConfig:
    visualize_dataset_summary: bool
    visualize_split_summary: bool
    visualize_samples: int


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    project: str
    entity: str | None
    mode: str
    group: str | None
    job_type: str
    tags: tuple[str, ...]
    log_code: bool
    log_model_artifact: bool
    log_config_artifact: bool
    validation_sample_count: int
    log_validation_every_n_epochs: int
    validation_prior_sample_count: int
    validation_prior_temperature: float


@dataclass(frozen=True)
class TrainingConfig:
    config_schema: int
    version: str
    version_token: str
    config_path: Path
    repo_root: Path
    config_hash: str
    experiment: ExperimentConfig
    paths: PathsConfig
    data: DataConfig
    splits: SplitsConfig
    loader: LoaderConfig
    model: ModelConfig
    training: TrainingSettings
    checkpointing: CheckpointingConfig
    reproducibility: ReproducibilityConfig
    notebook: NotebookConfig
    wandb: WandbConfig
    resolved_payload: dict[str, Any]

    @property
    def image_size_tuple(self) -> tuple[int, int]:
        return (self.model.image_size, self.model.image_size)

    @property
    def processed_dirs(self) -> list[Path]:
        return [dataset.processed_dir for dataset in self.data.datasets]

    @property
    def split_artifact_path(self) -> Path:
        return self.paths.split_root / f"{self.version_token}_{self.config_hash[:12]}_splits.json"

    @property
    def artifact_dir(self) -> Path:
        return self.paths.artifact_root / f"{self.version_token}_{self.config_hash[:12]}"

    @property
    def resolved_config_path(self) -> Path:
        return self.artifact_dir / "resolved_config.yaml"

    def to_checkpoint_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.resolved_payload))

    def summary_lines(self) -> list[str]:
        dataset_lines = [
            f" - {dataset.name}: {dataset.processed_dir}"
            for dataset in self.data.datasets
        ]
        lines = [
            f"config_path: {self.config_path}",
            f"VERSION: {self.version}",
            f"config_hash: {self.config_hash}",
            f"run_name: {self.experiment.run_name}",
            f"image_size: {self.model.image_size}",
            f"model_kind: {self.model.kind}",
            f"coord_space: {self.data.coord_space}",
            f"latent_dim: {self.model.latent_dim}",
            f"batch_size: {self.loader.batch_size}",
            f"epochs: {self.training.epochs}",
            f"lr: {self.training.lr}",
            f"optimizer: {self.training.optimizer}",
            f"grad_accum_steps: {self.training.grad_accum_steps}",
            f"resume_mode: {self.training.resume.mode}",
            (
                f"wandb: {self.wandb.mode}:{self.wandb.project}"
                if self.wandb.enabled
                else "wandb: disabled"
            ),
            (
                "data_filters: "
                f"drop_missing={self.data.drop_missing}, "
                f"drop_quality_flagged={self.data.drop_quality_flagged}, "
                f"require_finger={self.data.require_finger}, "
                f"require_shirt={self.data.require_shirt}, "
                f"min_shirt_confidence={self.data.min_shirt_confidence}, "
                f"min_shirt_sample_count={self.data.min_shirt_sample_count}"
            ),
            "datasets:",
            *dataset_lines,
        ]
        return lines


def _load_yaml_payload(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"Config file is empty: {config_path}")
    return _require_mapping(payload, context=str(config_path))


def _apply_overrides(payload: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    result = json.loads(json.dumps(payload))
    for override in overrides or []:
        _apply_override(result, override)
    return result


def _validate_schema(payload: dict[str, Any], *, context: str) -> int:
    schema = _require_int(_pop_required(payload, "config_schema", context=context), context=f"{context}.config_schema", min_value=1)
    if schema != CURRENT_CONFIG_SCHEMA:
        raise ValueError(
            f"Unsupported config_schema {schema}. Current supported schema is {CURRENT_CONFIG_SCHEMA}."
        )
    return schema


def _parse_experiment(payload: dict[str, Any], *, context: str) -> ExperimentConfig:
    experiment = _require_mapping(payload, context=context)
    run_name = _require_string(_pop_required(experiment, "run_name", context=context), context=f"{context}.run_name")
    description = _optional_string(_pop_optional(experiment, "description"), context=f"{context}.description")
    tags_payload = _pop_optional(experiment, "tags", [])
    tags = tuple(_require_string(tag, context=f"{context}.tags[]") for tag in _require_list(tags_payload, context=f"{context}.tags"))
    _ensure_no_extra_keys(experiment, context=context, allowed={"run_name", "description", "tags"})
    return ExperimentConfig(run_name=run_name, description=description, tags=tags)


def _parse_paths(payload: dict[str, Any], *, context: str) -> PathsConfig:
    paths = _require_mapping(payload, context=context)
    checkpoint_root = _resolve_repo_path(
        _require_string(_pop_required(paths, "checkpoint_root", context=context), context=f"{context}.checkpoint_root")
    )
    split_root = _resolve_repo_path(
        _require_string(_pop_required(paths, "split_root", context=context), context=f"{context}.split_root")
    )
    artifact_root = _resolve_repo_path(
        _require_string(_pop_required(paths, "artifact_root", context=context), context=f"{context}.artifact_root")
    )
    _ensure_no_extra_keys(paths, context=context, allowed={"checkpoint_root", "split_root", "artifact_root"})
    return PathsConfig(
        checkpoint_root=checkpoint_root,
        split_root=split_root,
        artifact_root=artifact_root,
    )


def _parse_data(payload: dict[str, Any], *, context: str) -> DataConfig:
    data = _require_mapping(payload, context=context)
    datasets_payload = _require_list(_pop_required(data, "datasets", context=context), context=f"{context}.datasets")
    datasets: list[DatasetConfig] = []
    for index, item in enumerate(datasets_payload):
        dataset_context = f"{context}.datasets[{index}]"
        dataset = _require_mapping(item, context=dataset_context)
        name = _require_string(_pop_required(dataset, "name", context=dataset_context), context=f"{dataset_context}.name")
        processed_dir_input = _require_string(
            _pop_required(dataset, "processed_dir", context=dataset_context),
            context=f"{dataset_context}.processed_dir",
        )
        processed_dir = _resolve_repo_path(processed_dir_input)
        if not processed_dir.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {processed_dir}")
        _ensure_no_extra_keys(dataset, context=dataset_context, allowed={"name", "processed_dir"})
        datasets.append(
            DatasetConfig(
                name=name,
                processed_dir=processed_dir,
                processed_dir_input=processed_dir_input,
            )
        )

    normalized_coords = _require_bool(_pop_optional(data, "normalized_coords", True), context=f"{context}.normalized_coords")
    coord_space = _require_string(
        _pop_optional(data, "coord_space", "zero_to_one"),
        context=f"{context}.coord_space",
    )
    if coord_space not in {"zero_to_one", "minus_one_to_one"}:
        raise ValueError(f"{context}.coord_space must be zero_to_one or minus_one_to_one.")
    drop_missing = _require_bool(_pop_optional(data, "drop_missing", True), context=f"{context}.drop_missing")
    drop_quality_flagged = _require_bool(
        _pop_optional(data, "drop_quality_flagged", True),
        context=f"{context}.drop_quality_flagged",
    )
    require_finger = _require_bool(_pop_optional(data, "require_finger", False), context=f"{context}.require_finger")
    require_shirt = _require_bool(_pop_optional(data, "require_shirt", False), context=f"{context}.require_shirt")
    min_shirt_confidence = _require_float(
        _pop_optional(data, "min_shirt_confidence", 0.0),
        context=f"{context}.min_shirt_confidence",
        min_value=0.0,
    )
    min_shirt_sample_count = _require_int(
        _pop_optional(data, "min_shirt_sample_count", 1),
        context=f"{context}.min_shirt_sample_count",
        min_value=1,
    )
    _ensure_no_extra_keys(
        data,
        context=context,
        allowed={
            "datasets",
            "normalized_coords",
            "coord_space",
            "drop_missing",
            "drop_quality_flagged",
            "require_finger",
            "require_shirt",
            "min_shirt_confidence",
            "min_shirt_sample_count",
        },
    )
    return DataConfig(
        datasets=tuple(datasets),
        normalized_coords=normalized_coords,
        coord_space=coord_space,
        drop_missing=drop_missing,
        drop_quality_flagged=drop_quality_flagged,
        require_finger=require_finger,
        require_shirt=require_shirt,
        min_shirt_confidence=min_shirt_confidence,
        min_shirt_sample_count=min_shirt_sample_count,
    )


def _parse_splits(payload: dict[str, Any], *, context: str) -> SplitsConfig:
    splits = _require_mapping(payload, context=context)
    val_fraction = _require_float(_pop_optional(splits, "val_fraction", 0.1), context=f"{context}.val_fraction", min_value=0.0)
    test_fraction = _require_float(_pop_optional(splits, "test_fraction", 0.1), context=f"{context}.test_fraction", min_value=0.0)
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("splits.val_fraction + splits.test_fraction must be < 1.0")
    seed = _require_int(_pop_optional(splits, "seed", 42), context=f"{context}.seed")
    persist = _require_bool(_pop_optional(splits, "persist", True), context=f"{context}.persist")
    reuse_existing = _require_bool(
        _pop_optional(splits, "reuse_existing", True),
        context=f"{context}.reuse_existing",
    )
    _ensure_no_extra_keys(splits, context=context, allowed={"val_fraction", "test_fraction", "seed", "persist", "reuse_existing"})
    return SplitsConfig(
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        persist=persist,
        reuse_existing=reuse_existing,
    )


def _parse_loader(payload: dict[str, Any], *, context: str) -> LoaderConfig:
    loader = _require_mapping(payload, context=context)
    batch_size = _require_int(_pop_optional(loader, "batch_size", 8), context=f"{context}.batch_size", min_value=1)
    num_workers = _require_int(_pop_optional(loader, "num_workers", 0), context=f"{context}.num_workers", min_value=0)
    pin_memory = _require_bool(_pop_optional(loader, "pin_memory", True), context=f"{context}.pin_memory")
    shuffle_train = _require_bool(_pop_optional(loader, "shuffle_train", True), context=f"{context}.shuffle_train")
    _ensure_no_extra_keys(loader, context=context, allowed={"batch_size", "num_workers", "pin_memory", "shuffle_train"})
    return LoaderConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle_train=shuffle_train,
    )


def _parse_model(payload: dict[str, Any], *, context: str) -> ModelConfig:
    model = _require_mapping(payload, context=context)
    kind = _require_string(_pop_optional(model, "kind", "coord_to_image_unet"), context=f"{context}.kind")
    if kind not in {"coord_to_image_unet", "pointing_cvae"}:
        raise ValueError(f"{context}.kind must be coord_to_image_unet or pointing_cvae.")
    image_size = _normalize_image_size(_pop_optional(model, "image_size", 128), context=f"{context}.image_size")
    coord_dim = _require_int(_pop_optional(model, "coord_dim", 2), context=f"{context}.coord_dim", min_value=1)
    base_channels = _require_int(_pop_optional(model, "base_channels", 32), context=f"{context}.base_channels", min_value=1)
    latent_channels = _require_int(_pop_optional(model, "latent_channels", 256), context=f"{context}.latent_channels", min_value=1)
    bottleneck_size = _require_int(_pop_optional(model, "bottleneck_size", 8), context=f"{context}.bottleneck_size", min_value=1)
    latent_dim = _require_int(_pop_optional(model, "latent_dim", 32), context=f"{context}.latent_dim", min_value=1)
    posterior_base_channels = _require_int(
        _pop_optional(model, "posterior_base_channels", 32),
        context=f"{context}.posterior_base_channels",
        min_value=1,
    )
    prior_hidden_dim = _require_int(
        _pop_optional(model, "prior_hidden_dim", 128),
        context=f"{context}.prior_hidden_dim",
        min_value=1,
    )
    _ensure_no_extra_keys(
        model,
        context=context,
        allowed={
            "kind",
            "image_size",
            "coord_dim",
            "base_channels",
            "latent_channels",
            "bottleneck_size",
            "latent_dim",
            "posterior_base_channels",
            "prior_hidden_dim",
        },
    )
    return ModelConfig(
        kind=kind,
        image_size=image_size,
        coord_dim=coord_dim,
        base_channels=base_channels,
        latent_channels=latent_channels,
        bottleneck_size=bottleneck_size,
        latent_dim=latent_dim,
        posterior_base_channels=posterior_base_channels,
        prior_hidden_dim=prior_hidden_dim,
    )


def _parse_resume(payload: Any, *, context: str) -> ResumeConfig:
    resume = _require_mapping(payload, context=context)
    mode = _require_string(_pop_optional(resume, "mode", "never"), context=f"{context}.mode")
    if mode not in {"never", "latest_matching_version", "exact_path"}:
        raise ValueError(f"{context}.mode must be one of never, latest_matching_version, exact_path.")
    path_input = _optional_string(_pop_optional(resume, "path"), context=f"{context}.path")
    path = _resolve_repo_path(path_input) if path_input is not None else None
    if mode == "exact_path" and path is None:
        raise ValueError(f"{context}.path is required when mode is exact_path.")
    _ensure_no_extra_keys(resume, context=context, allowed={"mode", "path"})
    return ResumeConfig(mode=mode, path=path)


def _parse_training(payload: dict[str, Any], *, context: str) -> TrainingSettings:
    training = _require_mapping(payload, context=context)
    device = _require_string(_pop_optional(training, "device", "auto"), context=f"{context}.device")
    epochs = _require_int(_pop_optional(training, "epochs", 20), context=f"{context}.epochs", min_value=1)
    lr = _require_float(_pop_optional(training, "lr", 1e-3), context=f"{context}.lr", min_value=0.0)
    amp_value = _pop_optional(training, "amp", "auto")
    if amp_value != "auto":
        amp_value = _require_bool(amp_value, context=f"{context}.amp")
    resume = _parse_resume(_pop_optional(training, "resume", {"mode": "never"}), context=f"{context}.resume")
    optimizer = _require_string(_pop_optional(training, "optimizer", "adam"), context=f"{context}.optimizer")
    if optimizer not in {"adam", "adamw"}:
        raise ValueError(f"{context}.optimizer must be adam or adamw.")
    weight_decay = _require_float(
        _pop_optional(training, "weight_decay", 0.0),
        context=f"{context}.weight_decay",
        min_value=0.0,
    )
    grad_accum_steps = _require_int(
        _pop_optional(training, "grad_accum_steps", 1),
        context=f"{context}.grad_accum_steps",
        min_value=1,
    )
    raw_grad_clip_norm = _pop_optional(training, "grad_clip_norm", None)
    grad_clip_norm = (
        None
        if raw_grad_clip_norm is None
        else _require_float(raw_grad_clip_norm, context=f"{context}.grad_clip_norm", min_value=0.0)
    )
    recon_mse_weight = _require_float(
        _pop_optional(training, "recon_mse_weight", 0.25),
        context=f"{context}.recon_mse_weight",
        min_value=0.0,
    )
    kl_beta_max = _require_float(
        _pop_optional(training, "kl_beta_max", 1e-3),
        context=f"{context}.kl_beta_max",
        min_value=0.0,
    )
    kl_beta_warmup_steps = _require_int(
        _pop_optional(training, "kl_beta_warmup_steps", 5000),
        context=f"{context}.kl_beta_warmup_steps",
        min_value=0,
    )
    prior_sample_temperature = _require_float(
        _pop_optional(training, "prior_sample_temperature", 0.7),
        context=f"{context}.prior_sample_temperature",
        min_value=0.0,
    )
    _ensure_no_extra_keys(
        training,
        context=context,
        allowed={
            "device",
            "epochs",
            "lr",
            "amp",
            "resume",
            "optimizer",
            "weight_decay",
            "grad_accum_steps",
            "grad_clip_norm",
            "recon_mse_weight",
            "kl_beta_max",
            "kl_beta_warmup_steps",
            "prior_sample_temperature",
        },
    )
    return TrainingSettings(
        device=device,
        epochs=epochs,
        lr=lr,
        amp=amp_value,
        resume=resume,
        optimizer=optimizer,
        weight_decay=weight_decay,
        grad_accum_steps=grad_accum_steps,
        grad_clip_norm=grad_clip_norm,
        recon_mse_weight=recon_mse_weight,
        kl_beta_max=kl_beta_max,
        kl_beta_warmup_steps=kl_beta_warmup_steps,
        prior_sample_temperature=prior_sample_temperature,
    )


def _parse_checkpointing(payload: dict[str, Any], *, context: str) -> CheckpointingConfig:
    checkpointing = _require_mapping(payload, context=context)
    save_final = _require_bool(_pop_optional(checkpointing, "save_final", True), context=f"{context}.save_final")
    save_best = _require_bool(_pop_optional(checkpointing, "save_best", False), context=f"{context}.save_best")
    keep_history = _require_bool(_pop_optional(checkpointing, "keep_history", True), context=f"{context}.keep_history")
    raw_save_every_n_epochs = _pop_optional(checkpointing, "save_every_n_epochs", None)
    save_every_n_epochs = (
        None
        if raw_save_every_n_epochs is None
        else _require_int(
            raw_save_every_n_epochs,
            context=f"{context}.save_every_n_epochs",
            min_value=1,
        )
    )
    raw_save_every_n_optimizer_steps = _pop_optional(checkpointing, "save_every_n_optimizer_steps", None)
    save_every_n_optimizer_steps = (
        None
        if raw_save_every_n_optimizer_steps is None
        else _require_int(
            raw_save_every_n_optimizer_steps,
            context=f"{context}.save_every_n_optimizer_steps",
            min_value=1,
        )
    )
    filename_template = _require_string(
        _pop_optional(checkpointing, "filename_template", "{checkpoint_id}_{VERSION}_{run_name}_{timestamp}.pt"),
        context=f"{context}.filename_template",
    )
    _ensure_no_extra_keys(
        checkpointing,
        context=context,
        allowed={
            "save_final",
            "save_best",
            "keep_history",
            "save_every_n_epochs",
            "save_every_n_optimizer_steps",
            "filename_template",
        },
    )
    return CheckpointingConfig(
        save_final=save_final,
        save_best=save_best,
        keep_history=keep_history,
        save_every_n_epochs=save_every_n_epochs,
        save_every_n_optimizer_steps=save_every_n_optimizer_steps,
        filename_template=filename_template,
    )


def _parse_reproducibility(payload: dict[str, Any], *, context: str) -> ReproducibilityConfig:
    reproducibility = _require_mapping(payload, context=context)
    seed = _require_int(_pop_optional(reproducibility, "seed", 42), context=f"{context}.seed")
    enforce_determinism = _require_bool(
        _pop_optional(reproducibility, "enforce_determinism", False),
        context=f"{context}.enforce_determinism",
    )
    capture_git_state = _require_bool(
        _pop_optional(reproducibility, "capture_git_state", True),
        context=f"{context}.capture_git_state",
    )
    save_resolved_config = _require_bool(
        _pop_optional(reproducibility, "save_resolved_config", True),
        context=f"{context}.save_resolved_config",
    )
    save_split_artifact = _require_bool(
        _pop_optional(reproducibility, "save_split_artifact", True),
        context=f"{context}.save_split_artifact",
    )
    _ensure_no_extra_keys(
        reproducibility,
        context=context,
        allowed={"seed", "enforce_determinism", "capture_git_state", "save_resolved_config", "save_split_artifact"},
    )
    return ReproducibilityConfig(
        seed=seed,
        enforce_determinism=enforce_determinism,
        capture_git_state=capture_git_state,
        save_resolved_config=save_resolved_config,
        save_split_artifact=save_split_artifact,
    )


def _parse_notebook(payload: dict[str, Any], *, context: str) -> NotebookConfig:
    notebook = _require_mapping(payload, context=context)
    visualize_dataset_summary = _require_bool(
        _pop_optional(notebook, "visualize_dataset_summary", True),
        context=f"{context}.visualize_dataset_summary",
    )
    visualize_split_summary = _require_bool(
        _pop_optional(notebook, "visualize_split_summary", True),
        context=f"{context}.visualize_split_summary",
    )
    visualize_samples = _require_int(
        _pop_optional(notebook, "visualize_samples", 8),
        context=f"{context}.visualize_samples",
        min_value=1,
    )
    _ensure_no_extra_keys(
        notebook,
        context=context,
        allowed={"visualize_dataset_summary", "visualize_split_summary", "visualize_samples"},
    )
    return NotebookConfig(
        visualize_dataset_summary=visualize_dataset_summary,
        visualize_split_summary=visualize_split_summary,
        visualize_samples=visualize_samples,
    )


def _parse_wandb(payload: dict[str, Any], *, context: str) -> WandbConfig:
    wandb_payload = _require_mapping(payload, context=context)
    enabled = _require_bool(_pop_optional(wandb_payload, "enabled", True), context=f"{context}.enabled")
    project = _require_string(_pop_optional(wandb_payload, "project", "worldmodel"), context=f"{context}.project")
    entity = _optional_string(_pop_optional(wandb_payload, "entity"), context=f"{context}.entity")
    mode = _require_string(_pop_optional(wandb_payload, "mode", "online"), context=f"{context}.mode")
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError(f"{context}.mode must be one of online, offline, disabled.")
    group = _optional_string(_pop_optional(wandb_payload, "group"), context=f"{context}.group")
    job_type = _require_string(_pop_optional(wandb_payload, "job_type", "train"), context=f"{context}.job_type")
    tags_payload = _pop_optional(wandb_payload, "tags", [])
    tags = tuple(_require_string(tag, context=f"{context}.tags[]") for tag in _require_list(tags_payload, context=f"{context}.tags"))
    log_code = _require_bool(_pop_optional(wandb_payload, "log_code", True), context=f"{context}.log_code")
    log_model_artifact = _require_bool(
        _pop_optional(wandb_payload, "log_model_artifact", True),
        context=f"{context}.log_model_artifact",
    )
    log_config_artifact = _require_bool(
        _pop_optional(wandb_payload, "log_config_artifact", True),
        context=f"{context}.log_config_artifact",
    )
    validation_sample_count = _require_int(
        _pop_optional(wandb_payload, "validation_sample_count", 8),
        context=f"{context}.validation_sample_count",
        min_value=0,
    )
    log_validation_every_n_epochs = _require_int(
        _pop_optional(wandb_payload, "log_validation_every_n_epochs", 1),
        context=f"{context}.log_validation_every_n_epochs",
        min_value=1,
    )
    validation_prior_sample_count = _require_int(
        _pop_optional(wandb_payload, "validation_prior_sample_count", 4),
        context=f"{context}.validation_prior_sample_count",
        min_value=0,
    )
    validation_prior_temperature = _require_float(
        _pop_optional(wandb_payload, "validation_prior_temperature", 0.7),
        context=f"{context}.validation_prior_temperature",
        min_value=0.0,
    )
    _ensure_no_extra_keys(
        wandb_payload,
        context=context,
        allowed={
            "enabled",
            "project",
            "entity",
            "mode",
            "group",
            "job_type",
            "tags",
            "log_code",
            "log_model_artifact",
            "log_config_artifact",
            "validation_sample_count",
            "log_validation_every_n_epochs",
            "validation_prior_sample_count",
            "validation_prior_temperature",
        },
    )
    return WandbConfig(
        enabled=enabled,
        project=project,
        entity=entity,
        mode=mode,
        group=group,
        job_type=job_type,
        tags=tags,
        log_code=log_code,
        log_model_artifact=log_model_artifact,
        log_config_artifact=log_config_artifact,
        validation_sample_count=validation_sample_count,
        log_validation_every_n_epochs=log_validation_every_n_epochs,
        validation_prior_sample_count=validation_prior_sample_count,
        validation_prior_temperature=validation_prior_temperature,
    )


def load_training_config(
    config_path: str | Path,
    *,
    overrides: list[str] | None = None,
) -> TrainingConfig:
    config_path = Path(config_path).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve() if (REPO_ROOT / config_path).exists() else config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    payload = _apply_overrides(_load_yaml_payload(config_path), overrides)
    root_context = f"config({config_path.name})"
    root = dict(payload)

    config_schema = _validate_schema(root, context=root_context)
    version = _require_string(_pop_required(root, "VERSION", context=root_context), context=f"{root_context}.VERSION")
    experiment = _parse_experiment(_pop_required(root, "experiment", context=root_context), context=f"{root_context}.experiment")
    paths = _parse_paths(_pop_required(root, "paths", context=root_context), context=f"{root_context}.paths")
    data = _parse_data(_pop_required(root, "data", context=root_context), context=f"{root_context}.data")
    splits = _parse_splits(_pop_required(root, "splits", context=root_context), context=f"{root_context}.splits")
    loader = _parse_loader(_pop_required(root, "loader", context=root_context), context=f"{root_context}.loader")
    model = _parse_model(_pop_required(root, "model", context=root_context), context=f"{root_context}.model")
    training = _parse_training(_pop_required(root, "training", context=root_context), context=f"{root_context}.training")
    checkpointing = _parse_checkpointing(
        _pop_required(root, "checkpointing", context=root_context),
        context=f"{root_context}.checkpointing",
    )
    reproducibility = _parse_reproducibility(
        _pop_required(root, "reproducibility", context=root_context),
        context=f"{root_context}.reproducibility",
    )
    notebook = _parse_notebook(_pop_required(root, "notebook", context=root_context), context=f"{root_context}.notebook")
    wandb_config = _parse_wandb(_pop_required(root, "wandb", context=root_context), context=f"{root_context}.wandb")
    if model.kind == "pointing_cvae" and not data.normalized_coords:
        raise ValueError("pointing_cvae requires data.normalized_coords=true.")
    _ensure_no_extra_keys(
        root,
        context=root_context,
        allowed={
            "config_schema",
            "VERSION",
            "experiment",
            "paths",
            "data",
            "splits",
            "loader",
            "model",
            "training",
            "checkpointing",
            "reproducibility",
            "notebook",
            "wandb",
        },
    )

    resolved_payload = {
        "config_schema": config_schema,
        "VERSION": version,
        "config_path": str(config_path),
        "repo_root": str(REPO_ROOT),
        "experiment": {
            "run_name": experiment.run_name,
            "description": experiment.description,
            "tags": list(experiment.tags),
        },
        "paths": {
            "checkpoint_root": str(paths.checkpoint_root),
            "split_root": str(paths.split_root),
            "artifact_root": str(paths.artifact_root),
        },
        "data": {
            "datasets": [
                {
                    "name": dataset.name,
                    "processed_dir": dataset.processed_dir_input,
                    "resolved_processed_dir": str(dataset.processed_dir),
                }
                for dataset in data.datasets
            ],
            "normalized_coords": data.normalized_coords,
            "coord_space": data.coord_space,
            "drop_missing": data.drop_missing,
            "drop_quality_flagged": data.drop_quality_flagged,
            "require_finger": data.require_finger,
            "require_shirt": data.require_shirt,
            "min_shirt_confidence": data.min_shirt_confidence,
            "min_shirt_sample_count": data.min_shirt_sample_count,
        },
        "splits": {
            "val_fraction": splits.val_fraction,
            "test_fraction": splits.test_fraction,
            "seed": splits.seed,
            "persist": splits.persist,
            "reuse_existing": splits.reuse_existing,
        },
        "loader": {
            "batch_size": loader.batch_size,
            "num_workers": loader.num_workers,
            "pin_memory": loader.pin_memory,
            "shuffle_train": loader.shuffle_train,
        },
        "model": {
            "kind": model.kind,
            "image_size": model.image_size,
            "coord_dim": model.coord_dim,
            "base_channels": model.base_channels,
            "latent_channels": model.latent_channels,
            "bottleneck_size": model.bottleneck_size,
            "latent_dim": model.latent_dim,
            "posterior_base_channels": model.posterior_base_channels,
            "prior_hidden_dim": model.prior_hidden_dim,
        },
        "training": {
            "device": training.device,
            "epochs": training.epochs,
            "lr": training.lr,
            "amp": training.amp,
            "optimizer": training.optimizer,
            "weight_decay": training.weight_decay,
            "grad_accum_steps": training.grad_accum_steps,
            "grad_clip_norm": training.grad_clip_norm,
            "recon_mse_weight": training.recon_mse_weight,
            "kl_beta_max": training.kl_beta_max,
            "kl_beta_warmup_steps": training.kl_beta_warmup_steps,
            "prior_sample_temperature": training.prior_sample_temperature,
            "resume": {
                "mode": training.resume.mode,
                "path": str(training.resume.path) if training.resume.path is not None else None,
            },
        },
        "checkpointing": {
            "save_final": checkpointing.save_final,
            "save_best": checkpointing.save_best,
            "keep_history": checkpointing.keep_history,
            "save_every_n_epochs": checkpointing.save_every_n_epochs,
            "save_every_n_optimizer_steps": checkpointing.save_every_n_optimizer_steps,
            "filename_template": checkpointing.filename_template,
        },
        "reproducibility": {
            "seed": reproducibility.seed,
            "enforce_determinism": reproducibility.enforce_determinism,
            "capture_git_state": reproducibility.capture_git_state,
            "save_resolved_config": reproducibility.save_resolved_config,
            "save_split_artifact": reproducibility.save_split_artifact,
        },
        "notebook": {
            "visualize_dataset_summary": notebook.visualize_dataset_summary,
            "visualize_split_summary": notebook.visualize_split_summary,
            "visualize_samples": notebook.visualize_samples,
        },
        "wandb": {
            "enabled": wandb_config.enabled,
            "project": wandb_config.project,
            "entity": wandb_config.entity,
            "mode": wandb_config.mode,
            "group": wandb_config.group,
            "job_type": wandb_config.job_type,
            "tags": list(wandb_config.tags),
            "log_code": wandb_config.log_code,
            "log_model_artifact": wandb_config.log_model_artifact,
            "log_config_artifact": wandb_config.log_config_artifact,
            "validation_sample_count": wandb_config.validation_sample_count,
            "log_validation_every_n_epochs": wandb_config.log_validation_every_n_epochs,
            "validation_prior_sample_count": wandb_config.validation_prior_sample_count,
            "validation_prior_temperature": wandb_config.validation_prior_temperature,
        },
    }
    hash_payload = {
        "config_schema": config_schema,
        "VERSION": version,
        "experiment": payload["experiment"],
        "paths": payload["paths"],
        "data": payload["data"],
        "splits": payload["splits"],
        "loader": payload["loader"],
        "model": payload["model"],
        "training": payload["training"],
        "checkpointing": payload["checkpointing"],
        "reproducibility": payload["reproducibility"],
        "notebook": payload["notebook"],
        "wandb": payload["wandb"],
    }
    config_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return TrainingConfig(
        config_schema=config_schema,
        version=version,
        version_token=_safe_token(version),
        config_path=config_path,
        repo_root=REPO_ROOT,
        config_hash=config_hash,
        experiment=experiment,
        paths=paths,
        data=data,
        splits=splits,
        loader=loader,
        model=model,
        training=training,
        checkpointing=checkpointing,
        reproducibility=reproducibility,
        notebook=notebook,
        wandb=wandb_config,
        resolved_payload=resolved_payload,
    )


def save_resolved_config(config: TrainingConfig) -> Path:
    config.resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_checkpoint_dict()
    payload["config_hash"] = config.config_hash
    config.resolved_config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return config.resolved_config_path
