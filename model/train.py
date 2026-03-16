from __future__ import annotations

import json
import math
import random
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset

from model.checkpoints import (
    checkpoint_glob_pattern,
    latest_checkpoint,
    list_checkpoints,
    make_checkpoint_path,
)
from model.config import TrainingConfig, save_resolved_config
from model.cvae import PointingCVAE, clamp_logvar, kl_beta, kl_diag_gaussians, reparameterize
from model.load import FingerVideoDataset
from model.model import CoordinateToImageUNet
from model.wandb_utils import TrainingWandbTracker


@dataclass
class TrainingHistory:
    train_losses: list[float]
    val_losses: list[float]
    test_losses: list[float]


@dataclass
class TrainingLoopArtifacts:
    history: TrainingHistory
    best_state_dict: Optional[dict[str, Tensor]]
    best_epoch: Optional[int]
    best_val_loss: Optional[float]


@dataclass
class TrainingRunResult:
    model: nn.Module
    history: TrainingHistory
    device: str
    train_loader: DataLoader[tuple[Tensor, Tensor]]
    val_loader: DataLoader[tuple[Tensor, Tensor]]
    test_loader: Optional[DataLoader[tuple[Tensor, Tensor]]]
    dataset: FingerVideoDataset
    final_checkpoint_path: Optional[Path]
    best_checkpoint_path: Optional[Path]
    resumed_from_checkpoint: Optional[Path]
    split_artifact_path: Optional[Path]
    resolved_config_path: Optional[Path]
    git_state: Optional[dict[str, Any]]
    wandb_run_id: Optional[str]
    wandb_run_name: Optional[str]
    wandb_run_url: Optional[str]


def reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    mse_weight: float = 0.25,
) -> Tensor:
    l1 = F.l1_loss(prediction, target)
    mse = F.mse_loss(prediction, target)
    return l1 + mse_weight * mse


def _clone_state_dict_to_cpu(model_state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model_state_dict.items()}


def _epoch_token(epoch: int | None) -> str:
    return f"epoch{epoch:04d}" if epoch is not None else "epochunknown"


def _optimizer_step_token(optimizer_step: int | None) -> str:
    return f"step{optimizer_step:06d}" if optimizer_step is not None else "nostep"


def _set_reproducibility(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _resolve_amp(enabled: bool | str, *, device: str) -> bool:
    if enabled == "auto":
        return device.startswith("cuda")
    return bool(enabled) and device.startswith("cuda")


def _capture_git_state(repo_root: Path) -> Optional[dict[str, Any]]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip().splitlines()
    except Exception:
        return None
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_short": status,
    }


def _create_index_splits(
    dataset_size: int,
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    test_size = int(dataset_size * test_fraction)
    val_size = int(dataset_size * val_fraction)
    if test_fraction > 0 and test_size == 0:
        test_size = 1
    if val_fraction > 0 and val_size == 0:
        val_size = 1
    train_size = dataset_size - val_size - test_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested train/val/test split.")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(dataset_size, generator=generator).tolist()
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size : train_size + val_size]
    test_indices = permutation[train_size + val_size :]
    return train_indices, val_indices, test_indices


def _split_payload(
    dataset: FingerVideoDataset,
    config: TrainingConfig,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
) -> dict[str, Any]:
    return {
        "split_format_version": 1,
        "config_path": str(config.config_path),
        "config_hash": config.config_hash,
        "VERSION": config.version,
        "dataset_size": len(dataset),
        "splits": {
            "train": [dataset.sample_identifier(index) for index in train_indices],
            "val": [dataset.sample_identifier(index) for index in val_indices],
            "test": [dataset.sample_identifier(index) for index in test_indices],
        },
        "counts": {
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices),
        },
    }


def _load_split_indices(
    dataset: FingerVideoDataset,
    split_artifact_path: Path,
    *,
    expected_config_hash: str,
) -> tuple[list[int], list[int], list[int]]:
    payload = json.loads(split_artifact_path.read_text(encoding="utf-8"))
    if payload.get("config_hash") != expected_config_hash:
        raise ValueError(
            f"Split artifact {split_artifact_path} was created for a different config hash."
        )

    sample_index_by_id = {
        dataset.sample_identifier(index): index for index in range(len(dataset))
    }

    resolved_splits: list[list[int]] = []
    for split_name in ("train", "val", "test"):
        sample_ids = payload.get("splits", {}).get(split_name)
        if not isinstance(sample_ids, list):
            raise ValueError(f"Split artifact is missing splits.{split_name}")
        indices: list[int] = []
        missing_ids: list[str] = []
        for sample_id in sample_ids:
            index = sample_index_by_id.get(sample_id)
            if index is None:
                missing_ids.append(sample_id)
            else:
                indices.append(index)
        if missing_ids:
            preview = ", ".join(missing_ids[:3])
            raise ValueError(
                f"Split artifact {split_artifact_path} references samples not present in the current dataset: {preview}"
            )
        resolved_splits.append(indices)

    return resolved_splits[0], resolved_splits[1], resolved_splits[2]


def _resolve_dataset_splits(
    dataset: FingerVideoDataset,
    config: TrainingConfig,
) -> tuple[list[int], list[int], list[int], Optional[Path]]:
    split_artifact_path = config.split_artifact_path
    if (
        config.splits.persist
        and config.splits.reuse_existing
        and split_artifact_path.exists()
    ):
        train_indices, val_indices, test_indices = _load_split_indices(
            dataset,
            split_artifact_path,
            expected_config_hash=config.config_hash,
        )
        return train_indices, val_indices, test_indices, split_artifact_path

    train_indices, val_indices, test_indices = _create_index_splits(
        len(dataset),
        val_fraction=config.splits.val_fraction,
        test_fraction=config.splits.test_fraction,
        seed=config.splits.seed,
    )

    if config.splits.persist and config.reproducibility.save_split_artifact:
        split_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _split_payload(dataset, config, train_indices, val_indices, test_indices)
        split_artifact_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return train_indices, val_indices, test_indices, split_artifact_path

    return train_indices, val_indices, test_indices, None


def _make_loader(
    dataset: Dataset[tuple[Tensor, Tensor]],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[tuple[Tensor, Tensor]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_dataset_from_config(config: TrainingConfig) -> FingerVideoDataset:
    return FingerVideoDataset(
        processed_dir=config.processed_dirs,
        image_size=config.image_size_tuple,
        normalized_coords=config.data.normalized_coords,
        coord_space=config.data.coord_space,
        drop_missing=config.data.drop_missing,
        drop_quality_flagged=config.data.drop_quality_flagged,
        require_finger=config.data.require_finger,
        require_shirt=config.data.require_shirt,
        min_shirt_confidence=config.data.min_shirt_confidence,
        min_shirt_sample_count=config.data.min_shirt_sample_count,
    )


def build_model_from_config(config: TrainingConfig) -> nn.Module:
    if config.model.kind == "coord_to_image_unet":
        return CoordinateToImageUNet(
            image_size=config.model.image_size,
            coord_dim=config.model.coord_dim,
            base_channels=config.model.base_channels,
            latent_channels=config.model.latent_channels,
            bottleneck_size=config.model.bottleneck_size,
        )
    if config.model.kind == "pointing_cvae":
        return PointingCVAE(
            image_size=config.model.image_size,
            coord_dim=config.model.coord_dim,
            latent_dim=config.model.latent_dim,
            base_channels=config.model.base_channels,
            latent_channels=config.model.latent_channels,
            bottleneck_size=config.model.bottleneck_size,
            posterior_base_channels=config.model.posterior_base_channels,
            prior_hidden_dim=config.model.prior_hidden_dim,
        )
    raise ValueError(f"Unsupported model kind: {config.model.kind}")


def make_train_val_test_loaders(
    processed_dir: str | Path | list[str | Path] = "data",
    *,
    image_size: tuple[int, int] = (128, 128),
    batch_size: int = 16,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    normalized_coords: bool = True,
    coord_space: str = "zero_to_one",
    drop_missing: bool = True,
    drop_quality_flagged: bool = True,
    require_finger: bool = False,
    require_shirt: bool = False,
    min_shirt_confidence: float = 0.0,
    min_shirt_sample_count: int = 1,
    pin_memory: bool = True,
    glob_pattern: str = "processed-finger*",
) -> tuple[
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
]:
    dataset = FingerVideoDataset(
        processed_dir=processed_dir,
        image_size=image_size,
        normalized_coords=normalized_coords,
        coord_space=coord_space,
        drop_missing=drop_missing,
        drop_quality_flagged=drop_quality_flagged,
        require_finger=require_finger,
        require_shirt=require_shirt,
        min_shirt_confidence=min_shirt_confidence,
        min_shirt_sample_count=min_shirt_sample_count,
        glob_pattern=glob_pattern,
    )
    train_indices, val_indices, test_indices = _create_index_splits(
        len(dataset),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    train_loader = _make_loader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = _make_loader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = _make_loader(
        Subset(dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def make_train_val_loaders(
    processed_dir: str | Path | list[str | Path] = "data",
    *,
    image_size: tuple[int, int] = (128, 128),
    batch_size: int = 16,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    normalized_coords: bool = True,
    coord_space: str = "zero_to_one",
    drop_missing: bool = True,
    drop_quality_flagged: bool = True,
    require_finger: bool = False,
    require_shirt: bool = False,
    min_shirt_confidence: float = 0.0,
    min_shirt_sample_count: int = 1,
    pin_memory: bool = True,
    glob_pattern: str = "processed-finger*",
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    train_loader, val_loader, _ = make_train_val_test_loaders(
        processed_dir=processed_dir,
        image_size=image_size,
        batch_size=batch_size,
        val_fraction=val_fraction,
        test_fraction=0.0,
        seed=seed,
        num_workers=num_workers,
        normalized_coords=normalized_coords,
        coord_space=coord_space,
        drop_missing=drop_missing,
        drop_quality_flagged=drop_quality_flagged,
        require_finger=require_finger,
        require_shirt=require_shirt,
        min_shirt_confidence=min_shirt_confidence,
        min_shirt_sample_count=min_shirt_sample_count,
        pin_memory=pin_memory,
        glob_pattern=glob_pattern,
    )
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    *,
    device: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_examples = 0

    for coords, frames in loader:
        coords = coords.to(device, non_blocking=True)
        frames = frames.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with autocast(device_type="cuda", enabled=use_amp):
                predictions = model(coords)
                loss = reconstruction_loss(predictions, frames)

        if training:
            optimizer.zero_grad(set_to_none=True)
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = coords.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size

    return total_loss / max(1, total_examples)


def _build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    optimizer_cls = (
        torch.optim.AdamW if config.training.optimizer == "adamw" else torch.optim.Adam
    )
    return optimizer_cls(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )


def _compute_configured_batch_metrics(
    model: nn.Module,
    coords: Tensor,
    frames: Tensor,
    *,
    config: TrainingConfig,
    optimizer_step: int,
    training: bool,
) -> tuple[Tensor, dict[str, float]]:
    if config.model.kind == "pointing_cvae":
        assert isinstance(model, PointingCVAE)
        mu_post, logvar_post = model.posterior(frames)
        mu_prior, logvar_prior = model.prior(coords)
        z = reparameterize(mu_post, logvar_post) if training else mu_post
        predictions = model.decode(coords, z)
        recon_l1 = F.l1_loss(predictions, frames)
        recon_mse = F.mse_loss(predictions, frames)
        recon_total = recon_l1 + config.training.recon_mse_weight * recon_mse
        kl = kl_diag_gaussians(
            mu_q=mu_post,
            logvar_q=logvar_post,
            mu_p=mu_prior,
            logvar_p=logvar_prior,
        ).mean()
        beta = kl_beta(
            optimizer_step,
            warmup_steps=config.training.kl_beta_warmup_steps,
            max_beta=config.training.kl_beta_max,
        )
        loss = recon_total + beta * kl
        stats = {
            "loss": float(loss.detach().cpu()),
            "recon_total": float(recon_total.detach().cpu()),
            "recon_l1": float(recon_l1.detach().cpu()),
            "recon_mse": float(recon_mse.detach().cpu()),
            "kl": float(kl.detach().cpu()),
            "beta": float(beta),
            "posterior_mu_abs": float(mu_post.detach().abs().mean().cpu()),
            "prior_mu_abs": float(mu_prior.detach().abs().mean().cpu()),
            "posterior_std": float(torch.exp(0.5 * clamp_logvar(logvar_post)).mean().detach().cpu()),
            "prior_std": float(torch.exp(0.5 * clamp_logvar(logvar_prior)).mean().detach().cpu()),
        }
        return loss, stats

    predictions = model(coords)
    recon_l1 = F.l1_loss(predictions, frames)
    recon_mse = F.mse_loss(predictions, frames)
    recon_total = recon_l1 + config.training.recon_mse_weight * recon_mse
    return recon_total, {
        "loss": float(recon_total.detach().cpu()),
        "recon_total": float(recon_total.detach().cpu()),
        "recon_l1": float(recon_l1.detach().cpu()),
        "recon_mse": float(recon_mse.detach().cpu()),
    }


def _run_configured_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    *,
    config: TrainingConfig,
    device: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None,
    optimizer_step: int = 0,
    optimizer_step_callback: Optional[Callable[[int, nn.Module], None]] = None,
) -> tuple[float, dict[str, float], int]:
    training = optimizer is not None
    model.train(training)

    metric_sums: dict[str, float] = {}
    total_examples = 0
    current_optimizer_step = optimizer_step
    grad_accum_steps = config.training.grad_accum_steps if training else 1

    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, (coords, frames) in enumerate(loader):
        coords = coords.to(device, non_blocking=True)
        frames = frames.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with autocast(device_type="cuda", enabled=use_amp):
                loss, metrics = _compute_configured_batch_metrics(
                    model,
                    coords,
                    frames,
                    config=config,
                    optimizer_step=current_optimizer_step,
                    training=training,
                )
                scaled_loss = loss / grad_accum_steps if training else loss

        if training:
            if use_amp and scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = (
                (batch_index + 1) % grad_accum_steps == 0
                or batch_index + 1 == len(loader)
            )
            if should_step:
                if use_amp and scaler is not None:
                    if config.training.grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=config.training.grad_clip_norm,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if config.training.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=config.training.grad_clip_norm,
                        )
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                current_optimizer_step += 1
                if optimizer_step_callback is not None:
                    optimizer_step_callback(current_optimizer_step, model)

        batch_size = coords.shape[0]
        for name, value in metrics.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + value * batch_size
        total_examples += batch_size

    mean_metrics = {
        name: total / max(1, total_examples)
        for name, total in metric_sums.items()
    }
    return mean_metrics.get("loss", 0.0), mean_metrics, current_optimizer_step


def _run_training_loop(
    model: nn.Module,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    test_loader: Optional[DataLoader[tuple[Tensor, Tensor]]] = None,
    *,
    device: str,
    epochs: int,
    lr: float,
    use_amp: bool,
    history: Optional[TrainingHistory] = None,
    track_best: bool = False,
    config: Optional[TrainingConfig] = None,
    epoch_callback: Optional[
        Callable[[int, float, float, float, nn.Module, dict[str, float], dict[str, float], dict[str, float]], None]
    ] = None,
    optimizer_step_callback: Optional[Callable[[int, nn.Module], None]] = None,
) -> TrainingLoopArtifacts:
    optimizer = _build_optimizer(model, config) if config is not None else torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler("cuda", enabled=use_amp)
    history = history or TrainingHistory(train_losses=[], val_losses=[], test_losses=[])
    start_epoch = len(history.train_losses)
    best_val_loss = min(history.val_losses) if history.val_losses else None
    best_state_dict: Optional[dict[str, Tensor]] = None
    best_epoch = None
    optimizer_step = (
        start_epoch * math.ceil(len(train_loader) / max(1, config.training.grad_accum_steps))
        if config is not None
        else 0
    )

    model.to(device)
    for epoch in range(epochs):
        if config is None:
            train_loss = run_epoch(
                model,
                train_loader,
                device=device,
                optimizer=optimizer,
                use_amp=use_amp,
                scaler=scaler,
            )
            train_metrics = {"loss": train_loss}
            val_loss = run_epoch(model, val_loader, device=device, use_amp=use_amp)
            val_metrics = {"loss": val_loss}
            test_loss = (
                run_epoch(model, test_loader, device=device, use_amp=use_amp)
                if test_loader is not None
                else float("nan")
            )
            test_metrics = {"loss": test_loss}
        else:
            train_loss, train_metrics, optimizer_step = _run_configured_epoch(
                model,
                train_loader,
                config=config,
                device=device,
                optimizer=optimizer,
                use_amp=use_amp,
                scaler=scaler,
                optimizer_step=optimizer_step,
                optimizer_step_callback=optimizer_step_callback,
            )
            val_loss, val_metrics, _ = _run_configured_epoch(
                model,
                val_loader,
                config=config,
                device=device,
                use_amp=use_amp,
                optimizer_step=optimizer_step,
            )
            test_loss, test_metrics, _ = (
                _run_configured_epoch(
                    model,
                    test_loader,
                    config=config,
                    device=device,
                    use_amp=use_amp,
                    optimizer_step=optimizer_step,
                )
                if test_loader is not None
                else (float("nan"), {"loss": float("nan")}, optimizer_step)
            )
        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)
        history.test_losses.append(test_loss)

        epoch_number = start_epoch + epoch + 1
        print(
            f"epoch {epoch_number:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | test_loss={test_loss:.6f}",
            flush=True,
        )
        if epoch_callback is not None:
            epoch_callback(
                epoch_number,
                train_loss,
                val_loss,
                test_loss,
                model,
                train_metrics,
                val_metrics,
                test_metrics,
            )

        if track_best and (best_val_loss is None or val_loss < best_val_loss):
            best_val_loss = val_loss
            best_epoch = epoch_number
            best_state_dict = _clone_state_dict_to_cpu(model.state_dict())

    return TrainingLoopArtifacts(
        history=history,
        best_state_dict=best_state_dict,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    test_loader: Optional[DataLoader[tuple[Tensor, Tensor]]] = None,
    *,
    device: str = "cuda",
    epochs: int = 20,
    lr: float = 1e-3,
) -> TrainingHistory:
    resolved_device = _resolve_device(device)
    use_amp = resolved_device.startswith("cuda")
    return _run_training_loop(
        model,
        train_loader,
        val_loader,
        test_loader,
        device=resolved_device,
        epochs=epochs,
        lr=lr,
        use_amp=use_amp,
    ).history


@torch.inference_mode()
def predict_frames(model: nn.Module, coords: Tensor, *, device: str = "cuda") -> Tensor:
    model.eval()
    return model(coords.to(device)).cpu()


def save_checkpoint(
    checkpoint_path: str | Path | None,
    model: nn.Module,
    *,
    history: Optional[TrainingHistory] = None,
    extra: Optional[dict[str, Any]] = None,
    run_name: str = "coord_to_image_unet",
    version: str | None = None,
    checkpoint_root: str | Path = "model/checkpoints",
    filename_template: str | None = None,
    filename_run_name: str | None = None,
    model_state_dict: Optional[dict[str, Tensor]] = None,
    training_config: Optional[TrainingConfig] = None,
    split_artifact_path: Optional[Path] = None,
    resolved_config_path: Optional[Path] = None,
    git_state: Optional[dict[str, Any]] = None,
    checkpoint_kind: str | None = None,
) -> Path:
    checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else make_checkpoint_path(
            run_name=filename_run_name or run_name,
            version=version,
            root=checkpoint_root,
            filename_template=filename_template,
        )
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    compatibility_extra: dict[str, Any] = dict(extra or {})
    if training_config is not None:
        compatibility_extra.setdefault("run_name", training_config.experiment.run_name)
        compatibility_extra.setdefault("VERSION", training_config.version)
        compatibility_extra.setdefault("model_kind", training_config.model.kind)
        compatibility_extra.setdefault("image_size", training_config.image_size_tuple)
        compatibility_extra.setdefault("base_channels", training_config.model.base_channels)
        compatibility_extra.setdefault("latent_channels", training_config.model.latent_channels)
        compatibility_extra.setdefault("coord_dim", training_config.model.coord_dim)
        compatibility_extra.setdefault("bottleneck_size", training_config.model.bottleneck_size)
        compatibility_extra.setdefault("latent_dim", training_config.model.latent_dim)
        compatibility_extra.setdefault(
            "posterior_base_channels",
            training_config.model.posterior_base_channels,
        )
        compatibility_extra.setdefault(
            "prior_hidden_dim",
            training_config.model.prior_hidden_dim,
        )
        compatibility_extra.setdefault("coord_space", training_config.data.coord_space)
        compatibility_extra.setdefault("config_path", str(training_config.config_path))
        compatibility_extra.setdefault("config_hash", training_config.config_hash)
        compatibility_extra.setdefault(
            "resolved_training_config",
            training_config.to_checkpoint_dict(),
        )

    payload: dict[str, Any] = {
        "checkpoint_format_version": 2,
        "model_state_dict": model_state_dict or model.state_dict(),
        "run_name": run_name,
        "VERSION": version,
        "extra": compatibility_extra,
    }
    if training_config is not None:
        payload["config_path"] = str(training_config.config_path)
        payload["config_hash"] = training_config.config_hash
        payload["training_config"] = training_config.to_checkpoint_dict()
    if history is not None:
        payload["history"] = {
            "train_losses": history.train_losses,
            "val_losses": history.val_losses,
            "test_losses": history.test_losses,
        }
    if split_artifact_path is not None:
        payload["split_artifact_path"] = str(split_artifact_path)
    if resolved_config_path is not None:
        payload["resolved_config_path"] = str(resolved_config_path)
    if git_state is not None:
        payload["git_state"] = git_state
    if checkpoint_kind is not None:
        payload["checkpoint_kind"] = checkpoint_kind
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    *,
    device: str = "cpu",
) -> tuple[nn.Module, Optional[TrainingHistory], Optional[dict[str, Any]]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    history_payload = checkpoint.get("history")
    history = None
    if history_payload is not None:
        history = TrainingHistory(
            train_losses=list(history_payload.get("train_losses", [])),
            val_losses=list(history_payload.get("val_losses", [])),
            test_losses=list(history_payload.get("test_losses", [])),
        )

    extra = checkpoint.get("extra")
    return model, history, extra


def latest_checkpoint_for_run(
    run_name: str,
    *,
    checkpoint_root: str | Path = "model/checkpoints",
    version: str | None = None,
) -> Optional[Path]:
    checkpoint_root = Path(checkpoint_root)
    checkpoint_infos = list_checkpoints(
        root=checkpoint_root,
        glob_pattern=checkpoint_glob_pattern(run_name=run_name, version=version),
    )
    ordered_infos = sorted(
        checkpoint_infos,
        key=lambda info: (
            info.checkpoint_id if info.checkpoint_id is not None else -1,
            info.created_at_utc.isoformat() if info.created_at_utc is not None else "",
        ),
        reverse=True,
    )
    for info in ordered_infos:
        try:
            payload = _load_checkpoint_payload(info.path)
        except Exception:
            continue
        payload_run_name = payload.get("run_name") or payload.get("extra", {}).get("run_name")
        payload_version = payload.get("VERSION") or payload.get("extra", {}).get("VERSION")
        if payload_run_name == run_name and (version is None or payload_version == version):
            return info.path

    return latest_checkpoint(
        root=checkpoint_root,
        glob_pattern=checkpoint_glob_pattern(run_name=run_name, version=version),
    )


def _load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(checkpoint).__name__}")
    return checkpoint


def _validate_resume_checkpoint(
    checkpoint_path: Path,
    payload: dict[str, Any],
    config: TrainingConfig,
) -> None:
    payload_hash = payload.get("config_hash")
    payload_run_name = payload.get("run_name") or payload.get("extra", {}).get("run_name")
    if payload_hash is not None and payload_hash != config.config_hash:
        raise ValueError(
            f"Resume checkpoint {checkpoint_path} was created for a different config hash."
        )
    if payload_run_name is not None and payload_run_name != config.experiment.run_name:
        raise ValueError(
            f"Resume checkpoint {checkpoint_path} was created for run_name={payload_run_name!r}, expected {config.experiment.run_name!r}."
        )


def _resolve_resume_checkpoint(config: TrainingConfig) -> Optional[Path]:
    if config.training.resume.mode == "never":
        return None
    if config.training.resume.mode == "exact_path":
        assert config.training.resume.path is not None
        checkpoint_path = config.training.resume.path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        return checkpoint_path
    return latest_checkpoint_for_run(
        config.experiment.run_name,
        checkpoint_root=config.paths.checkpoint_root,
        version=config.version,
    )


def run_training_from_config(config: TrainingConfig) -> TrainingRunResult:
    _set_reproducibility(
        config.reproducibility.seed,
        deterministic=config.reproducibility.enforce_determinism,
    )

    resolved_config_path = (
        save_resolved_config(config)
        if config.reproducibility.save_resolved_config
        else None
    )
    git_state = (
        _capture_git_state(config.repo_root)
        if config.reproducibility.capture_git_state
        else None
    )

    dataset = build_dataset_from_config(config)
    train_indices, val_indices, test_indices, split_artifact_path = _resolve_dataset_splits(
        dataset,
        config,
    )

    train_loader = _make_loader(
        Subset(dataset, train_indices),
        batch_size=config.loader.batch_size,
        shuffle=config.loader.shuffle_train,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
    )
    val_loader = _make_loader(
        Subset(dataset, val_indices),
        batch_size=config.loader.batch_size,
        shuffle=False,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
    )
    test_loader: Optional[DataLoader[tuple[Tensor, Tensor]]] = None
    if test_indices:
        test_loader = _make_loader(
            Subset(dataset, test_indices),
            batch_size=config.loader.batch_size,
            shuffle=False,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
        )

    device = _resolve_device(config.training.device)
    use_amp = _resolve_amp(config.training.amp, device=device)
    model = build_model_from_config(config).to(device)

    resumed_from_checkpoint = _resolve_resume_checkpoint(config)
    history: Optional[TrainingHistory] = None
    if resumed_from_checkpoint is not None:
        checkpoint_payload = _load_checkpoint_payload(resumed_from_checkpoint)
        _validate_resume_checkpoint(resumed_from_checkpoint, checkpoint_payload, config)
        model, history, _ = load_checkpoint(
            resumed_from_checkpoint,
            model,
            device=device,
        )
        print(f"resuming from checkpoint: {resumed_from_checkpoint}")
    history = history or TrainingHistory(train_losses=[], val_losses=[], test_losses=[])

    wandb_tracker = TrainingWandbTracker(
        config,
        dataset=dataset,
        val_indices=val_indices,
        device=device,
        train_examples=len(train_indices),
        val_examples=len(val_indices),
        test_examples=len(test_indices),
        resolved_config_path=resolved_config_path,
        split_artifact_path=split_artifact_path,
        git_state=git_state,
        resumed_from_checkpoint=resumed_from_checkpoint,
    )

    checkpoint_extra = {
        "train_examples": len(train_indices),
        "val_examples": len(val_indices),
        "test_examples": len(test_indices),
        "device": device,
        "amp": use_amp,
        "model_kind": config.model.kind,
        "coord_space": config.data.coord_space,
        "resumed_from_checkpoint": str(resumed_from_checkpoint) if resumed_from_checkpoint else None,
        "wandb_run_id": wandb_tracker.info.run_id,
        "wandb_run_name": wandb_tracker.info.run_name,
        "wandb_run_url": wandb_tracker.info.run_url,
        "wandb_project": config.wandb.project if config.wandb.enabled else None,
        "wandb_entity": config.wandb.entity if config.wandb.enabled else None,
    }

    def maybe_save_periodic_checkpoint(optimizer_step_number: int, current_model: nn.Module) -> None:
        interval = config.checkpointing.save_every_n_optimizer_steps
        if interval is None or optimizer_step_number % interval != 0:
            return
        save_checkpoint(
            None,
            current_model,
            history=history if config.checkpointing.keep_history else None,
            extra={
                **checkpoint_extra,
                "optimizer_step": optimizer_step_number,
            },
            run_name=config.experiment.run_name,
            version=config.version,
            checkpoint_root=config.paths.checkpoint_root,
            filename_template=config.checkpointing.filename_template,
            filename_run_name=f"{config.experiment.run_name}_{_optimizer_step_token(optimizer_step_number)}",
            training_config=config,
            split_artifact_path=split_artifact_path,
            resolved_config_path=resolved_config_path,
            git_state=git_state,
            checkpoint_kind="periodic_step",
        )

    def handle_epoch_end(
        epoch_number: int,
        train_loss: float,
        val_loss: float,
        test_loss: float,
        current_model: nn.Module,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        test_metrics: dict[str, float],
    ) -> None:
        wandb_tracker.log_epoch(
            epoch_number=epoch_number,
            train_loss=train_loss,
            val_loss=val_loss,
            test_loss=test_loss,
            model=current_model,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )
        interval = config.checkpointing.save_every_n_epochs
        if interval is None or epoch_number % interval != 0:
            return
        save_checkpoint(
            None,
            current_model,
            history=history if config.checkpointing.keep_history else None,
            extra={
                **checkpoint_extra,
                "epoch": epoch_number,
            },
            run_name=config.experiment.run_name,
            version=config.version,
            checkpoint_root=config.paths.checkpoint_root,
            filename_template=config.checkpointing.filename_template,
            filename_run_name=f"{config.experiment.run_name}_{_epoch_token(epoch_number)}",
            training_config=config,
            split_artifact_path=split_artifact_path,
            resolved_config_path=resolved_config_path,
            git_state=git_state,
            checkpoint_kind="periodic_epoch",
        )

    exit_code = 1
    try:
        loop_artifacts = _run_training_loop(
            model,
            train_loader,
            val_loader,
            test_loader,
            device=device,
            epochs=config.training.epochs,
            lr=config.training.lr,
            use_amp=use_amp,
            history=history,
            track_best=config.checkpointing.save_best,
            config=config,
            optimizer_step_callback=maybe_save_periodic_checkpoint,
            epoch_callback=handle_epoch_end,
        )

        final_checkpoint_path: Optional[Path] = None
        if config.checkpointing.save_final:
            final_checkpoint_path = save_checkpoint(
                None,
                model,
                history=loop_artifacts.history if config.checkpointing.keep_history else None,
                extra=checkpoint_extra,
                run_name=config.experiment.run_name,
                version=config.version,
                checkpoint_root=config.paths.checkpoint_root,
                filename_template=config.checkpointing.filename_template,
                filename_run_name=(
                    f"{config.experiment.run_name}_final_"
                    f"{_epoch_token(len(loop_artifacts.history.train_losses))}"
                ),
                training_config=config,
                split_artifact_path=split_artifact_path,
                resolved_config_path=resolved_config_path,
                git_state=git_state,
                checkpoint_kind="final",
            )

        best_checkpoint_path: Optional[Path] = None
        if config.checkpointing.save_best and loop_artifacts.best_state_dict is not None:
            best_checkpoint_path = save_checkpoint(
                None,
                model,
                history=loop_artifacts.history if config.checkpointing.keep_history else None,
                extra={
                    **checkpoint_extra,
                    "best_epoch": loop_artifacts.best_epoch,
                    "best_val_loss": loop_artifacts.best_val_loss,
                },
                run_name=f"{config.experiment.run_name}_best",
                version=config.version,
                checkpoint_root=config.paths.checkpoint_root,
                filename_template=config.checkpointing.filename_template,
                filename_run_name=(
                    f"{config.experiment.run_name}_best_"
                    f"{_epoch_token(loop_artifacts.best_epoch)}"
                ),
                model_state_dict=loop_artifacts.best_state_dict,
                training_config=config,
                split_artifact_path=split_artifact_path,
                resolved_config_path=resolved_config_path,
                git_state=git_state,
                checkpoint_kind="best",
            )

        wandb_tracker.log_training_outputs(
            history=loop_artifacts.history,
            final_checkpoint_path=final_checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
            best_epoch=loop_artifacts.best_epoch,
            best_val_loss=loop_artifacts.best_val_loss,
        )

        result = TrainingRunResult(
            model=model,
            history=loop_artifacts.history,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            dataset=dataset,
            final_checkpoint_path=final_checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
            resumed_from_checkpoint=resumed_from_checkpoint,
            split_artifact_path=split_artifact_path,
            resolved_config_path=resolved_config_path,
            git_state=git_state,
            wandb_run_id=wandb_tracker.info.run_id,
            wandb_run_name=wandb_tracker.info.run_name,
            wandb_run_url=wandb_tracker.info.run_url,
        )
        exit_code = 0
        return result
    finally:
        wandb_tracker.finish(exit_code=exit_code)


def train_or_load_model(
    model: nn.Module,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    test_loader: Optional[DataLoader[tuple[Tensor, Tensor]]] = None,
    *,
    run_name: str = "coord_to_image_unet",
    checkpoint_root: str | Path = "model/checkpoints",
    prefer_existing_checkpoint: bool = True,
    device: str = "cuda",
    epochs: int = 20,
    lr: float = 1e-3,
    extra: Optional[dict[str, Any]] = None,
) -> tuple[nn.Module, Optional[TrainingHistory], Optional[dict[str, Any]], Path, bool]:
    existing_checkpoint = latest_checkpoint_for_run(
        run_name=run_name,
        checkpoint_root=checkpoint_root,
    )
    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)

    if prefer_existing_checkpoint and existing_checkpoint is not None:
        model, loaded_history, loaded_extra = load_checkpoint(
            existing_checkpoint,
            model,
            device=resolved_device,
        )
        return model, loaded_history, loaded_extra, existing_checkpoint, True

    history = train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        device=resolved_device,
        epochs=epochs,
        lr=lr,
    )
    checkpoint_path = save_checkpoint(
        None,
        model,
        history=history,
        extra=extra,
        run_name=run_name,
        checkpoint_root=checkpoint_root,
    )
    return model, history, extra, checkpoint_path, False
