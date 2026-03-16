from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from model.config import TrainingConfig
from model.cvae import PointingCVAE
from model.load import FingerVideoDataset


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    token = token.strip("._-")
    return token or "unnamed"


def load_repo_env(repo_root: Path, *, override: bool = False) -> Path | None:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path


def _include_repo_code(path: str, root: str | None = None) -> bool:
    file_path = Path(path)
    if root is not None:
        try:
            relative = file_path.relative_to(root)
        except ValueError:
            relative = file_path
    else:
        relative = file_path
    if any(
        part in {".git", "__pycache__", ".ipynb_checkpoints", "wandb", "node_modules"}
        for part in relative.parts
    ):
        return False
    if relative.parts and relative.parts[0] == "data":
        return False
    if relative.parts[:2] in {("model", "checkpoints"), ("model", "artifacts"), ("model", "splits")}:
        return False
    if file_path.name == "Dockerfile":
        return True
    return file_path.suffix.lower() in {".py", ".ipynb", ".md", ".yaml", ".yml", ".txt"}


def _sample_preview_indices(indices: Sequence[int], sample_count: int) -> list[int]:
    if sample_count <= 0 or not indices:
        return []
    if len(indices) <= sample_count:
        return list(indices)
    if sample_count == 1:
        return [indices[0]]

    selected: list[int] = []
    last_position = len(indices) - 1
    for slot in range(sample_count):
        position = round(slot * last_position / (sample_count - 1))
        index = indices[position]
        if not selected or selected[-1] != index:
            selected.append(index)
    return selected


def _tensor_to_uint8_image(tensor: Tensor) -> np.ndarray:
    image = tensor.detach().cpu().float()
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got {tuple(image.shape)}")
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    image = image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (image * 255.0).round().astype(np.uint8)


def _preview_panel(*images: Tensor) -> np.ndarray:
    return np.concatenate(
        [_tensor_to_uint8_image(image) for image in images],
        axis=1,
    )


@dataclass(frozen=True)
class WandbRunInfo:
    run_id: str | None
    run_name: str | None
    run_url: str | None


class TrainingWandbTracker:
    def __init__(
        self,
        config: TrainingConfig,
        *,
        dataset: FingerVideoDataset,
        val_indices: Sequence[int],
        device: str,
        train_examples: int,
        val_examples: int,
        test_examples: int,
        resolved_config_path: Path | None,
        split_artifact_path: Path | None,
        git_state: dict[str, Any] | None,
        resumed_from_checkpoint: Path | None,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.device = device
        self._resolved_config_path = resolved_config_path
        self._split_artifact_path = split_artifact_path
        self._git_state = git_state
        self._train_examples = train_examples
        self._val_examples = val_examples
        self._test_examples = test_examples
        self._resumed_from_checkpoint = resumed_from_checkpoint
        self._preview_indices = _sample_preview_indices(
            val_indices,
            config.wandb.validation_sample_count,
        )
        self._run = None
        self._wandb = None

        if not config.wandb.enabled or config.wandb.mode == "disabled":
            self.info = WandbRunInfo(run_id=None, run_name=None, run_url=None)
            return

        load_repo_env(config.repo_root)

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "wandb is enabled in the training config but the package is not installed. "
                "Install it with `python3 -m pip install wandb`."
            ) from exc

        api_key = os.environ.get("WANDB_API_KEY", "").strip()
        if api_key:
            wandb.login(key=api_key, relogin=True, force=True)

        config_payload = config.to_checkpoint_dict()
        config_payload["config_hash"] = config.config_hash
        if resolved_config_path is not None:
            config_payload["resolved_config_path"] = str(resolved_config_path)
        if split_artifact_path is not None:
            config_payload["split_artifact_path"] = str(split_artifact_path)

        tags = list(
            dict.fromkeys(
                [
                    *config.experiment.tags,
                    *config.wandb.tags,
                    f"version:{config.version_token}",
                    f"run:{_safe_token(config.experiment.run_name)}",
                ]
            )
        )

        config.paths.artifact_root.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            dir=str(config.paths.artifact_root),
            name=f"{config.version_token}-{_safe_token(config.experiment.run_name)}",
            notes=config.experiment.description,
            tags=tags,
            config=config_payload,
            group=config.wandb.group,
            job_type=config.wandb.job_type,
            mode=config.wandb.mode,
            reinit="create_new",
            save_code=False,
        )

        self._wandb = wandb
        self._run = run
        self.info = WandbRunInfo(
            run_id=getattr(run, "id", None),
            run_name=getattr(run, "name", None),
            run_url=getattr(run, "url", None),
        )

        if config.wandb.log_code:
            run.log_code(root=str(config.repo_root), include_fn=_include_repo_code)

        summary_payload = {
            "VERSION": config.version,
            "config_hash": config.config_hash,
            "config_path": str(config.config_path),
            "resolved_config_path": str(resolved_config_path) if resolved_config_path else None,
            "split_artifact_path": str(split_artifact_path) if split_artifact_path else None,
            "run_name": config.experiment.run_name,
            "dataset_count": len(config.data.datasets),
            "dataset_names": [dataset_config.name for dataset_config in config.data.datasets],
            "dataset_dirs": [str(dataset_config.processed_dir) for dataset_config in config.data.datasets],
            "dataset_size": len(dataset),
            "train_examples": train_examples,
            "val_examples": val_examples,
            "test_examples": test_examples,
            "image_size": config.model.image_size,
            "model_kind": config.model.kind,
            "base_channels": config.model.base_channels,
            "latent_channels": config.model.latent_channels,
            "latent_dim": config.model.latent_dim,
            "posterior_base_channels": config.model.posterior_base_channels,
            "prior_hidden_dim": config.model.prior_hidden_dim,
            "coord_dim": config.model.coord_dim,
            "coord_space": config.data.coord_space,
            "batch_size": config.loader.batch_size,
            "epochs": config.training.epochs,
            "learning_rate": config.training.lr,
            "optimizer": config.training.optimizer,
            "weight_decay": config.training.weight_decay,
            "grad_accum_steps": config.training.grad_accum_steps,
            "grad_clip_norm": config.training.grad_clip_norm,
            "recon_mse_weight": config.training.recon_mse_weight,
            "kl_beta_max": config.training.kl_beta_max,
            "kl_beta_warmup_steps": config.training.kl_beta_warmup_steps,
            "prior_sample_temperature": config.training.prior_sample_temperature,
            "validation_prior_sample_count": config.wandb.validation_prior_sample_count,
            "validation_prior_temperature": config.wandb.validation_prior_temperature,
            "device": device,
            "amp": config.training.amp,
            "resume_mode": config.training.resume.mode,
            "resumed_from_checkpoint": (
                str(resumed_from_checkpoint) if resumed_from_checkpoint is not None else None
            ),
            "split_seed": config.splits.seed,
        }
        if git_state is not None:
            summary_payload["git_commit"] = git_state.get("commit")
            summary_payload["git_dirty"] = git_state.get("dirty")
        run.summary.update(summary_payload)

        if config.wandb.log_config_artifact:
            self._log_setup_artifacts()

    def _log_setup_artifacts(self) -> None:
        assert self._run is not None
        assert self._wandb is not None

        metadata = {
            "VERSION": self.config.version,
            "config_hash": self.config.config_hash,
            "run_name": self.config.experiment.run_name,
        }
        config_artifact = self._wandb.Artifact(
            name=f"{_safe_token(self.config.experiment.run_name)}-{self.config.version_token}-config",
            type="training-config",
            description="Training config, resolved config snapshot, and split artifact for this run.",
            metadata=metadata,
        )
        config_artifact.add_file(str(self.config.config_path), name=f"original/{self.config.config_path.name}")
        if self._resolved_config_path is not None and self._resolved_config_path.exists():
            config_artifact.add_file(str(self._resolved_config_path), name="resolved/resolved_config.yaml")
        if self._split_artifact_path is not None and self._split_artifact_path.exists():
            config_artifact.add_file(str(self._split_artifact_path), name="splits/splits.json")
        self._run.log_artifact(
            config_artifact,
            aliases=["latest", self.config.version_token],
        )

    def _log_baseline_preview(self, model: nn.Module, *, epoch_number: int) -> None:
        assert self._run is not None
        assert self._wandb is not None

        preview_table = self._wandb.Table(
            columns=[
                "sample_id",
                "source_dir",
                "frame_index",
                "coord_x",
                "coord_y",
                "target",
                "prediction",
                "abs_diff",
                "comparison",
            ]
        )
        preview_gallery: list[Any] = []

        for sample_index in self._preview_indices:
            coords, target = self.dataset[sample_index]
            prediction = model(coords.unsqueeze(0).to(self.device)).squeeze(0).detach().cpu()
            target = target.detach().cpu()
            diff = (prediction - target).abs()
            sample = self.dataset.samples[sample_index]
            sample_id = self.dataset.sample_identifier(sample_index)
            coord_x = float(coords[0].item())
            coord_y = float(coords[1].item())
            comparison = _preview_panel(target, prediction, diff)
            caption = (
                f"{sample_id} | coords=({coord_x:.4f}, {coord_y:.4f}) | "
                "columns: target | prediction | abs_diff"
            )
            comparison_image = self._wandb.Image(comparison, caption=caption)
            preview_gallery.append(comparison_image)
            preview_table.add_data(
                sample_id,
                str(sample.source_dir),
                sample.record.frame_index,
                coord_x,
                coord_y,
                self._wandb.Image(_tensor_to_uint8_image(target)),
                self._wandb.Image(_tensor_to_uint8_image(prediction)),
                self._wandb.Image(_tensor_to_uint8_image(diff)),
                comparison_image,
            )

        self._run.log(
            {
                "validation/sample_table": preview_table,
                "validation/sample_gallery": preview_gallery,
            },
            step=epoch_number,
        )

    def _log_cvae_preview(self, model: PointingCVAE, *, epoch_number: int) -> None:
        assert self._run is not None
        assert self._wandb is not None

        sample_columns = [
            f"prior_sample_{index + 1}"
            for index in range(self.config.wandb.validation_prior_sample_count)
        ]
        preview_table = self._wandb.Table(
            columns=[
                "sample_id",
                "source_dir",
                "frame_index",
                "coord_x",
                "coord_y",
                "target",
                "posterior_mean",
                "prior_mean",
                *sample_columns,
                "comparison",
            ]
        )
        preview_gallery: list[Any] = []

        for sample_index in self._preview_indices:
            coords, target = self.dataset[sample_index]
            coord_batch = coords.unsqueeze(0).to(self.device)
            target_batch = target.unsqueeze(0).to(self.device)

            posterior_mean = model.reconstruct_from_posterior_mean(coord_batch, target_batch)[
                "img_hat"
            ].squeeze(0).detach().cpu()
            prior_mean = model.sample_prior_mean(coord_batch)["img_hat"].squeeze(0).detach().cpu()

            prior_samples: list[Tensor] = []
            for _ in range(self.config.wandb.validation_prior_sample_count):
                prior_sample = model.sample_from_prior(
                    coord_batch,
                    temperature=self.config.wandb.validation_prior_temperature,
                )["img_hat"].squeeze(0).detach().cpu()
                prior_samples.append(prior_sample)

            target = target.detach().cpu()
            sample = self.dataset.samples[sample_index]
            sample_id = self.dataset.sample_identifier(sample_index)
            coord_x = float(coords[0].item())
            coord_y = float(coords[1].item())

            comparison = _preview_panel(target, posterior_mean, prior_mean, *prior_samples)
            caption = (
                f"{sample_id} | coords=({coord_x:.4f}, {coord_y:.4f}) | "
                "columns: target | posterior_mean | prior_mean"
            )
            if prior_samples:
                caption += " | " + " | ".join(sample_columns)
            comparison_image = self._wandb.Image(comparison, caption=caption)
            preview_gallery.append(comparison_image)

            preview_table.add_data(
                sample_id,
                str(sample.source_dir),
                sample.record.frame_index,
                coord_x,
                coord_y,
                self._wandb.Image(_tensor_to_uint8_image(target)),
                self._wandb.Image(_tensor_to_uint8_image(posterior_mean)),
                self._wandb.Image(_tensor_to_uint8_image(prior_mean)),
                *[self._wandb.Image(_tensor_to_uint8_image(image)) for image in prior_samples],
                comparison_image,
            )

        self._run.log(
            {
                "validation/sample_table": preview_table,
                "validation/sample_gallery": preview_gallery,
            },
            step=epoch_number,
        )

    def log_epoch(
        self,
        *,
        epoch_number: int,
        train_loss: float,
        val_loss: float,
        test_loss: float,
        model: nn.Module,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        test_metrics: dict[str, float],
    ) -> None:
        if self._run is None or self._wandb is None:
            return

        metrics = {
            "epoch": epoch_number,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "test/loss": test_loss,
        }
        for prefix, metric_payload in (
            ("train", train_metrics),
            ("val", val_metrics),
            ("test", test_metrics),
        ):
            for name, value in metric_payload.items():
                if name == "loss":
                    continue
                metrics[f"{prefix}/{name}"] = value
        self._run.log(metrics, step=epoch_number)
        self._run.summary["latest_epoch"] = epoch_number
        self._run.summary["latest_train_loss"] = train_loss
        self._run.summary["latest_val_loss"] = val_loss
        self._run.summary["latest_test_loss"] = test_loss

        should_log_preview = (
            self.config.wandb.validation_sample_count > 0
            and self._preview_indices
            and epoch_number % self.config.wandb.log_validation_every_n_epochs == 0
        )
        if not should_log_preview:
            return

        was_training = model.training
        model.eval()
        with torch.inference_mode():
            if isinstance(model, PointingCVAE):
                self._log_cvae_preview(model, epoch_number=epoch_number)
            else:
                self._log_baseline_preview(model, epoch_number=epoch_number)
        model.train(was_training)

    def log_training_outputs(
        self,
        *,
        history: Any,
        final_checkpoint_path: Path | None,
        best_checkpoint_path: Path | None,
        best_epoch: int | None,
        best_val_loss: float | None,
    ) -> None:
        if self._run is None or self._wandb is None:
            return

        if history.train_losses:
            self._run.summary["final_train_loss"] = history.train_losses[-1]
        if history.val_losses:
            self._run.summary["final_val_loss"] = history.val_losses[-1]
        if history.test_losses:
            self._run.summary["final_test_loss"] = history.test_losses[-1]
        if best_epoch is not None:
            self._run.summary["best_epoch"] = best_epoch
        if best_val_loss is not None:
            self._run.summary["best_val_loss"] = best_val_loss
        if final_checkpoint_path is not None:
            self._run.summary["final_checkpoint_path"] = str(final_checkpoint_path)
        if best_checkpoint_path is not None:
            self._run.summary["best_checkpoint_path"] = str(best_checkpoint_path)

        if self.config.wandb.log_model_artifact:
            if final_checkpoint_path is not None and final_checkpoint_path.exists():
                self._log_model_artifact(final_checkpoint_path, checkpoint_kind="final")
            if best_checkpoint_path is not None and best_checkpoint_path.exists():
                self._log_model_artifact(best_checkpoint_path, checkpoint_kind="best")

    def _log_model_artifact(self, checkpoint_path: Path, *, checkpoint_kind: str) -> None:
        assert self._run is not None
        assert self._wandb is not None

        artifact = self._wandb.Artifact(
            name=(
                f"{_safe_token(self.config.experiment.run_name)}-"
                f"{self.config.version_token}-{checkpoint_kind}"
            ),
            type="model",
            description=f"{checkpoint_kind} checkpoint for {self.config.experiment.run_name}.",
            metadata={
                "VERSION": self.config.version,
                "config_hash": self.config.config_hash,
                "checkpoint_kind": checkpoint_kind,
                "run_name": self.config.experiment.run_name,
            },
        )
        artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)
        if self._resolved_config_path is not None and self._resolved_config_path.exists():
            artifact.add_file(str(self._resolved_config_path), name="resolved_config.yaml")
        self._run.log_artifact(
            artifact,
            aliases=[checkpoint_kind, self.config.version_token, "latest"],
        )

    def finish(self, *, exit_code: int) -> None:
        if self._run is not None:
            self._run.finish(exit_code=exit_code)
