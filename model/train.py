from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

from model.checkpoints import latest_checkpoint, make_checkpoint_path
from model.load import FingerVideoDataset


@dataclass
class TrainingHistory:
    train_losses: list[float]
    val_losses: list[float]
    test_losses: list[float]


def reconstruction_loss(prediction: Tensor, target: Tensor) -> Tensor:
    l1 = F.l1_loss(prediction, target)
    mse = F.mse_loss(prediction, target)
    return l1 + 0.25 * mse


def make_train_val_test_loaders(
    processed_dir: str | Path = "data",
    *,
    image_size: tuple[int, int] = (128, 128),
    batch_size: int = 16,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
]:
    dataset = FingerVideoDataset(
        processed_dir=processed_dir,
        image_size=image_size,
        normalized_coords=True,
        drop_missing=True,
    )

    test_size = int(len(dataset) * test_fraction)
    val_size = int(len(dataset) * val_fraction)
    if test_fraction > 0 and test_size == 0:
        test_size = 1
    if val_fraction > 0 and val_size == 0:
        val_size = 1
    train_size = len(dataset) - val_size - test_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested train/val/test split.")
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def make_train_val_loaders(
    processed_dir: str | Path = "data",
    *,
    image_size: tuple[int, int] = (128, 128),
    batch_size: int = 16,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    train_loader, val_loader, _ = make_train_val_test_loaders(
        processed_dir=processed_dir,
        image_size=image_size,
        batch_size=batch_size,
        val_fraction=val_fraction,
        test_fraction=0.0,
        seed=seed,
        num_workers=num_workers,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = TrainingHistory(train_losses=[], val_losses=[], test_losses=[])
    use_amp = device.startswith("cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    model.to(device)
    for epoch in range(epochs):
        train_loss = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_loss = run_epoch(model, val_loader, device=device, use_amp=use_amp)
        test_loss = (
            run_epoch(model, test_loader, device=device, use_amp=use_amp)
            if test_loader is not None
            else float("nan")
        )
        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)
        history.test_losses.append(test_loss)
        print(
            f"epoch {epoch + 1:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | test_loss={test_loss:.6f}",
            flush=True,
        )

    return history


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
    checkpoint_root: str | Path = "model/checkpoints",
) -> Path:
    checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else make_checkpoint_path(run_name=run_name, root=checkpoint_root)
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
    }
    if history is not None:
        payload["history"] = {
            "train_losses": history.train_losses,
            "val_losses": history.val_losses,
            "test_losses": history.test_losses,
        }
    if extra is not None:
        payload["extra"] = extra
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
) -> Optional[Path]:
    return latest_checkpoint(root=checkpoint_root, glob_pattern=f"{run_name}*.pt")


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
    model = model.to(device)

    if prefer_existing_checkpoint and existing_checkpoint is not None:
        model, loaded_history, loaded_extra = load_checkpoint(
            existing_checkpoint,
            model,
            device=device,
        )
        return model, loaded_history, loaded_extra, existing_checkpoint, True

    history = train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        device=device,
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
