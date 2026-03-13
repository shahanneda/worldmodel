from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class FingerSampleRecord:
    frame_index: int
    x_px: Optional[int]
    y_px: Optional[int]
    x_norm: Optional[float]
    y_norm: Optional[float]


def _load_records(index_json_path: Path) -> list[FingerSampleRecord]:
    with index_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    positions = payload.get("positions")
    if not isinstance(positions, list):
        raise ValueError(f"Expected 'positions' list in {index_json_path}")

    records: list[FingerSampleRecord] = []
    for item in positions:
        records.append(
            FingerSampleRecord(
                frame_index=int(item["frame_index"]),
                x_px=item.get("x_px"),
                y_px=item.get("y_px"),
                x_norm=item.get("x_norm"),
                y_norm=item.get("y_norm"),
            )
        )
    return records


def _frame_sort_key(path: Path) -> int:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse frame index from {path}")
    return int(digits)


def _load_frame_paths(segmented_frames_dir: Path) -> list[Path]:
    frame_paths = sorted(segmented_frames_dir.glob("frame*.jpg"), key=_frame_sort_key)
    if not frame_paths:
        raise FileNotFoundError(f"No frame*.jpg files found in {segmented_frames_dir}")
    return frame_paths


class FingerVideoDataset(Dataset[tuple[Tensor, Tensor]]):
    """
    Returns `(coords, frame)` so this works cleanly with PyTorch training loops:

    - `coords`: tensor shaped `(2,)`
    - `frame`: tensor shaped `(C, H, W)` in `[0, 1]`

    By default `coords` are normalized to `[0, 1]`.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        *,
        image_size: tuple[int, int] | None = (128, 128),
        normalized_coords: bool = True,
        drop_missing: bool = True,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.index_json_path = self.processed_dir / "index_finger_positions.json"
        self.segmented_frames_dir = self.processed_dir / "segmented_frames"

        if not self.index_json_path.exists():
            raise FileNotFoundError(f"Missing {self.index_json_path}")
        if not self.segmented_frames_dir.exists():
            raise FileNotFoundError(f"Missing {self.segmented_frames_dir}")

        self.records = _load_records(self.index_json_path)
        self.frame_paths = _load_frame_paths(self.segmented_frames_dir)
        self.image_size = image_size
        self.normalized_coords = normalized_coords
        self.transform = transform

        if len(self.records) != len(self.frame_paths):
            raise ValueError(
                "Frame count mismatch between JSON positions and segmented frames: "
                f"{len(self.records)} records vs {len(self.frame_paths)} frame files"
            )

        self.sample_indices = list(range(len(self.records)))
        if drop_missing:
            self.sample_indices = [
                idx
                for idx, record in enumerate(self.records)
                if record.x_px is not None and record.y_px is not None
            ]

        if not self.sample_indices:
            raise ValueError("Dataset is empty after filtering missing fingertip coordinates.")

    def __len__(self) -> int:
        return len(self.sample_indices)

    def _coords_tensor(self, record: FingerSampleRecord) -> Tensor:
        if self.normalized_coords:
            if record.x_norm is None or record.y_norm is None:
                raise ValueError(f"Missing normalized coordinates for frame {record.frame_index}")
            return torch.tensor([record.x_norm, record.y_norm], dtype=torch.float32)

        if record.x_px is None or record.y_px is None:
            raise ValueError(f"Missing pixel coordinates for frame {record.frame_index}")
        return torch.tensor([record.x_px, record.y_px], dtype=torch.float32)

    def _frame_tensor(self, frame_path: Path) -> Tensor:
        frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Could not read frame {frame_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        if self.image_size is not None:
            frame = F.interpolate(
                frame.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if self.transform is not None:
            frame = self.transform(frame)
        return frame

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample_index = self.sample_indices[index]
        record = self.records[sample_index]
        frame_path = self.frame_paths[sample_index]

        coords = self._coords_tensor(record)
        frame = self._frame_tensor(frame_path)
        return coords, frame


def build_finger_dataloader(
    processed_dir: str | Path,
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    image_size: tuple[int, int] | None = (128, 128),
    normalized_coords: bool = True,
    drop_missing: bool = True,
    transform: Optional[Callable[[Tensor], Tensor]] = None,
    pin_memory: bool = True,
) -> DataLoader[tuple[Tensor, Tensor]]:
    dataset = FingerVideoDataset(
        processed_dir=processed_dir,
        image_size=image_size,
        normalized_coords=normalized_coords,
        drop_missing=drop_missing,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def demo_batch(
    processed_dir: str | Path = "data/processed-finger-sam-2026-01-30T22-41-47-949Z",
    *,
    batch_size: int = 4,
    image_size: tuple[int, int] | None = (128, 128),
) -> tuple[Tensor, Tensor]:
    """
    Quick sanity helper for notebooks:

    >>> coords, frames = demo_batch()
    >>> coords.shape
    torch.Size([4, 2])
    >>> frames.shape
    torch.Size([4, 3, H, W])
    """
    loader = build_finger_dataloader(
        processed_dir=processed_dir,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )
    return next(iter(loader))
