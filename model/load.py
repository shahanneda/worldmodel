from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

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


@dataclass(frozen=True)
class FingerSample:
    source_dir: Path
    frame_path: Path
    record: FingerSampleRecord


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


def _is_processed_dir(path: Path) -> bool:
    return (path / "index_finger_positions.json").exists() and (path / "segmented_frames").is_dir()


def discover_processed_dirs(
    root: str | Path = "data",
    *,
    glob_pattern: str = "processed-finger*",
) -> list[Path]:
    root = Path(root)
    if _is_processed_dir(root):
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Could not find data root: {root}")

    processed_dirs = sorted(
        path for path in root.glob(glob_pattern) if path.is_dir() and _is_processed_dir(path)
    )
    if not processed_dirs:
        raise FileNotFoundError(f"No processed dataset directories found under {root}")
    return processed_dirs


def _resolve_processed_dirs(
    processed_dir: str | Path | Sequence[str | Path],
    *,
    glob_pattern: str = "processed-finger*",
) -> list[Path]:
    if isinstance(processed_dir, (str, Path)):
        return discover_processed_dirs(processed_dir, glob_pattern=glob_pattern)

    resolved: list[Path] = []
    seen: set[Path] = set()
    for item in processed_dir:
        for path in discover_processed_dirs(item, glob_pattern=glob_pattern):
            if path not in seen:
                seen.add(path)
                resolved.append(path)
    if not resolved:
        raise FileNotFoundError("No processed dataset directories were resolved.")
    return resolved


class FingerVideoDataset(Dataset[tuple[Tensor, Tensor]]):
    """
    Returns `(coords, frame)` for training loops.

    - `coords`: tensor shaped `(2,)`
    - `frame`: tensor shaped `(C, H, W)` in `[0, 1]`

    By default:
    - coordinates are normalized to `[0, 1]`
    - images are resized to `128 x 128`
    - passing `processed_dir='data'` uses every matching processed dataset under `data/`
    """

    def __init__(
        self,
        processed_dir: str | Path | Sequence[str | Path] = "data",
        *,
        image_size: tuple[int, int] | None = (128, 128),
        normalized_coords: bool = True,
        drop_missing: bool = True,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        glob_pattern: str = "processed-finger*",
    ) -> None:
        self.processed_dirs = _resolve_processed_dirs(processed_dir, glob_pattern=glob_pattern)
        self.image_size = image_size
        self.normalized_coords = normalized_coords
        self.transform = transform

        self.samples: list[FingerSample] = []
        for directory in self.processed_dirs:
            records = _load_records(directory / "index_finger_positions.json")
            frame_paths = _load_frame_paths(directory / "segmented_frames")

            if len(records) != len(frame_paths):
                raise ValueError(
                    "Frame count mismatch between JSON positions and segmented frames: "
                    f"{len(records)} records vs {len(frame_paths)} frame files in {directory}"
                )

            for record, frame_path in zip(records, frame_paths):
                if drop_missing and (record.x_px is None or record.y_px is None):
                    continue
                self.samples.append(
                    FingerSample(
                        source_dir=directory,
                        frame_path=frame_path,
                        record=record,
                    )
                )

        if not self.samples:
            raise ValueError("Dataset is empty after resolving processed dirs and filtering samples.")

    def __len__(self) -> int:
        return len(self.samples)

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
        sample = self.samples[index]
        coords = self._coords_tensor(sample.record)
        frame = self._frame_tensor(sample.frame_path)
        return coords, frame


def build_finger_dataloader(
    processed_dir: str | Path | Sequence[str | Path] = "data",
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    image_size: tuple[int, int] | None = (128, 128),
    normalized_coords: bool = True,
    drop_missing: bool = True,
    transform: Optional[Callable[[Tensor], Tensor]] = None,
    pin_memory: bool = True,
    glob_pattern: str = "processed-finger*",
) -> DataLoader[tuple[Tensor, Tensor]]:
    dataset = FingerVideoDataset(
        processed_dir=processed_dir,
        image_size=image_size,
        normalized_coords=normalized_coords,
        drop_missing=drop_missing,
        transform=transform,
        glob_pattern=glob_pattern,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def demo_batch(
    processed_dir: str | Path | Sequence[str | Path] = "data",
    *,
    batch_size: int = 4,
    image_size: tuple[int, int] | None = (128, 128),
) -> tuple[Tensor, Tensor]:
    loader = build_finger_dataloader(
        processed_dir=processed_dir,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )
    return next(iter(loader))
