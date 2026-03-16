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


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FingerSampleRecord:
    frame_index: int
    x_px: Optional[int]
    y_px: Optional[int]
    x_norm: Optional[float]
    y_norm: Optional[float]
    quality_flagged: bool = False
    finger_present: bool = False
    shirt_color_source: Optional[str] = None
    shirt_color_confidence: Optional[float] = None
    shirt_color_sample_count: Optional[int] = None


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
                quality_flagged=bool(item.get("quality_flagged", False)),
                finger_present=bool(
                    item.get("finger_present", item.get("x_px") is not None and item.get("y_px") is not None)
                ),
                shirt_color_source=item.get("shirt_color_source"),
                shirt_color_confidence=(
                    float(item["shirt_color_confidence"])
                    if item.get("shirt_color_confidence") is not None
                    else None
                ),
                shirt_color_sample_count=(
                    int(item["shirt_color_sample_count"])
                    if item.get("shirt_color_sample_count") is not None
                    else None
                ),
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
    - `require_finger=True` drops frames without a fingertip label
    - `require_shirt=True` drops frames where shirt color had to be filled instead of observed directly
    """

    def __init__(
        self,
        processed_dir: str | Path | Sequence[str | Path] = "data",
        *,
        image_size: tuple[int, int] | None = (128, 128),
        normalized_coords: bool = True,
        coord_space: str = "zero_to_one",
        drop_missing: bool = True,
        drop_quality_flagged: bool = True,
        require_finger: bool = False,
        require_shirt: bool = False,
        min_shirt_confidence: float = 0.0,
        min_shirt_sample_count: int = 1,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        glob_pattern: str = "processed-finger*",
    ) -> None:
        self.processed_dirs = _resolve_processed_dirs(processed_dir, glob_pattern=glob_pattern)
        self.image_size = image_size
        self.normalized_coords = normalized_coords
        if coord_space not in {"zero_to_one", "minus_one_to_one"}:
            raise ValueError("coord_space must be 'zero_to_one' or 'minus_one_to_one'.")
        self.coord_space = coord_space
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

            if require_shirt and all(record.shirt_color_source is None for record in records):
                raise ValueError(
                    "Shirt-based filtering was requested, but shirt color metadata is missing in "
                    f"{directory}. Run scripts/extract_processed_features.py for this dataset first."
                )

            for record, frame_path in zip(records, frame_paths):
                if drop_missing and (record.x_px is None or record.y_px is None):
                    continue
                if drop_quality_flagged and record.quality_flagged:
                    continue
                if require_finger and not record.finger_present:
                    continue
                if require_shirt and not _record_has_direct_shirt_signal(
                    record,
                    min_shirt_confidence=min_shirt_confidence,
                    min_shirt_sample_count=min_shirt_sample_count,
                ):
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

    def sample_identifier(self, index: int) -> str:
        sample = self.samples[index]
        source_dir = sample.source_dir.resolve()
        try:
            source_display = source_dir.relative_to(REPO_ROOT)
        except ValueError:
            source_display = source_dir
        return f"{source_display.as_posix()}::frame:{sample.record.frame_index}"

    def sample_identifiers(self) -> list[str]:
        return [self.sample_identifier(index) for index in range(len(self.samples))]

    def _coords_tensor(self, record: FingerSampleRecord) -> Tensor:
        if self.normalized_coords:
            if record.x_norm is None or record.y_norm is None:
                raise ValueError(f"Missing normalized coordinates for frame {record.frame_index}")
            coords = torch.tensor([record.x_norm, record.y_norm], dtype=torch.float32)
            if self.coord_space == "minus_one_to_one":
                coords = coords * 2.0 - 1.0
            return coords

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
    coord_space: str = "zero_to_one",
    drop_missing: bool = True,
    drop_quality_flagged: bool = True,
    require_finger: bool = False,
    require_shirt: bool = False,
    min_shirt_confidence: float = 0.0,
    min_shirt_sample_count: int = 1,
    transform: Optional[Callable[[Tensor], Tensor]] = None,
    pin_memory: bool = True,
    glob_pattern: str = "processed-finger*",
) -> DataLoader[tuple[Tensor, Tensor]]:
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


def _record_has_direct_shirt_signal(
    record: FingerSampleRecord,
    *,
    min_shirt_confidence: float,
    min_shirt_sample_count: int,
) -> bool:
    if record.shirt_color_source not in {"direct", "direct_temporal_blend"}:
        return False
    if record.shirt_color_confidence is None or record.shirt_color_sample_count is None:
        return False
    return (
        record.shirt_color_confidence >= min_shirt_confidence
        and record.shirt_color_sample_count >= min_shirt_sample_count
    )
