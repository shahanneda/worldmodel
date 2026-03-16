from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityFilterConfig:
    max_foreground_ratio: float = 0.85
    max_border_touch_ratio: float = 0.75
    max_area_ratio_vs_window: float = 1.75
    min_border_touch_for_spike: float = 0.20
    window_size: int = 15


@dataclass(frozen=True)
class FrameQualityMetrics:
    frame_index: int
    foreground_ratio: float
    border_touch_ratio: float
    area_ratio_vs_window_median: float
    flagged: bool
    reasons: tuple[str, ...]


def _frame_sort_key(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse frame index from {path}")
    return int(digits)


def _load_payload(index_json_path: Path) -> dict[str, Any]:
    with index_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level object in {index_json_path}")
    positions = payload.get("positions")
    if not isinstance(positions, list):
        raise ValueError(f"Expected 'positions' list in {index_json_path}")
    return payload


def _mask_metrics(mask_path: Path) -> tuple[float, float]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask frame {mask_path}")

    foreground = mask > 0
    foreground_ratio = float(np.count_nonzero(foreground)) / float(foreground.size)

    border_pixels = np.concatenate(
        [
            foreground[0, :],
            foreground[-1, :],
            foreground[:, 0],
            foreground[:, -1],
        ]
    )
    border_touch_ratio = float(np.count_nonzero(border_pixels)) / float(border_pixels.size)
    return foreground_ratio, border_touch_ratio


def _window_median(values: list[float], center: int, window_radius: int) -> float:
    start = max(0, center - window_radius)
    stop = min(len(values), center + window_radius + 1)
    window = values[start:stop]
    if not window:
        return 0.0
    return float(np.median(np.asarray(window, dtype=np.float32)))


def _resolve_processed_dirs(path: Path, glob_pattern: str) -> list[Path]:
    if (path / "index_finger_positions.json").exists() and (path / "mask_frames").is_dir():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Could not find path: {path}")
    dirs = sorted(
        candidate
        for candidate in path.glob(glob_pattern)
        if candidate.is_dir()
        and (candidate / "index_finger_positions.json").exists()
        and (candidate / "mask_frames").is_dir()
    )
    if not dirs:
        raise FileNotFoundError(f"No processed dataset directories found under {path}")
    return dirs


def analyze_processed_dir(
    processed_dir: str | Path,
    *,
    config: QualityFilterConfig | None = None,
) -> list[FrameQualityMetrics]:
    config = config or QualityFilterConfig()
    processed_dir = Path(processed_dir)
    mask_dir = processed_dir / "mask_frames"
    frame_paths = sorted(mask_dir.glob("frame*.png"), key=_frame_sort_key)
    if not frame_paths:
        raise FileNotFoundError(f"No mask frame files found in {mask_dir}")

    window_radius = max(1, config.window_size // 2)
    foreground_ratios: list[float] = []
    border_touch_ratios: list[float] = []
    for frame_path in frame_paths:
        foreground_ratio, border_touch_ratio = _mask_metrics(frame_path)
        foreground_ratios.append(foreground_ratio)
        border_touch_ratios.append(border_touch_ratio)

    metrics: list[FrameQualityMetrics] = []
    for idx, frame_path in enumerate(frame_paths):
        frame_index = _frame_sort_key(frame_path)
        window_median = max(1e-6, _window_median(foreground_ratios, idx, window_radius))
        area_ratio_vs_window = foreground_ratios[idx] / window_median

        reasons: list[str] = []
        if (
            foreground_ratios[idx] >= config.max_foreground_ratio
            and border_touch_ratios[idx] >= config.max_border_touch_ratio
        ):
            reasons.append("foreground_and_border_ratio")
        if (
            area_ratio_vs_window >= config.max_area_ratio_vs_window
            and border_touch_ratios[idx] >= config.min_border_touch_for_spike
        ):
            reasons.append("area_spike_vs_window")

        metrics.append(
            FrameQualityMetrics(
                frame_index=frame_index,
                foreground_ratio=foreground_ratios[idx],
                border_touch_ratio=border_touch_ratios[idx],
                area_ratio_vs_window_median=area_ratio_vs_window,
                flagged=bool(reasons),
                reasons=tuple(reasons),
            )
        )

    return metrics


def apply_quality_filter(
    processed_dir: str | Path,
    *,
    config: QualityFilterConfig | None = None,
) -> dict[str, Any]:
    config = config or QualityFilterConfig()
    processed_dir = Path(processed_dir)
    index_json_path = processed_dir / "index_finger_positions.json"
    payload = _load_payload(index_json_path)
    positions = payload["positions"]
    metrics = analyze_processed_dir(processed_dir, config=config)

    if len(positions) != len(metrics):
        raise ValueError(
            "Frame count mismatch between JSON positions and mask frames: "
            f"{len(positions)} positions vs {len(metrics)} masks in {processed_dir}"
        )

    flagged_indices: list[int] = []
    for item, metric in zip(positions, metrics):
        item["quality_foreground_ratio"] = metric.foreground_ratio
        item["quality_border_touch_ratio"] = metric.border_touch_ratio
        item["quality_area_ratio_vs_window_median"] = metric.area_ratio_vs_window_median
        item["quality_flagged"] = metric.flagged
        item["quality_reasons"] = list(metric.reasons)
        if metric.flagged:
            flagged_indices.append(metric.frame_index)

    payload["quality_filter"] = {
        "enabled": True,
        "max_foreground_ratio": config.max_foreground_ratio,
        "max_border_touch_ratio": config.max_border_touch_ratio,
        "max_area_ratio_vs_window": config.max_area_ratio_vs_window,
        "min_border_touch_for_spike": config.min_border_touch_for_spike,
        "window_size": config.window_size,
        "flagged_frame_count": len(flagged_indices),
        "flagged_frame_indices": flagged_indices,
    }

    with index_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "processed_dir": str(processed_dir),
        "frame_count": len(metrics),
        "flagged_frame_count": len(flagged_indices),
        "flagged_frame_indices": flagged_indices,
    }


def apply_quality_filter_to_many(
    input_path: str | Path,
    *,
    config: QualityFilterConfig | None = None,
    glob_pattern: str = "processed-finger*",
) -> list[dict[str, Any]]:
    input_path = Path(input_path)
    results: list[dict[str, Any]] = []
    for processed_dir in _resolve_processed_dirs(input_path, glob_pattern):
        results.append(apply_quality_filter(processed_dir, config=config))
    return results
