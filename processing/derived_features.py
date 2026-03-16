from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class ShirtColorConfig:
    torso_x_min: float = 0.28
    torso_x_max: float = 0.72
    torso_y_min: float = 0.42
    torso_y_max: float = 0.95
    min_direct_pixel_count: int = 1500
    min_local_window_direct_frames: int = 3
    local_window_size: int = 11
    direct_blend_weight: float = 0.70


@dataclass(frozen=True)
class ShirtColorEstimate:
    rgb: np.ndarray
    direct_pixel_count: int
    torso_pixel_count: int
    confidence: float
    direct_available: bool


FeatureApplier = Callable[[Path, dict[str, Any], bool], dict[str, Any]]


def _frame_sort_key(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse frame index from {path}")
    return int(digits)


def _resolve_processed_dirs(path: Path, glob_pattern: str) -> list[Path]:
    if (path / "index_finger_positions.json").exists() and (path / "segmented_frames").is_dir():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Could not find path: {path}")
    dirs = sorted(
        candidate
        for candidate in path.glob(glob_pattern)
        if candidate.is_dir()
        and (candidate / "index_finger_positions.json").exists()
        and (candidate / "segmented_frames").is_dir()
    )
    if not dirs:
        raise FileNotFoundError(f"No processed dataset directories found under {path}")
    return dirs


def _load_payload(index_json_path: Path) -> dict[str, Any]:
    with index_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level object in {index_json_path}")
    positions = payload.get("positions")
    if not isinstance(positions, list):
        raise ValueError(f"Expected 'positions' list in {index_json_path}")
    return payload


def _write_payload(index_json_path: Path, payload: dict[str, Any]) -> None:
    with index_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _hex_from_rgb(rgb: np.ndarray) -> str:
    r, g, b = [int(np.clip(round(v), 0, 255)) for v in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def _skin_mask(frame_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    return (
        (ycrcb[:, :, 1] >= 133)
        & (ycrcb[:, :, 1] <= 173)
        & (ycrcb[:, :, 2] >= 77)
        & (ycrcb[:, :, 2] <= 135)
    )


def _estimate_shirt_color(
    frame_path: Path,
    mask_path: Path,
    config: ShirtColorConfig,
) -> ShirtColorEstimate | None:
    frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if frame_bgr is None:
        raise RuntimeError(f"Could not read frame {frame_path}")
    if mask is None:
        raise RuntimeError(f"Could not read mask {mask_path}")

    foreground = mask > 0
    ys, xs = np.where(foreground)
    if xs.size == 0 or ys.size == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    width = x2 - x1 + 1
    height = y2 - y1 + 1

    yy, xx = np.indices(foreground.shape)
    torso_roi = (
        (xx >= int(round(x1 + config.torso_x_min * width)))
        & (xx <= int(round(x1 + config.torso_x_max * width)))
        & (yy >= int(round(y1 + config.torso_y_min * height)))
        & (yy <= int(round(y1 + config.torso_y_max * height)))
    )
    torso_mask = foreground & torso_roi
    torso_pixel_count = int(np.count_nonzero(torso_mask))
    if torso_pixel_count == 0:
        return None

    skin = _skin_mask(frame_bgr)
    hair = (frame_bgr.mean(axis=2) < 40) & (yy < int(round(y1 + 0.45 * height)))
    shirt_pixels = torso_mask & ~skin & ~hair

    direct_pixel_count = int(np.count_nonzero(shirt_pixels))
    direct_available = direct_pixel_count >= config.min_direct_pixel_count
    if not direct_available:
        shirt_pixels = torso_mask
        direct_pixel_count = int(np.count_nonzero(shirt_pixels))

    rgb = frame_bgr[shirt_pixels][:, ::-1].mean(axis=0)
    confidence = float(direct_pixel_count) / float(max(1, torso_pixel_count))
    return ShirtColorEstimate(
        rgb=rgb.astype(np.float32),
        direct_pixel_count=direct_pixel_count,
        torso_pixel_count=torso_pixel_count,
        confidence=confidence,
        direct_available=direct_available,
    )


def _median_rgb(values: list[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(values, axis=0), axis=0)


def _apply_shirt_color_feature(
    processed_dir: Path,
    payload: dict[str, Any],
    force: bool,
    config: ShirtColorConfig | None = None,
) -> dict[str, Any]:
    config = config or ShirtColorConfig()
    derived_features = payload.setdefault("derived_features", {})
    existing = derived_features.get("shirt_color")
    if existing and existing.get("complete") and not force:
        return {
            "feature": "shirt_color",
            "skipped": True,
            "frame_count": len(payload["positions"]),
            "frames_with_direct_estimate": existing.get("frames_with_direct_estimate"),
        }

    positions = payload["positions"]
    frame_paths = sorted((processed_dir / "segmented_frames").glob("frame*.jpg"), key=_frame_sort_key)
    mask_paths = sorted((processed_dir / "mask_frames").glob("frame*.png"), key=_frame_sort_key)
    if len(frame_paths) != len(positions) or len(mask_paths) != len(positions):
        raise ValueError(
            f"Processed directory {processed_dir} has mismatched frame counts: "
            f"{len(frame_paths)} segmented frames, {len(mask_paths)} masks, {len(positions)} positions"
        )

    direct_colors: list[np.ndarray | None] = []
    local_candidates: list[np.ndarray | None] = []
    direct_available_count = 0
    quality_flagged = [bool(item.get("quality_flagged", False)) for item in positions]

    for position, frame_path, mask_path in zip(positions, frame_paths, mask_paths):
        estimate = _estimate_shirt_color(frame_path, mask_path, config)
        if estimate is None:
            direct_colors.append(None)
            local_candidates.append(None)
            position["shirt_color_rgb"] = None
            position["shirt_color_hex"] = None
            position["shirt_color_source"] = "missing"
            position["shirt_color_direct_available"] = False
            position["shirt_color_confidence"] = 0.0
            position["shirt_color_sample_count"] = 0
            continue

        raw_color = estimate.rgb
        direct_available = estimate.direct_available and not bool(position.get("quality_flagged", False))
        direct_colors.append(raw_color if direct_available else None)
        local_candidates.append(raw_color)
        if direct_available:
            direct_available_count += 1

        position["shirt_color_direct_available"] = direct_available
        position["shirt_color_confidence"] = round(estimate.confidence, 4)
        position["shirt_color_sample_count"] = estimate.direct_pixel_count

    anchor_candidates = [color for color in direct_colors if color is not None]
    if not anchor_candidates:
        anchor_candidates = [color for color in local_candidates if color is not None]
    if not anchor_candidates:
        raise ValueError(f"Could not estimate shirt color for any frame in {processed_dir}")
    global_color = _median_rgb(anchor_candidates)

    window_radius = max(1, config.local_window_size // 2)
    fallback_count = 0
    for idx, position in enumerate(positions):
        local_window_colors = [
            direct_colors[j]
            for j in range(max(0, idx - window_radius), min(len(positions), idx + window_radius + 1))
            if direct_colors[j] is not None
        ]
        if len(local_window_colors) >= config.min_local_window_direct_frames:
            local_color = _median_rgb([color for color in local_window_colors if color is not None])
            fallback_source = "temporal_fill"
        elif direct_colors[idx] is not None:
            local_color = direct_colors[idx]
            fallback_source = "direct"
        else:
            local_color = global_color
            fallback_source = "global_fill"

        if direct_colors[idx] is not None:
            final_rgb = (config.direct_blend_weight * direct_colors[idx]) + (
                (1.0 - config.direct_blend_weight) * local_color
            )
            source = "direct_temporal_blend" if fallback_source == "temporal_fill" else "direct"
        else:
            final_rgb = local_color
            source = fallback_source
            fallback_count += 1

        final_rgb = np.clip(np.round(final_rgb), 0, 255).astype(np.int32)
        position["shirt_color_rgb"] = [int(v) for v in final_rgb]
        position["shirt_color_hex"] = _hex_from_rgb(final_rgb)
        position["shirt_color_source"] = source

    derived_features["shirt_color"] = {
        "complete": True,
        "version": "torso_roi_temporal_smooth_v1",
        "frame_count": len(positions),
        "frames_with_direct_estimate": direct_available_count,
        "frames_using_fallback": fallback_count,
        "global_anchor_rgb": [int(v) for v in np.clip(np.round(global_color), 0, 255)],
        "config": {
            "torso_x_min": config.torso_x_min,
            "torso_x_max": config.torso_x_max,
            "torso_y_min": config.torso_y_min,
            "torso_y_max": config.torso_y_max,
            "min_direct_pixel_count": config.min_direct_pixel_count,
            "min_local_window_direct_frames": config.min_local_window_direct_frames,
            "local_window_size": config.local_window_size,
            "direct_blend_weight": config.direct_blend_weight,
        },
    }
    return {
        "feature": "shirt_color",
        "skipped": False,
        "frame_count": len(positions),
        "frames_with_direct_estimate": direct_available_count,
        "frames_using_fallback": fallback_count,
        "global_anchor_rgb": derived_features["shirt_color"]["global_anchor_rgb"],
    }


def apply_features_to_processed_dir(
    processed_dir: str | Path,
    *,
    feature_names: list[str] | tuple[str, ...] = ("shirt_color",),
    force: bool = False,
    shirt_color_config: ShirtColorConfig | None = None,
) -> dict[str, Any]:
    processed_dir = Path(processed_dir)
    index_json_path = processed_dir / "index_finger_positions.json"
    payload = _load_payload(index_json_path)

    registry: dict[str, FeatureApplier] = {
        "shirt_color": lambda path, current_payload, current_force: _apply_shirt_color_feature(
            path,
            current_payload,
            current_force,
            config=shirt_color_config,
        )
    }

    summaries: list[dict[str, Any]] = []
    for feature_name in feature_names:
        if feature_name not in registry:
            raise ValueError(f"Unknown feature extractor: {feature_name}")
        summaries.append(registry[feature_name](processed_dir, payload, force))

    _write_payload(index_json_path, payload)
    return {
        "processed_dir": str(processed_dir),
        "features": summaries,
    }


def apply_features_to_many(
    input_path: str | Path,
    *,
    feature_names: list[str] | tuple[str, ...] = ("shirt_color",),
    force: bool = False,
    glob_pattern: str = "processed-finger-sam-*",
    shirt_color_config: ShirtColorConfig | None = None,
) -> list[dict[str, Any]]:
    input_path = Path(input_path)
    results: list[dict[str, Any]] = []
    for processed_dir in _resolve_processed_dirs(input_path, glob_pattern):
        results.append(
            apply_features_to_processed_dir(
                processed_dir,
                feature_names=feature_names,
                force=force,
                shirt_color_config=shirt_color_config,
            )
        )
    return results
