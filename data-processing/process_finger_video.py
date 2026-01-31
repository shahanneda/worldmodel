#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Optional

import subprocess

import cv2
import numpy as np


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_landmarks(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected landmarks JSON to be a list of per-frame dicts.")
    return data


def _pick_index_finger_tip(
    frame_data: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    hands = frame_data.get("multiHandLandmarks") or []
    if not hands:
        return None
    # MediaPipe Hands landmark index 8 = index finger tip
    hand0 = hands[0]
    if len(hand0) <= 8:
        return None
    return hand0[8]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment video, export frames, and overlay index finger tip."
    )
    parser.add_argument("--video", required=True, help="Input .webm video path")
    parser.add_argument("--landmarks", required=True, help="Input landmarks .json path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--model-selection", type=int, default=0)
    parser.add_argument("--mask-threshold", type=float, default=0.18)
    parser.add_argument("--mask-blur", type=int, default=13)
    parser.add_argument("--mask-dilate", type=int, default=5)
    parser.add_argument("--temporal-smoothing", type=float, default=0.6)
    parser.add_argument("--keep-largest", action="store_true", default=True)
    args = parser.parse_args()

    _ensure_dir(args.out)
    frames_dir = os.path.join(args.out, "segmented_frames")
    _ensure_dir(frames_dir)

    landmarks = _load_landmarks(args.landmarks)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    segmented_video_path = os.path.join(args.out, "segmented_video.mp4")
    debug_video_path = os.path.join(args.out, "debug_video.mp4")

    segmented_frames_dir = frames_dir
    debug_frames_dir = os.path.join(args.out, "debug_frames")
    _ensure_dir(debug_frames_dir)

    # Prefer MediaPipe Selfie Segmentation when available for better quality.
    mp_selfie = None
    try:
        import mediapipe as mp  # type: ignore

        mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=args.model_selection
        )
    except Exception:
        mp_selfie = None

    fgbg = None
    kernel = None
    if mp_selfie is None:
        fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    positions: List[Dict[str, Any]] = []
    frame_index = 0
    prev_mask: Optional[np.ndarray] = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if mp_selfie is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_selfie.process(rgb)
            mask = results.segmentation_mask
            if prev_mask is None:
                prev_mask = mask
            if args.temporal_smoothing > 0:
                mask = (args.temporal_smoothing * prev_mask) + (
                    (1.0 - args.temporal_smoothing) * mask
                )
                prev_mask = mask
            # Smooth + threshold to reduce speckle, then keep largest blob.
            blur_k = max(3, args.mask_blur | 1)
            mask_blur = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
            fg = (mask_blur > args.mask_threshold).astype(np.uint8) * 255
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            if args.mask_dilate > 0:
                fg = cv2.dilate(
                    fg, np.ones((args.mask_dilate, args.mask_dilate), np.uint8), 1
                )
            if args.keep_largest:
                num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
                if num > 1:
                    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    fg = np.where(labels == largest, 255, 0).astype(np.uint8)
            white_bg = np.full_like(frame, 255)
            segmented = np.where(fg[:, :, None] > 0, frame, white_bg)
        else:
            fgmask = fgbg.apply(frame)
            _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)
            white_bg = np.full_like(frame, 255)
            segmented = np.where(fgmask[:, :, None] == 255, frame, white_bg)

        frame_file = os.path.join(segmented_frames_dir, f"frame_{frame_index:05d}.png")
        cv2.imwrite(frame_file, segmented)

        frame_data = landmarks[frame_index] if frame_index < len(landmarks) else {}
        tip = _pick_index_finger_tip(frame_data)

        if tip is not None:
            x_norm = float(tip.get("x", 0.0))
            y_norm = float(tip.get("y", 0.0))
            x_px = int(round(x_norm * width))
            y_px = int(round(y_norm * height))
            positions.append(
                {
                    "frame_index": frame_index,
                    "timestamp_ms": frame_data.get("t"),
                    "x_px": x_px,
                    "y_px": y_px,
                    "x_norm": x_norm,
                    "y_norm": y_norm,
                }
            )
            debug_frame = segmented.copy()
            cv2.circle(debug_frame, (x_px, y_px), 6, (0, 0, 255), -1)
        else:
            positions.append(
                {
                    "frame_index": frame_index,
                    "timestamp_ms": frame_data.get("t"),
                    "x_px": None,
                    "y_px": None,
                    "x_norm": None,
                    "y_norm": None,
                }
            )
            debug_frame = segmented

        debug_file = os.path.join(debug_frames_dir, f"frame_{frame_index:05d}.png")
        cv2.imwrite(debug_file, debug_frame)
        frame_index += 1

    cap.release()
    if mp_selfie is not None:
        mp_selfie.close()

    out_json = os.path.join(args.out, "index_finger_positions.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "video_path": args.video,
                "landmarks_path": args.landmarks,
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_index,
                "positions": positions,
            },
            f,
            indent=2,
        )

    # Encode videos with ffmpeg for broad compatibility.
    def _encode(frames_dir: str, out_path: str) -> None:
        pattern = os.path.join(frames_dir, "frame_%05d.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:.3f}",
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            out_path,
        ]
        subprocess.run(cmd, check=True)

    _encode(segmented_frames_dir, segmented_video_path)
    _encode(debug_frames_dir, debug_video_path)


if __name__ == "__main__":
    main()
