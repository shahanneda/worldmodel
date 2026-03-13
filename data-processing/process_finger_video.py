#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)


PERSON_CLASS_ID = 1


@dataclass
class FrameTip:
    x_px: Optional[int]
    y_px: Optional[int]
    x_norm: Optional[float]
    y_norm: Optional[float]
    timestamp_ms: Optional[float]


@dataclass
class CoarsePersonMask:
    mask: np.ndarray
    box: np.ndarray
    score: float


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_landmarks(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected landmarks JSON to be a list of per-frame dicts.")
    return data


def _pick_index_finger_tip(frame_data: Dict[str, Any], width: int, height: int) -> FrameTip:
    hands = frame_data.get("multiHandLandmarks") or []
    timestamp = frame_data.get("t")
    if not hands or len(hands[0]) <= 8:
        return FrameTip(None, None, None, None, timestamp)

    tip = hands[0][8]
    x_norm = float(tip.get("x", 0.0))
    y_norm = float(tip.get("y", 0.0))
    x_px = int(round(np.clip(x_norm, 0.0, 1.0) * (width - 1)))
    y_px = int(round(np.clip(y_norm, 0.0, 1.0) * (height - 1)))
    return FrameTip(x_px, y_px, x_norm, y_norm, timestamp)


def _hand_box(frame_data: Dict[str, Any], width: int, height: int, padding_px: int) -> Optional[np.ndarray]:
    hands = frame_data.get("multiHandLandmarks") or []
    if not hands:
        return None

    pts = []
    for landmark in hands[0]:
        x = int(round(np.clip(float(landmark.get("x", 0.0)), 0.0, 1.0) * (width - 1)))
        y = int(round(np.clip(float(landmark.get("y", 0.0)), 0.0, 1.0) * (height - 1)))
        pts.append((x, y))

    if not pts:
        return None

    xs, ys = zip(*pts)
    x1 = max(0, min(xs) - padding_px)
    y1 = max(0, min(ys) - padding_px)
    x2 = min(width - 1, max(xs) + padding_px)
    y2 = min(height - 1, max(ys) + padding_px)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _mask_box(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _expand_box(box: np.ndarray, width: int, height: int, scale: float) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_w = (x2 - x1) * scale / 2.0
    half_h = (y2 - y1) * scale / 2.0
    return np.array(
        [
            max(0.0, cx - half_w),
            max(0.0, cy - half_h),
            min(float(width - 1), cx + half_w),
            min(float(height - 1), cy + half_h),
        ],
        dtype=np.float32,
    )


def _union_box(*boxes: Optional[np.ndarray]) -> Optional[np.ndarray]:
    valid = [box for box in boxes if box is not None]
    if not valid:
        return None
    stacked = np.stack(valid, axis=0)
    return np.array(
        [
            float(np.min(stacked[:, 0])),
            float(np.min(stacked[:, 1])),
            float(np.max(stacked[:, 2])),
            float(np.max(stacked[:, 3])),
        ],
        dtype=np.float32,
    )


def _iou(box_a: Optional[np.ndarray], box_b: Optional[np.ndarray]) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    flood = mask_u8.copy()
    h, w = flood.shape
    fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, fill_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask_u8, holes)


def _postprocess_mask(
    mask: np.ndarray,
    tip: FrameTip,
    morph_kernel: int,
    width: int,
    height: int,
) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8)) * 255
    kernel_size = max(3, morph_kernel | 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = _fill_holes(mask_u8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8

    target_label = None
    if tip.x_px is not None and tip.y_px is not None:
        label = labels[tip.y_px, tip.x_px]
        if label > 0:
            target_label = int(label)

    if target_label is None:
        areas = stats[1:, cv2.CC_STAT_AREA]
        target_label = 1 + int(np.argmax(areas))

    clean_mask = np.where(labels == target_label, 255, 0).astype(np.uint8)
    min_area = int(width * height * 0.01)
    if np.count_nonzero(clean_mask) < min_area:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        clean_mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    return clean_mask


class PersonMaskModel:
    def __init__(self, device: str, score_threshold: float) -> None:
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.model = maskrcnn_resnet50_fpn_v2(weights=weights)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.score_threshold = score_threshold

    @torch.inference_mode()
    def predict(self, frame_bgr: np.ndarray) -> List[CoarsePersonMask]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(torch.from_numpy(rgb).permute(2, 0, 1))
        output = self.model([tensor.to(self.device)])[0]

        masks = output["masks"].detach().cpu().numpy()
        boxes = output["boxes"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()

        people: List[CoarsePersonMask] = []
        for mask, box, label, score in zip(masks, boxes, labels, scores):
            if int(label) != PERSON_CLASS_ID or float(score) < self.score_threshold:
                continue
            people.append(
                CoarsePersonMask(
                    mask=(mask[0] > 0.5).astype(np.uint8) * 255,
                    box=box.astype(np.float32),
                    score=float(score),
                )
            )
        return people


class SamRefiner:
    def __init__(self, checkpoint_path: str, model_type: str, device: str) -> None:
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def refine(
        self,
        frame_bgr: np.ndarray,
        prompt_box: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        coarse_mask: np.ndarray,
        prev_mask_box: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, float]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=prompt_box[None, :],
            multimask_output=True,
        )

        coarse_box = _mask_box(coarse_mask > 0)
        best_idx = 0
        best_score = -1e9
        for idx, (mask, score) in enumerate(zip(masks, scores)):
            mask_box = _mask_box(mask)
            heuristic = float(score)
            heuristic += 1.2 * _mask_iou(mask.astype(np.uint8), coarse_mask.astype(np.uint8))
            heuristic += 0.5 * _iou(mask_box, coarse_box)
            heuristic += 0.3 * _iou(mask_box, prev_mask_box)
            area_ratio = float(np.count_nonzero(mask)) / float(mask.shape[0] * mask.shape[1])
            if area_ratio < 0.03:
                heuristic -= 2.0
            if area_ratio > 0.75:
                heuristic -= 2.0
            if heuristic > best_score:
                best_idx = idx
                best_score = heuristic
        return masks[best_idx], float(scores[best_idx])


def _select_person_mask(
    candidates: Sequence[CoarsePersonMask],
    width: int,
    height: int,
    tip: FrameTip,
    prev_box: Optional[np.ndarray],
) -> Optional[CoarsePersonMask]:
    if not candidates:
        return None

    frame_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    best_candidate = None
    best_score = -1e9
    for candidate in candidates:
        x1, y1, x2, y2 = candidate.box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = max(1.0, (x2 - x1) * (y2 - y1))
        metric = candidate.score
        metric += min(1.0, area / float(width * height)) * 0.4
        metric -= (np.linalg.norm(np.array([cx, cy]) - frame_center) / max(width, height)) * 0.2
        if prev_box is not None:
            metric += _iou(candidate.box, prev_box) * 0.8
        if tip.x_px is not None and tip.y_px is not None:
            contains_tip = candidate.mask[tip.y_px, tip.x_px] > 0
            metric += 0.8 if contains_tip else -0.6
        if metric > best_score:
            best_score = metric
            best_candidate = candidate
    return best_candidate


def _mask_prompt_points(mask_u8: np.ndarray, tip: FrameTip) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask_u8 > 0)
    coords = np.column_stack([xs, ys])
    if coords.size == 0:
        raise ValueError("Cannot sample prompt points from an empty mask.")

    centroid = coords.mean(axis=0)
    targets = [
        centroid,
        np.array([xs.min(), ys.mean()]),
        np.array([xs.max(), ys.mean()]),
        np.array([xs.mean(), ys.min()]),
        np.array([xs.mean(), ys.max()]),
    ]
    pos_points: List[np.ndarray] = []
    for target in targets:
        distances = ((coords[:, 0] - target[0]) ** 2) + ((coords[:, 1] - target[1]) ** 2)
        pos_points.append(coords[int(np.argmin(distances))].astype(np.float32))

    if tip.x_px is not None and tip.y_px is not None:
        pos_points.append(np.array([tip.x_px, tip.y_px], dtype=np.float32))

    x1, y1, x2, y2 = _mask_box(mask_u8 > 0)
    neg_points = np.array(
        [
            [max(0.0, x1 - 20.0), max(0.0, y1 - 20.0)],
            [min(mask_u8.shape[1] - 1.0, x2 + 20.0), max(0.0, y1 - 20.0)],
            [max(0.0, x1 - 20.0), min(mask_u8.shape[0] - 1.0, y2 + 20.0)],
            [min(mask_u8.shape[1] - 1.0, x2 + 20.0), min(mask_u8.shape[0] - 1.0, y2 + 20.0)],
        ],
        dtype=np.float32,
    )

    point_coords = np.concatenate([np.stack(pos_points), neg_points], axis=0)
    point_labels = np.concatenate(
        [
            np.ones(len(pos_points), dtype=np.int32),
            np.zeros(len(neg_points), dtype=np.int32),
        ]
    )
    return point_coords, point_labels


def _write_video_from_frames(frames_dir: str, out_path: str, fps: float) -> None:
    pattern = os.path.join(frames_dir, "frame%d.jpg")
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


def _draw_debug(
    segmented: np.ndarray,
    mask: np.ndarray,
    coarse_box: Optional[np.ndarray],
    prompt_box: Optional[np.ndarray],
    tip: FrameTip,
    coarse_score: Optional[float],
    sam_score: Optional[float],
    fallback_used: bool,
) -> np.ndarray:
    debug = segmented.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(debug, contours, -1, (0, 180, 0), 2)

    if coarse_box is not None:
        x1, y1, x2, y2 = coarse_box.astype(int)
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 140, 0), 2)
        cv2.putText(
            debug,
            "coarse person mask",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 140, 0),
            2,
            cv2.LINE_AA,
        )

    if prompt_box is not None:
        x1, y1, x2, y2 = prompt_box.astype(int)
        cv2.rectangle(debug, (x1, y1), (x2, y2), (180, 0, 180), 2)

    if tip.x_px is not None and tip.y_px is not None:
        cv2.circle(debug, (tip.x_px, tip.y_px), 8, (0, 0, 255), -1)
        cv2.circle(debug, (tip.x_px, tip.y_px), 14, (255, 255, 255), 2)

    text = f"maskrcnn {coarse_score:.3f}" if coarse_score is not None else "maskrcnn n/a"
    if sam_score is not None:
        text += f" | sam {sam_score:.3f}"
    if fallback_used:
        text += " | coarse fallback"
    cv2.putText(
        debug,
        text,
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        debug,
        text,
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return debug


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment a person with Mask R-CNN + Meta SAM, export frames, and save fingertip debug artifacts."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--landmarks", required=True, help="Input landmarks JSON path.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--sam-checkpoint",
        default="models/sam_vit_b_01ec64.pth",
        help="Path to a Meta Segment Anything checkpoint.",
    )
    parser.add_argument(
        "--sam-model-type",
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h", "default"],
        help="SAM model type for the checkpoint.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--person-score-threshold", type=float, default=0.80)
    parser.add_argument("--detect-every", type=int, default=3)
    parser.add_argument("--box-expand", type=float, default=1.08)
    parser.add_argument("--hand-box-padding", type=int, default=48)
    parser.add_argument("--morph-kernel", type=int, default=11)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
    if not os.path.exists(args.sam_checkpoint):
        raise FileNotFoundError(
            f"SAM checkpoint not found at {args.sam_checkpoint}. Download it before running this script."
        )

    _ensure_dir(args.out)
    segmented_frames_dir = os.path.join(args.out, "segmented_frames")
    debug_frames_dir = os.path.join(args.out, "debug_frames")
    mask_frames_dir = os.path.join(args.out, "mask_frames")
    for path in (segmented_frames_dir, debug_frames_dir, mask_frames_dir):
        _ensure_dir(path)

    landmarks = _load_landmarks(args.landmarks)
    coarse_model = PersonMaskModel(device=args.device, score_threshold=args.person_score_threshold)
    refiner = SamRefiner(
        checkpoint_path=args.sam_checkpoint,
        model_type=args.sam_model_type,
        device=args.device,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    positions: List[Dict[str, Any]] = []
    prev_person: Optional[CoarsePersonMask] = None
    prev_mask_box: Optional[np.ndarray] = None
    last_detection_frame = -args.detect_every
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames is not None and frame_index >= args.max_frames:
            break

        frame_data = landmarks[frame_index] if frame_index < len(landmarks) else {}
        tip = _pick_index_finger_tip(frame_data, width, height)
        hand_box = _hand_box(frame_data, width, height, args.hand_box_padding)

        person = prev_person
        if (frame_index - last_detection_frame) >= args.detect_every or person is None:
            candidates = coarse_model.predict(frame)
            selected = _select_person_mask(
                candidates=candidates,
                width=width,
                height=height,
                tip=tip,
                prev_box=prev_person.box if prev_person is not None else None,
            )
            if selected is not None:
                person = selected
                last_detection_frame = frame_index

        if person is None:
            raise RuntimeError(f"Could not find a person mask on frame {frame_index}.")

        coarse_mask = person.mask
        coarse_box = _expand_box(person.box, width, height, args.box_expand)
        prompt_box = _union_box(coarse_box, hand_box, prev_mask_box)
        if prompt_box is None:
            prompt_box = coarse_box

        point_coords, point_labels = _mask_prompt_points(coarse_mask, tip)
        sam_mask_raw, sam_score = refiner.refine(
            frame_bgr=frame,
            prompt_box=prompt_box,
            point_coords=point_coords,
            point_labels=point_labels,
            coarse_mask=coarse_mask,
            prev_mask_box=prev_mask_box,
        )
        coarse_mask_bool = coarse_mask > 0
        sam_mask_clean = _postprocess_mask(sam_mask_raw, tip, args.morph_kernel, width, height)
        sam_vs_coarse = _mask_iou(sam_mask_clean, coarse_mask)

        fallback_used = False
        final_mask = sam_mask_clean
        if sam_vs_coarse < 0.75:
            final_mask = _postprocess_mask(coarse_mask_bool, tip, args.morph_kernel, width, height)
            fallback_used = True

        white_bg = np.full_like(frame, 255)
        segmented = np.where(final_mask[:, :, None] > 0, frame, white_bg)
        debug = _draw_debug(
            segmented=segmented,
            mask=final_mask,
            coarse_box=coarse_box,
            prompt_box=prompt_box,
            tip=tip,
            coarse_score=person.score,
            sam_score=sam_score,
            fallback_used=fallback_used,
        )

        cv2.imwrite(os.path.join(segmented_frames_dir, f"frame{frame_index}.jpg"), segmented)
        cv2.imwrite(os.path.join(debug_frames_dir, f"frame{frame_index}.jpg"), debug)
        cv2.imwrite(os.path.join(mask_frames_dir, f"frame{frame_index}.png"), final_mask)

        positions.append(
            {
                "frame_index": frame_index,
                "timestamp_ms": tip.timestamp_ms,
                "x_px": tip.x_px,
                "y_px": tip.y_px,
                "x_norm": tip.x_norm,
                "y_norm": tip.y_norm,
                "tip_source": "landmarks_json" if tip.x_px is not None else None,
                "coarse_mask_score": person.score,
                "sam_score": sam_score,
                "sam_vs_coarse_iou": sam_vs_coarse,
                "fallback_used": fallback_used,
            }
        )

        prev_person = person
        prev_mask_box = _mask_box(final_mask > 0)
        frame_index += 1

        if frame_index % 25 == 0:
            print(f"Processed {frame_index} frames...", flush=True)

    cap.release()

    out_json = os.path.join(args.out, "index_finger_positions.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video_path": args.video,
                "landmarks_path": args.landmarks,
                "sam_checkpoint": args.sam_checkpoint,
                "sam_model_type": args.sam_model_type,
                "device": args.device,
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_index,
                "positions": positions,
            },
            f,
            indent=2,
        )

    _write_video_from_frames(segmented_frames_dir, os.path.join(args.out, "segmented_video.mp4"), fps)
    _write_video_from_frames(debug_frames_dir, os.path.join(args.out, "debug_video.mp4"), fps)


if __name__ == "__main__":
    main()
