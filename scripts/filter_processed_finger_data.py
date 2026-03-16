#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from processing.quality_filter import QualityFilterConfig, apply_quality_filter_to_many


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Annotate suspicious segmentation frames in existing processed datasets."
    )
    parser.add_argument(
        "input_path",
        help="Processed dataset directory or a root directory that contains processed datasets.",
    )
    parser.add_argument("--glob-pattern", default="processed-finger*")
    parser.add_argument("--max-foreground-ratio", type=float, default=0.85)
    parser.add_argument("--max-border-touch-ratio", type=float, default=0.75)
    parser.add_argument("--max-area-ratio-vs-window", type=float, default=1.75)
    parser.add_argument("--min-border-touch-for-spike", type=float, default=0.20)
    parser.add_argument("--window-size", type=int, default=15)
    args = parser.parse_args()

    config = QualityFilterConfig(
        max_foreground_ratio=args.max_foreground_ratio,
        max_border_touch_ratio=args.max_border_touch_ratio,
        max_area_ratio_vs_window=args.max_area_ratio_vs_window,
        min_border_touch_for_spike=args.min_border_touch_for_spike,
        window_size=args.window_size,
    )
    results = apply_quality_filter_to_many(
        args.input_path,
        config=config,
        glob_pattern=args.glob_pattern,
    )

    for result in results:
        print(
            f"{result['processed_dir']}: "
            f"flagged {result['flagged_frame_count']} / {result['frame_count']} frames"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
