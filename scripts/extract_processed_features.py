#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from processing.derived_features import ShirtColorConfig, apply_features_to_many


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute derived features from existing processed datasets without rerunning segmentation."
    )
    parser.add_argument(
        "input_path",
        help="Processed dataset directory or a root directory containing processed datasets.",
    )
    parser.add_argument(
        "--feature",
        dest="features",
        action="append",
        choices=["shirt_color"],
        help="Derived feature to compute. Repeat to request multiple features. Defaults to shirt_color.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute features even if already present.")
    parser.add_argument("--glob-pattern", default="processed-finger-sam-*")
    parser.add_argument("--shirt-min-direct-pixel-count", type=int, default=1500)
    parser.add_argument("--shirt-local-window-size", type=int, default=11)
    args = parser.parse_args()

    config = ShirtColorConfig(
        min_direct_pixel_count=args.shirt_min_direct_pixel_count,
        local_window_size=args.shirt_local_window_size,
    )
    results = apply_features_to_many(
        args.input_path,
        feature_names=args.features or ["shirt_color"],
        force=args.force,
        glob_pattern=args.glob_pattern,
        shirt_color_config=config,
    )
    for result in results:
        feature_summaries = ", ".join(
            f"{feature['feature']}({'skipped' if feature['skipped'] else 'updated'})"
            for feature in result["features"]
        )
        print(f"{result['processed_dir']}: {feature_summaries}")
        for feature in result["features"]:
            if feature["feature"] == "shirt_color" and not feature["skipped"]:
                print(
                    "  "
                    f"direct={feature['frames_with_direct_estimate']} "
                    f"fallback={feature['frames_using_fallback']} "
                    f"anchor_rgb={feature['global_anchor_rgb']}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
