#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.config import load_training_config
from model.train import run_training_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model training from a YAML config."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the training config YAML.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a config value using dotted.path=value. Repeat as needed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate the config, print the summary, and exit without training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_training_config(args.config, overrides=args.set)

    print("training config summary:")
    for line in config.summary_lines():
        print(line)

    if args.dry_run:
        return 0

    result = run_training_from_config(config)
    print(f"device: {result.device}")
    print(f"dataset_size: {len(result.dataset)}")
    print(f"train_batches: {len(result.train_loader)}")
    print(f"val_batches: {len(result.val_loader)}")
    print(f"test_batches: {len(result.test_loader) if result.test_loader is not None else 0}")
    print(f"resumed_from_checkpoint: {result.resumed_from_checkpoint}")
    print(f"split_artifact_path: {result.split_artifact_path}")
    print(f"resolved_config_path: {result.resolved_config_path}")
    print(f"final_checkpoint_path: {result.final_checkpoint_path}")
    print(f"best_checkpoint_path: {result.best_checkpoint_path}")
    print(f"wandb_run_id: {result.wandb_run_id}")
    print(f"wandb_run_name: {result.wandb_run_name}")
    print(f"wandb_run_url: {result.wandb_run_url}")
    print(f"train_losses: {result.history.train_losses}")
    print(f"val_losses: {result.history.val_losses}")
    print(f"test_losses: {result.history.test_losses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
