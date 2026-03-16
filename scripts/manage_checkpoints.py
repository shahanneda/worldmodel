#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.checkpoints import (
    find_checkpoint_by_id,
    format_checkpoint_label,
    latest_checkpoint,
    list_checkpoints,
    make_checkpoint_path,
    checkpoint_glob_pattern,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage local model checkpoints.")
    parser.add_argument(
        "--root",
        default="model/checkpoints",
        help="Checkpoint directory.",
    )
    parser.add_argument(
        "--action",
        choices=["list", "latest", "new-path", "resolve-id"],
        default="list",
        help="Checkpoint action to run.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name used for filtering. For --action new-path, defaults to coord_to_image_unet.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Optional VERSION token used when generating or filtering checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-id",
        type=int,
        default=None,
        help="Numeric checkpoint ID used with --action resolve-id.",
    )
    args = parser.parse_args()

    root = Path(args.root)

    if args.action == "list":
        glob_pattern = checkpoint_glob_pattern(run_name=args.run_name, version=args.version)
        checkpoints = list_checkpoints(root=root, glob_pattern=glob_pattern)
        if not checkpoints:
            print("No checkpoints found.")
            return 0
        for info in checkpoints:
            ckpt_id = format_checkpoint_label(info.checkpoint_id)
            ts = info.created_at_utc.isoformat() if info.created_at_utc is not None else "unknown"
            print(f"{ckpt_id} | {info.path} | {ts}")
        return 0

    if args.action == "latest":
        path = latest_checkpoint(
            root=root,
            glob_pattern=checkpoint_glob_pattern(run_name=args.run_name, version=args.version),
        )
        if path is None:
            print("No checkpoints found.")
            return 0
        print(path)
        return 0

    if args.action == "new-path":
        print(
            make_checkpoint_path(
                root=root,
                run_name=args.run_name or "coord_to_image_unet",
                version=args.version,
            )
        )
        return 0

    if args.action == "resolve-id":
        if args.checkpoint_id is None:
            parser.error("--checkpoint-id is required for --action resolve-id")
        path = find_checkpoint_by_id(args.checkpoint_id, root=root)
        if path is None:
            print(f"Checkpoint ID not found: {format_checkpoint_label(args.checkpoint_id)}")
            return 1
        print(path)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
