#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.checkpoints import latest_checkpoint, list_checkpoints, make_checkpoint_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage local model checkpoints.")
    parser.add_argument(
        "--root",
        default="model/checkpoints",
        help="Checkpoint directory.",
    )
    parser.add_argument(
        "--action",
        choices=["list", "latest", "new-path"],
        default="list",
        help="Checkpoint action to run.",
    )
    parser.add_argument(
        "--run-name",
        default="coord_to_image_unet",
        help="Run name used when generating a new timestamped path.",
    )
    args = parser.parse_args()

    root = Path(args.root)

    if args.action == "list":
        checkpoints = list_checkpoints(root=root)
        if not checkpoints:
            print("No checkpoints found.")
            return 0
        for info in checkpoints:
            ts = info.created_at_utc.isoformat() if info.created_at_utc is not None else "unknown"
            print(f"{info.path} | {ts}")
        return 0

    if args.action == "latest":
        path = latest_checkpoint(root=root)
        if path is None:
            print("No checkpoints found.")
            return 0
        print(path)
        return 0

    if args.action == "new-path":
        print(make_checkpoint_path(root=root, run_name=args.run_name))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
