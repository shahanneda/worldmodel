#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

import torch


EPOCH_TOKEN_RE = re.compile(r"_epoch\d{4}(?:_|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename checkpoints to include an epoch token when epoch metadata is available."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Only rename checkpoints for this VERSION value.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("model/checkpoints"),
        help="Checkpoint directory to scan.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
        help="Polling interval for watch mode.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one scan and exit.",
    )
    return parser.parse_args()


def epoch_from_payload(payload: dict[str, Any]) -> int | None:
    extra = payload.get("extra", {})
    checkpoint_kind = payload.get("checkpoint_kind")
    if checkpoint_kind == "periodic_epoch":
        epoch = extra.get("epoch")
        return int(epoch) if epoch is not None else None
    if checkpoint_kind == "best":
        epoch = extra.get("best_epoch")
        return int(epoch) if epoch is not None else None
    if checkpoint_kind == "final":
        history = payload.get("history", {})
        train_losses = history.get("train_losses", [])
        if isinstance(train_losses, list) and train_losses:
            return len(train_losses)
        epoch = extra.get("epoch")
        return int(epoch) if epoch is not None else None
    epoch = extra.get("epoch")
    return int(epoch) if epoch is not None else None


def rename_checkpoint(path: Path) -> bool:
    if EPOCH_TOKEN_RE.search(path.stem):
        return False

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return False

    epoch = epoch_from_payload(payload)
    if epoch is None:
        return False

    stem_parts = path.stem.split("_")
    if len(stem_parts) < 2:
        return False
    timestamp = stem_parts[-1]
    prefix = "_".join(stem_parts[:-1])
    target = path.with_name(f"{prefix}_epoch{epoch:04d}_{timestamp}{path.suffix}")
    if target.exists():
        return False
    path.rename(target)
    print(f"renamed {path.name} -> {target.name}", flush=True)
    return True


def scan(root: Path, *, version: str) -> int:
    renamed = 0
    for path in sorted(root.glob(f"*{version}*.pt")):
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            continue
        if not isinstance(payload, dict) or payload.get("VERSION") != version:
            continue
        epoch = epoch_from_payload(payload)
        if epoch is None or EPOCH_TOKEN_RE.search(path.stem):
            continue
        stem_parts = path.stem.split("_")
        if len(stem_parts) < 2:
            continue
        timestamp = stem_parts[-1]
        prefix = "_".join(stem_parts[:-1])
        target = path.with_name(f"{prefix}_epoch{epoch:04d}_{timestamp}{path.suffix}")
        if target.exists():
            continue
        path.rename(target)
        print(f"renamed {path.name} -> {target.name}", flush=True)
        renamed += 1
    return renamed


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.once:
        scan(root, version=args.version)
        return 0

    while True:
        scan(root, version=args.version)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
