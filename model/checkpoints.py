from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_CHECKPOINT_DIR = Path("model/checkpoints")


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    stem: str
    created_at_utc: Optional[datetime]


def checkpoint_dir(root: str | Path = DEFAULT_CHECKPOINT_DIR) -> Path:
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def make_checkpoint_path(
    *,
    run_name: str = "coord_to_image_unet",
    root: str | Path = DEFAULT_CHECKPOINT_DIR,
    suffix: str = ".pt",
) -> Path:
    directory = checkpoint_dir(root)
    safe_run_name = run_name.strip().replace(" ", "_")
    return directory / f"{safe_run_name}_{timestamp_utc()}{suffix}"


def list_checkpoints(
    *,
    root: str | Path = DEFAULT_CHECKPOINT_DIR,
    glob_pattern: str = "*.pt",
) -> list[CheckpointInfo]:
    directory = checkpoint_dir(root)
    infos: list[CheckpointInfo] = []
    for path in sorted(directory.glob(glob_pattern)):
        stem = path.stem
        created_at = None
        maybe_timestamp = stem.split("_")[-1]
        try:
            created_at = datetime.strptime(maybe_timestamp, "%Y-%m-%dT%H-%M-%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            created_at = None
        infos.append(CheckpointInfo(path=path, stem=stem, created_at_utc=created_at))
    return infos


def latest_checkpoint(
    *,
    root: str | Path = DEFAULT_CHECKPOINT_DIR,
    glob_pattern: str = "*.pt",
) -> Optional[Path]:
    checkpoints = list_checkpoints(root=root, glob_pattern=glob_pattern)
    if not checkpoints:
        return None
    with_timestamps = [info for info in checkpoints if info.created_at_utc is not None]
    if with_timestamps:
        return max(with_timestamps, key=lambda info: info.created_at_utc).path
    return checkpoints[-1].path
