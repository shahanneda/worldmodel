from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


DEFAULT_CHECKPOINT_DIR = Path("model/checkpoints")
COUNTER_FILENAME = ".checkpoint_counter"
CHECKPOINT_ID_PATTERN = re.compile(r"^ckpt(\d+)_")


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    stem: str
    checkpoint_id: Optional[int]
    created_at_utc: Optional[datetime]


def checkpoint_dir(root: str | Path = DEFAULT_CHECKPOINT_DIR) -> Path:
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _counter_path(root: str | Path = DEFAULT_CHECKPOINT_DIR) -> Path:
    return checkpoint_dir(root) / COUNTER_FILENAME


def parse_checkpoint_id(stem: str) -> Optional[int]:
    match = CHECKPOINT_ID_PATTERN.match(stem)
    if match is None:
        return None
    return int(match.group(1))


def parse_checkpoint_timestamp(stem: str) -> Optional[datetime]:
    maybe_timestamp = stem.split("_")[-1]
    try:
        return datetime.strptime(maybe_timestamp, "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _initial_counter_value(root: str | Path = DEFAULT_CHECKPOINT_DIR) -> int:
    directory = checkpoint_dir(root)
    checkpoint_files = list(directory.glob("*.pt"))
    highest_existing_id = 0
    for path in checkpoint_files:
        checkpoint_id = parse_checkpoint_id(path.stem)
        if checkpoint_id is not None:
            highest_existing_id = max(highest_existing_id, checkpoint_id)
    return max(highest_existing_id, len(checkpoint_files))


def next_checkpoint_id(root: str | Path = DEFAULT_CHECKPOINT_DIR) -> int:
    counter_file = _counter_path(root)
    if counter_file.exists():
        current_value = int(counter_file.read_text(encoding="utf-8").strip())
    else:
        current_value = _initial_counter_value(root)
    next_value = current_value + 1
    counter_file.write_text(f"{next_value}\n", encoding="utf-8")
    return next_value


def checkpoint_prefix(checkpoint_id: int) -> str:
    return f"ckpt{checkpoint_id:06d}"


def safe_checkpoint_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    token = token.strip("._-")
    return token or "unnamed"


def format_checkpoint_label(checkpoint_id: Optional[int]) -> str:
    if checkpoint_id is None:
        return "no-id"
    return checkpoint_prefix(checkpoint_id)


def make_checkpoint_path(
    *,
    run_name: str = "coord_to_image_unet",
    version: str | None = None,
    root: str | Path = DEFAULT_CHECKPOINT_DIR,
    filename_template: str | None = None,
    suffix: str = ".pt",
) -> Path:
    directory = checkpoint_dir(root)
    safe_run_name = safe_checkpoint_token(run_name)
    safe_version = safe_checkpoint_token(version) if version is not None else None
    checkpoint_id = next_checkpoint_id(root)
    timestamp = timestamp_utc()

    template = filename_template
    if template is None:
        template = (
            "{checkpoint_id}_{VERSION}_{run_name}_{timestamp}"
            if safe_version is not None
            else "{checkpoint_id}_{run_name}_{timestamp}"
        )
    filename = template.format(
        checkpoint_id=checkpoint_prefix(checkpoint_id),
        VERSION=safe_version or "no-version",
        run_name=safe_run_name,
        timestamp=timestamp,
    )
    if suffix and not filename.endswith(suffix):
        filename = f"{filename}{suffix}"
    return directory / filename


def checkpoint_glob_pattern(
    *,
    run_name: str | None = None,
    version: str | None = None,
    suffix: str = ".pt",
) -> str:
    parts: list[str] = []
    if version:
        parts.append(safe_checkpoint_token(version))
    if run_name:
        parts.append(safe_checkpoint_token(run_name))
    if not parts:
        return f"*{suffix}"
    return f"*{'*'.join(parts)}*{suffix}"


def list_checkpoints(
    *,
    root: str | Path = DEFAULT_CHECKPOINT_DIR,
    glob_pattern: str = "*.pt",
) -> list[CheckpointInfo]:
    directory = checkpoint_dir(root)
    infos: list[CheckpointInfo] = []
    for path in sorted(directory.glob(glob_pattern)):
        stem = path.stem
        infos.append(
            CheckpointInfo(
                path=path,
                stem=stem,
                checkpoint_id=parse_checkpoint_id(stem),
                created_at_utc=parse_checkpoint_timestamp(stem),
            )
        )
    return infos


def latest_checkpoint(
    *,
    root: str | Path = DEFAULT_CHECKPOINT_DIR,
    glob_pattern: str = "*.pt",
) -> Optional[Path]:
    checkpoints = list_checkpoints(root=root, glob_pattern=glob_pattern)
    if not checkpoints:
        return None

    with_ids = [info for info in checkpoints if info.checkpoint_id is not None]
    if with_ids:
        return max(with_ids, key=lambda info: info.checkpoint_id).path

    with_timestamps = [info for info in checkpoints if info.created_at_utc is not None]
    if with_timestamps:
        return max(with_timestamps, key=lambda info: info.created_at_utc).path

    return checkpoints[-1].path


def find_checkpoint_by_id(
    checkpoint_id: int,
    *,
    root: str | Path = DEFAULT_CHECKPOINT_DIR,
    glob_pattern: str = "*.pt",
) -> Optional[Path]:
    for info in list_checkpoints(root=root, glob_pattern=glob_pattern):
        if info.checkpoint_id == checkpoint_id:
            return info.path
    return None
