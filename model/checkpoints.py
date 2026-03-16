from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


DEFAULT_CHECKPOINT_DIR = Path("model/checkpoints")
COUNTER_FILENAME = ".checkpoint_counter"
CHECKPOINT_ID_PATTERN = re.compile(r"^ckpt(\d+)_")
FAMILY_RUN_NAMES = ("coord_to_image_unet", "pointing_cvae")


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    stem: str
    checkpoint_id: Optional[int]
    created_at_utc: Optional[datetime]


def _checkpoint_sort_key(info: CheckpointInfo) -> tuple[int, int, int, float, str]:
    return (
        1 if info.checkpoint_id is not None else 0,
        info.checkpoint_id if info.checkpoint_id is not None else -1,
        1 if info.created_at_utc is not None else 0,
        info.created_at_utc.timestamp() if info.created_at_utc is not None else float("-inf"),
        info.stem,
    )


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


def checkpoint_family_key(stem: str) -> str:
    family = stem
    checkpoint_id = parse_checkpoint_id(stem)
    if checkpoint_id is not None and "_" in family:
        family = family.split("_", 1)[1]

    maybe_timestamp = stem.split("_")[-1]
    if parse_checkpoint_timestamp(stem) is not None and family.endswith(f"_{maybe_timestamp}"):
        family = family[: -(len(maybe_timestamp) + 1)]

    for run_name in FAMILY_RUN_NAMES:
        if family == run_name:
            return family
        marker = f"_{run_name}"
        marker_index = family.find(marker)
        if marker_index != -1:
            return family[: marker_index + len(marker)]

    return family


def latest_checkpoints_by_family(
    *,
    root: str | Path = DEFAULT_CHECKPOINT_DIR,
    glob_pattern: str = "*.pt",
) -> list[CheckpointInfo]:
    grouped: dict[str, CheckpointInfo] = {}
    for info in list_checkpoints(root=root, glob_pattern=glob_pattern):
        family_key = checkpoint_family_key(info.stem)
        current = grouped.get(family_key)
        if current is None or _checkpoint_sort_key(info) > _checkpoint_sort_key(current):
            grouped[family_key] = info
    return sorted(grouped.values(), key=_checkpoint_sort_key)


def latest_checkpoint(
    *,
    root: str | Path = DEFAULT_CHECKPOINT_DIR,
    glob_pattern: str = "*.pt",
) -> Optional[Path]:
    checkpoints = list_checkpoints(root=root, glob_pattern=glob_pattern)
    if not checkpoints:
        return None

    return max(checkpoints, key=_checkpoint_sort_key).path


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
