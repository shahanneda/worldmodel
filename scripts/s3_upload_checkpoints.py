#!/usr/bin/env python3
import argparse
import fnmatch
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, List

import botocore.session
from s3transfer.manager import TransferConfig, TransferManager

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.checkpoints import latest_checkpoints_by_family


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"{name} is required. Example: {name}=my-bucket", file=sys.stderr)
        sys.exit(1)
    return value


def _aws_cli_available() -> bool:
    return shutil.which("aws") is not None


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload local checkpoints to S3.")
    parser.add_argument(
        "--selection",
        choices=("all", "latest-per-family"),
        default=os.environ.get("S3_SELECTION", "all").strip() or "all",
        help=(
            "Which checkpoints to upload. "
            "'all' preserves the original sync behavior. "
            "'latest-per-family' uploads only the newest checkpoint in each logical family."
        ),
    )
    return parser.parse_args()


def _checkpoint_files_for_selection(
    checkpoints_dir: Path,
    *,
    selection: str,
) -> list[Path]:
    if selection == "latest-per-family":
        return [info.path.resolve() for info in latest_checkpoints_by_family(root=checkpoints_dir)]
    return sorted(path.resolve() for path in checkpoints_dir.iterdir() if path.is_file())


def _apply_excludes(paths: Iterable[Path], *, root: Path, excludes: list[str]) -> list[Path]:
    if not excludes:
        return list(paths)

    kept: list[Path] = []
    for path in paths:
        rel_path = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(path.name, pattern) for pattern in excludes):
            continue
        kept.append(path)
    return kept


def _make_s3_client() -> object:
    session = botocore.session.get_session()
    if os.environ.get("AWS_PROFILE"):
        session.set_config_variable("profile", os.environ["AWS_PROFILE"])
    if os.environ.get("AWS_REGION"):
        session.set_config_variable("region", os.environ["AWS_REGION"])
    return session.create_client("s3")


def _list_remote_sizes(client: object, *, bucket: str, prefix: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    kwargs = {"Bucket": bucket, "Prefix": prefix}
    while True:
        response = client.list_objects_v2(**kwargs)
        for item in response.get("Contents", []):
            key = item["Key"]
            relative = key[len(prefix):] if key.startswith(prefix) else key
            sizes[relative] = int(item["Size"])
        if not response.get("IsTruncated"):
            break
        kwargs["ContinuationToken"] = response["NextContinuationToken"]
    return sizes


def _upload_with_python(
    *,
    bucket: str,
    checkpoints_dir: Path,
    prefix: str,
    files: list[Path],
) -> int:
    client = _make_s3_client()
    remote_sizes = _list_remote_sizes(client, bucket=bucket, prefix=prefix)
    pending: list[Path] = []
    for path in files:
        rel_path = path.relative_to(checkpoints_dir).as_posix()
        if remote_sizes.get(rel_path) == path.stat().st_size:
            print(f"Skipping unchanged: {rel_path}")
            continue
        pending.append(path)

    total_bytes = sum(path.stat().st_size for path in pending)
    print(
        f"Uploading {len(pending)} of {len(files)} file(s) "
        f"({total_bytes / (1024 ** 3):.2f} GiB) to s3://{bucket}/{prefix}"
    )
    if not pending:
        print("Nothing to upload.")
        return 0

    config = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,
        multipart_chunksize=16 * 1024 * 1024,
        max_request_concurrency=8,
    )
    manager = TransferManager(client, config=config)
    try:
        for index, path in enumerate(pending, start=1):
            rel_path = path.relative_to(checkpoints_dir).as_posix()
            key = f"{prefix}{rel_path}"
            size_gib = path.stat().st_size / (1024 ** 3)
            print(f"[{index}/{len(pending)}] Uploading {rel_path} ({size_gib:.2f} GiB)")
            with path.open("rb") as file_obj:
                manager.upload(file_obj, bucket, key).result()
        print("Done.")
        return 0
    finally:
        manager.shutdown()


def main() -> int:
    args = _parse_args()

    bucket = _require_env("S3_BUCKET")
    checkpoints_dir = Path(os.environ.get("CHECKPOINTS_DIR", "model/checkpoints").strip()).resolve()
    if not checkpoints_dir.is_dir():
        print(f"Checkpoints directory not found: {checkpoints_dir}", file=sys.stderr)
        return 1

    prefix = os.environ.get("S3_PREFIX", "worldmodel").strip()
    if prefix:
        prefix = prefix.rstrip("/") + "/"

    s3_uri = f"s3://{bucket}/{prefix}checkpoints/"
    excludes = _split_csv(os.environ.get("S3_EXCLUDE", ""))

    if args.selection == "all" and _aws_cli_available():
        cmd = ["aws"]
        if os.environ.get("AWS_PROFILE"):
            cmd += ["--profile", os.environ["AWS_PROFILE"]]
        if os.environ.get("AWS_REGION"):
            cmd += ["--region", os.environ["AWS_REGION"]]

        cmd += ["s3", "sync", f"{checkpoints_dir}/", s3_uri, "--only-show-errors"]

        if os.environ.get("S3_DELETE") == "1":
            cmd.append("--delete")

        for ex in excludes:
            cmd += ["--exclude", ex]

        print(f"Uploading {checkpoints_dir}/ -> {s3_uri}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            return result.returncode

        print("Done.")
        return 0

    if args.selection == "all" and not _aws_cli_available():
        print(
            "aws CLI not found, falling back to Python multipart upload.",
            file=sys.stderr,
        )

    selected_files = _checkpoint_files_for_selection(
        checkpoints_dir,
        selection=args.selection,
    )
    selected_files = _apply_excludes(selected_files, root=checkpoints_dir, excludes=excludes)
    return _upload_with_python(
        bucket=bucket,
        checkpoints_dir=checkpoints_dir,
        prefix=f"{prefix}checkpoints/",
        files=selected_files,
    )


if __name__ == "__main__":
    raise SystemExit(main())
