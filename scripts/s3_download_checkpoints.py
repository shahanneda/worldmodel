#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from typing import List


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


def main() -> int:
    if not _aws_cli_available():
        print(
            "aws CLI not found. Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html",
            file=sys.stderr,
        )
        return 1

    bucket = _require_env("S3_BUCKET")
    checkpoints_dir = os.environ.get("CHECKPOINTS_DIR", "model/checkpoints").strip()
    os.makedirs(checkpoints_dir, exist_ok=True)

    prefix = os.environ.get("S3_PREFIX", "worldmodel").strip()
    if prefix:
        prefix = prefix.rstrip("/") + "/"

    s3_uri = f"s3://{bucket}/{prefix}checkpoints/"

    cmd = ["aws"]
    if os.environ.get("AWS_PROFILE"):
        cmd += ["--profile", os.environ["AWS_PROFILE"]]
    if os.environ.get("AWS_REGION"):
        cmd += ["--region", os.environ["AWS_REGION"]]

    cmd += ["s3", "sync", s3_uri, f"{checkpoints_dir}/", "--only-show-errors"]

    if os.environ.get("S3_DELETE") == "1":
        cmd.append("--delete")

    excludes = _split_csv(os.environ.get("S3_EXCLUDE", ""))
    for ex in excludes:
        cmd += ["--exclude", ex]

    print(f"Downloading {s3_uri} -> {checkpoints_dir}/")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return result.returncode

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
