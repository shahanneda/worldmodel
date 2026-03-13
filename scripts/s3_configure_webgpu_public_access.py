#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"{name} is required. Example: {name}=my-bucket", file=sys.stderr)
        sys.exit(1)
    return value


def _aws_cli_available() -> bool:
    return shutil.which("aws") is not None


def _aws_base_cmd() -> list[str]:
    cmd = ["aws"]
    if os.environ.get("AWS_PROFILE"):
        cmd += ["--profile", os.environ["AWS_PROFILE"]]
    if os.environ.get("AWS_REGION"):
        cmd += ["--region", os.environ["AWS_REGION"]]
    return cmd


def _run_capture(cmd: list[str], *, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and not allow_failure:
        if result.stdout:
            print(result.stdout.strip(), file=sys.stderr)
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
        sys.exit(result.returncode)
    return result


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_prefix(value: str) -> str:
    return value.strip().strip("/")


def _load_public_access_block(result: subprocess.CompletedProcess[str]) -> dict[str, Any] | None:
    if result.returncode != 0:
        return None
    payload = json.loads(result.stdout or "{}")
    config = payload.get("PublicAccessBlockConfiguration")
    return config if isinstance(config, dict) else None


def _check_account_public_access_block() -> dict[str, Any] | None:
    identity = _run_capture(
        _aws_base_cmd() + ["sts", "get-caller-identity", "--query", "Account", "--output", "text"],
        allow_failure=True,
    )
    if identity.returncode != 0:
        return None
    account_id = identity.stdout.strip()
    if not account_id:
        return None
    result = _run_capture(
        _aws_base_cmd()
        + ["s3control", "get-public-access-block", "--account-id", account_id, "--output", "json"],
        allow_failure=True,
    )
    return _load_public_access_block(result)


def _check_bucket_public_access_block(bucket: str) -> dict[str, Any] | None:
    result = _run_capture(
        _aws_base_cmd() + ["s3api", "get-public-access-block", "--bucket", bucket, "--output", "json"],
        allow_failure=True,
    )
    return _load_public_access_block(result)


def _write_temp_json(payload: Any) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False)
    json.dump(payload, handle, indent=2)
    handle.write("\n")
    handle.close()
    return handle.name


def main() -> int:
    if not _aws_cli_available():
        print(
            "aws CLI not found. Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html",
            file=sys.stderr,
        )
        return 1

    bucket = _require_env("S3_BUCKET")
    prefix = _normalize_prefix(os.environ.get("S3_PREFIX", "worldmodel"))
    object_prefix = _normalize_prefix(
        os.environ.get(
            "WEBGPU_S3_PREFIX",
            f"{prefix}/webgpu-inference/model" if prefix else "webgpu-inference/model",
        )
    )
    allowed_origins = _split_csv(os.environ.get("WEBGPU_CORS_ALLOWED_ORIGINS", "*"))
    allowed_headers = _split_csv(os.environ.get("WEBGPU_CORS_ALLOWED_HEADERS", "*"))
    expose_headers = _split_csv(
        os.environ.get("WEBGPU_CORS_EXPOSE_HEADERS", "ETag,Content-Length,Content-Type")
    )
    max_age_seconds = int(os.environ.get("WEBGPU_CORS_MAX_AGE_SECONDS", "3000"))

    account_block = _check_account_public_access_block()
    bucket_block = _check_bucket_public_access_block(bucket)

    if account_block:
        print(f"account_public_access_block: {json.dumps(account_block, sort_keys=True)}")
    else:
        print("account_public_access_block: not found or not readable")

    if bucket_block:
        print(f"bucket_public_access_block: {json.dumps(bucket_block, sort_keys=True)}")
    else:
        print("bucket_public_access_block: not found")

    if os.environ.get("WEBGPU_DISABLE_BUCKET_PUBLIC_ACCESS_BLOCK", "0") == "1":
        public_access_block = {
            "BlockPublicAcls": False,
            "IgnorePublicAcls": False,
            "BlockPublicPolicy": False,
            "RestrictPublicBuckets": False,
        }
        _run_capture(
            _aws_base_cmd()
            + [
                "s3api",
                "put-public-access-block",
                "--bucket",
                bucket,
                "--public-access-block-configuration",
                json.dumps(public_access_block),
            ]
        )
        print("bucket_public_access_block: updated to allow public bucket policy usage")

    policy_resource = f"arn:aws:s3:::{bucket}/{object_prefix}/*" if object_prefix else f"arn:aws:s3:::{bucket}/*"
    bucket_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AllowPublicReadForWorldmodelWebgpu",
                "Effect": "Allow",
                "Principal": "*",
                "Action": ["s3:GetObject"],
                "Resource": [policy_resource],
            }
        ],
    }
    cors_configuration = {
        "CORSRules": [
            {
                "AllowedOrigins": allowed_origins,
                "AllowedMethods": ["GET", "HEAD"],
                "AllowedHeaders": allowed_headers,
                "ExposeHeaders": expose_headers,
                "MaxAgeSeconds": max_age_seconds,
            }
        ]
    }

    policy_path = _write_temp_json(bucket_policy)
    cors_path = _write_temp_json(cors_configuration)
    try:
        _run_capture(
            _aws_base_cmd()
            + ["s3api", "put-bucket-policy", "--bucket", bucket, "--policy", f"file://{policy_path}"]
        )
        _run_capture(
            _aws_base_cmd()
            + [
                "s3api",
                "put-bucket-cors",
                "--bucket",
                bucket,
                "--cors-configuration",
                f"file://{cors_path}",
            ]
        )
    finally:
        os.unlink(policy_path)
        os.unlink(cors_path)

    print(f"public_read_prefix: {policy_resource}")
    print(f"cors_allowed_origins: {json.dumps(allowed_origins)}")
    print(
        "note: account-level Block Public Access can still override this bucket policy if it remains enabled."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
