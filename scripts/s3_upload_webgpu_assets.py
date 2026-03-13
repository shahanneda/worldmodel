#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse


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


def _run_capture(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout.strip(), file=sys.stderr)
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
        sys.exit(result.returncode)
    return result.stdout.strip()


def _normalize_prefix(value: str) -> str:
    return value.strip().strip("/")


def _bucket_region(bucket: str) -> str:
    payload = _run_capture(
        _aws_base_cmd() + ["s3api", "get-bucket-location", "--bucket", bucket, "--output", "json"]
    )
    location = json.loads(payload).get("LocationConstraint")
    if location in (None, "", "null", "None"):
        return "us-east-1"
    if location == "EU":
        return "eu-west-1"
    return str(location)


def _join_key(prefix: str, name: str) -> str:
    return f"{prefix}/{name}" if prefix else name


def _public_root_url(bucket: str, region: str, key_prefix: str) -> str:
    custom = os.environ.get("WEBGPU_PUBLIC_ROOT_URL", "").strip()
    if custom:
        return custom.rstrip("/")

    if region == "us-east-1":
        root = f"https://{bucket}.s3.amazonaws.com"
    else:
        root = f"https://{bucket}.s3.{region}.amazonaws.com"

    if key_prefix:
        return f"{root}/{quote(key_prefix, safe='/')}"
    return root


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _local_path_from_manifest_url(base_dir: Path, url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme or parsed.netloc:
        raise ValueError(
            "Upload script expects a local manifest with relative artifact URLs, "
            f"but found absolute URL: {url}"
        )
    candidate = (base_dir / parsed.path).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Missing local artifact referenced by manifest: {candidate}")
    return candidate


def _write_runtime_config(runtime_config_path: Path, manifest_url: str) -> None:
    runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "window.WORLDMODEL_WEBGPU_CONFIG = {\n"
        f"  manifestUrl: {json.dumps(manifest_url)},\n"
        "};\n"
    )
    runtime_config_path.write_text(content, encoding="utf-8")


def _upload_file(local_path: Path, bucket: str, key: str, *, cache_control: str | None = None) -> None:
    cmd = _aws_base_cmd() + [
        "s3",
        "cp",
        str(local_path),
        f"s3://{bucket}/{key}",
        "--only-show-errors",
    ]
    if cache_control:
        cmd += ["--cache-control", cache_control]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> int:
    if not _aws_cli_available():
        print(
            "aws CLI not found. Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html",
            file=sys.stderr,
        )
        return 1

    bucket = _require_env("S3_BUCKET")
    model_dir = Path(os.environ.get("WEBGPU_MODEL_DIR", "webgpu-inference/model")).resolve()
    manifest_name = os.environ.get("WEBGPU_MANIFEST_NAME", "manifest.json").strip() or "manifest.json"
    manifest_path = model_dir / manifest_name
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    prefix = _normalize_prefix(os.environ.get("S3_PREFIX", "worldmodel"))
    upload_prefix = _normalize_prefix(
        os.environ.get(
            "WEBGPU_S3_PREFIX",
            f"{prefix}/webgpu-inference/model" if prefix else "webgpu-inference/model",
        )
    )
    runtime_config_path = Path(
        os.environ.get("WEBGPU_RUNTIME_CONFIG_PATH", "webgpu-inference/runtime-config.js")
    ).resolve()

    manifest = _load_json(manifest_path)
    graph_url = manifest.get("graphUrl")
    if not isinstance(graph_url, str) or not graph_url:
        print(f"Manifest is missing graphUrl: {manifest_path}", file=sys.stderr)
        return 1

    graph_local_path = _local_path_from_manifest_url(manifest_path.parent, graph_url)
    external_entries = manifest.get("externalData", [])
    if not isinstance(external_entries, list):
        print(f"Manifest externalData must be a list: {manifest_path}", file=sys.stderr)
        return 1

    region = _bucket_region(bucket)
    public_root = _public_root_url(bucket, region, upload_prefix)
    if "." in bucket and not os.environ.get("WEBGPU_PUBLIC_ROOT_URL"):
        print(
            "Warning: bucket names with dots can be awkward with virtual-hosted HTTPS URLs. "
            "If you hit TLS/hostname issues, use WEBGPU_PUBLIC_ROOT_URL with CloudFront or another host.",
            file=sys.stderr,
        )

    graph_key = _join_key(upload_prefix, graph_local_path.name)
    public_graph_url = f"{public_root}/{quote(graph_local_path.name)}"

    rewritten_manifest = dict(manifest)
    rewritten_manifest["graphUrl"] = public_graph_url
    rewritten_manifest["externalData"] = []

    _upload_file(
        graph_local_path,
        bucket,
        graph_key,
        cache_control="public, max-age=31536000, immutable",
    )

    for entry in external_entries:
        if not isinstance(entry, dict):
            print("Manifest externalData entries must be objects for S3 upload.", file=sys.stderr)
            return 1
        url = entry.get("url")
        if not isinstance(url, str) or not url:
            print("Manifest externalData entry missing url.", file=sys.stderr)
            return 1
        path_name = str(entry.get("path") or Path(urlparse(url).path).name)
        local_path = _local_path_from_manifest_url(manifest_path.parent, url)
        key = _join_key(upload_prefix, local_path.name)
        public_url = f"{public_root}/{quote(local_path.name)}"
        _upload_file(
            local_path,
            bucket,
            key,
            cache_control="public, max-age=31536000, immutable",
        )
        rewritten_manifest["externalData"].append(
            {
                "path": path_name,
                "url": public_url,
                "bytes": entry.get("bytes"),
            }
        )

    manifest_key = _join_key(upload_prefix, manifest_name)
    manifest_url = f"{public_root}/{quote(manifest_name)}"

    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump(rewritten_manifest, handle, indent=2)
        handle.write("\n")
        temp_manifest_path = Path(handle.name)

    try:
        _upload_file(
            temp_manifest_path,
            bucket,
            manifest_key,
            cache_control="no-store",
        )
    finally:
        temp_manifest_path.unlink(missing_ok=True)

    if os.environ.get("WEBGPU_WRITE_RUNTIME_CONFIG", "1") != "0":
        _write_runtime_config(runtime_config_path, manifest_url)
        print(f"runtime_config: {runtime_config_path}")

    print(f"bucket_region: {region}")
    print(f"model_prefix: s3://{bucket}/{upload_prefix}/")
    print(f"manifest_url: {manifest_url}")
    print(f"graph_url: {public_graph_url}")
    for entry in rewritten_manifest["externalData"]:
        print(f"external_url: {entry['url']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
