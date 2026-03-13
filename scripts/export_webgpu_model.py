from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxconverter_common import float16 as onnx_float16

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.engine import CoordinateToImageInference
from model.checkpoints import latest_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "webgpu-inference" / "model"
DEFAULT_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class ExportArtifacts:
    graph_path: Path
    external_data_paths: list[Path]
    image_size: int
    precision: str
    opset: int


def default_checkpoint_path() -> Path:
    latest = latest_checkpoint(
        root=REPO_ROOT / "model" / "checkpoints",
        glob_pattern="coord_to_image_unet*.pt",
    )
    if latest is not None:
        return latest
    return REPO_ROOT / "model" / "checkpoints" / "coord_to_image_unet.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the coordinate-to-image checkpoint for WebGPU browser inference."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint_path(),
        help="Checkpoint to export. Defaults to the latest coord_to_image_unet checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the generated ONNX files and manifest. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--manifest-name",
        default=DEFAULT_MANIFEST_NAME,
        help=f"Manifest filename to write inside the output directory. Default: {DEFAULT_MANIFEST_NAME}",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help="Base name for the exported ONNX files. Defaults to '<checkpoint-stem>_webgpu_<precision>'.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp16",
        help="Export precision for the browser model. Default: fp16.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version. Default: 18.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Torch device used during export. Default: "cpu".',
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the ONNX Runtime parity check after export.",
    )
    return parser.parse_args()


def _safe_unlink(path: Path) -> None:
    if path.exists():
        path.unlink()


def _repo_relative_display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _export_fp32_graph(
    model: torch.nn.Module,
    graph_path: Path,
    *,
    opset: int,
) -> list[Path]:
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    _safe_unlink(graph_path)
    _safe_unlink(graph_path.with_suffix(graph_path.suffix + ".data"))

    sample_coords = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    torch.onnx.export(
        model,
        sample_coords,
        graph_path,
        input_names=["coords"],
        output_names=["image"],
        opset_version=opset,
        dynamo=True,
        external_data=True,
        do_constant_folding=True,
        verify=False,
    )

    external_data_path = graph_path.with_suffix(graph_path.suffix + ".data")
    return [external_data_path] if external_data_path.exists() else []


def _convert_graph_to_fp16(
    source_graph_path: Path,
    target_graph_path: Path,
) -> list[Path]:
    _safe_unlink(target_graph_path)
    _safe_unlink(target_graph_path.with_suffix(target_graph_path.suffix + ".data"))

    model = onnx.load_model(source_graph_path, load_external_data=True)
    model_fp16 = onnx_float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save_model(
        model_fp16,
        target_graph_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=target_graph_path.name + ".data",
        size_threshold=1024,
        convert_attribute=False,
    )

    external_data_path = target_graph_path.with_suffix(target_graph_path.suffix + ".data")
    return [external_data_path] if external_data_path.exists() else []


def export_checkpoint(
    checkpoint_path: Path,
    *,
    output_dir: Path,
    basename: str | None,
    precision: str,
    opset: int,
    device: str,
) -> tuple[CoordinateToImageInference, ExportArtifacts]:
    engine = CoordinateToImageInference(checkpoint_path, device=device)
    model = engine.model.eval().cpu()

    base = basename or f"{checkpoint_path.stem}_webgpu_{precision}"
    graph_path = output_dir / f"{base}.onnx"

    if precision == "fp32":
        external_data_paths = _export_fp32_graph(model, graph_path, opset=opset)
    else:
        with tempfile.TemporaryDirectory(prefix="worldmodel-webgpu-export-") as tmp_dir:
            tmp_graph_path = Path(tmp_dir) / f"{base}_fp32_tmp.onnx"
            _export_fp32_graph(model, tmp_graph_path, opset=opset)
            external_data_paths = _convert_graph_to_fp16(tmp_graph_path, graph_path)

    return (
        engine,
        ExportArtifacts(
            graph_path=graph_path,
            external_data_paths=external_data_paths,
            image_size=engine.metadata.image_size,
            precision=precision,
            opset=opset,
        ),
    )


def verify_export(
    engine: CoordinateToImageInference,
    artifacts: ExportArtifacts,
) -> dict[str, Any]:
    coords = np.array([[0.25, 0.75]], dtype=np.float32)
    with torch.inference_mode():
        torch_output = engine.model(torch.from_numpy(coords)).detach().cpu().numpy()

    session = ort.InferenceSession(
        artifacts.graph_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    onnx_output = session.run(["image"], {"coords": coords})[0]

    return {
        "coords": coords[0].tolist(),
        "max_abs_diff": float(np.max(np.abs(torch_output - onnx_output))),
        "mean_abs_diff": float(np.mean(np.abs(torch_output - onnx_output))),
    }


def build_manifest(
    checkpoint_path: Path,
    artifacts: ExportArtifacts,
    verification: dict[str, Any] | None,
) -> dict[str, Any]:
    total_bytes = artifacts.graph_path.stat().st_size + sum(
        path.stat().st_size for path in artifacts.external_data_paths
    )
    external_entries = []
    for path in artifacts.external_data_paths:
        external_entries.append(
            {
                "path": path.name,
                "url": f"./{path.name}",
                "bytes": path.stat().st_size,
            }
        )

    return {
        "modelName": "CoordinateToImageUNet",
        "sourceCheckpoint": _repo_relative_display(checkpoint_path),
        "createdAtUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "precision": artifacts.precision,
        "opset": artifacts.opset,
        "imageSize": artifacts.image_size,
        "graphUrl": f"./{artifacts.graph_path.name}",
        "graphBytes": artifacts.graph_path.stat().st_size,
        "externalData": external_entries,
        "totalModelBytes": total_bytes,
        "input": {
            "name": "coords",
            "dtype": "float32",
            "shape": [1, 2],
        },
        "output": {
            "name": "image",
            "dtype": "float32",
            "shape": [1, 3, artifacts.image_size, artifacts.image_size],
        },
        "preferredExecutionProviders": ["webgpu", "wasm"],
        "verification": verification,
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    engine, artifacts = export_checkpoint(
        checkpoint_path,
        output_dir=output_dir,
        basename=args.basename,
        precision=args.precision,
        opset=args.opset,
        device=args.device,
    )

    verification = None if args.skip_verify else verify_export(engine, artifacts)
    manifest_path = output_dir / args.manifest_name
    manifest = build_manifest(checkpoint_path, artifacts, verification)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"checkpoint: {_repo_relative_display(checkpoint_path)}")
    print(f"graph: {_repo_relative_display(artifacts.graph_path)}")
    for external_path in artifacts.external_data_paths:
        print(f"external_data: {_repo_relative_display(external_path)}")
    print(f"manifest: {_repo_relative_display(manifest_path)}")
    print(f"precision: {artifacts.precision}")
    print(f"image_size: {artifacts.image_size}")
    print(f"total_model_bytes: {manifest['totalModelBytes']}")
    if verification is not None:
        print(
            "verification:"
            f" max_abs_diff={verification['max_abs_diff']:.9f}"
            f" mean_abs_diff={verification['mean_abs_diff']:.9f}"
        )


if __name__ == "__main__":
    main()
