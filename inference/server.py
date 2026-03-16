from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading

from flask import Flask, jsonify, render_template, request

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.engine import CoordinateToImageInference
from model.checkpoints import DEFAULT_CHECKPOINT_DIR, latest_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR_ABS = REPO_ROOT / DEFAULT_CHECKPOINT_DIR
WEBAPP_ROOT = REPO_ROOT / "inference" / "webapp"
DEFAULT_SAMPLE_COUNT = 4


def resolve_checkpoint_path(checkpoint_path: str | Path | None) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path).expanduser().resolve()

    resolved = latest_checkpoint(root=DEFAULT_CHECKPOINT_DIR_ABS)
    if resolved is None:
        raise FileNotFoundError(
            f"No checkpoint files found in {DEFAULT_CHECKPOINT_DIR_ABS}"
        )
    return resolved.resolve()


def _parse_optional_sample_count(payload: dict[str, object]) -> int:
    value = payload.get("sample_count", DEFAULT_SAMPLE_COUNT)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_count must be an integer.") from exc


def _parse_optional_temperature(payload: dict[str, object]) -> float | None:
    value = payload.get("temperature")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be numeric.") from exc


class InferenceEngineManager:
    def __init__(
        self,
        checkpoint_path: str | Path | None,
        *,
        device: str | None = None,
    ) -> None:
        self._explicit_checkpoint = checkpoint_path is not None
        self._requested_checkpoint = (
            Path(checkpoint_path).expanduser().resolve() if checkpoint_path is not None else None
        )
        self._device = device
        self._lock = threading.Lock()
        self._engine = self._load_engine()

    def _resolve_current_checkpoint_path(self) -> Path:
        if self._explicit_checkpoint:
            if self._requested_checkpoint is None:
                raise FileNotFoundError("Explicit checkpoint path was not set.")
            return self._requested_checkpoint
        return resolve_checkpoint_path(None)

    def _load_engine(self) -> CoordinateToImageInference:
        engine = CoordinateToImageInference(
            self._resolve_current_checkpoint_path(),
            device=self._device,
        )
        engine.warmup()
        return engine

    def get_engine(self) -> CoordinateToImageInference:
        if self._explicit_checkpoint:
            return self._engine

        latest_path = self._resolve_current_checkpoint_path()
        current_path = Path(self._engine.metadata.checkpoint_path).resolve()
        if latest_path == current_path:
            return self._engine

        with self._lock:
            current_path = Path(self._engine.metadata.checkpoint_path).resolve()
            if latest_path != current_path:
                self._engine = self._load_engine()
        return self._engine


def create_app(
    checkpoint_path: str | Path | None = None,
    *,
    device: str | None = None,
) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(WEBAPP_ROOT / "templates"),
        static_folder=str(WEBAPP_ROOT / "static"),
    )
    engine_manager = InferenceEngineManager(checkpoint_path, device=device)

    @app.get("/")
    def index() -> str:
        engine = engine_manager.get_engine()
        metadata = engine.metadata
        return render_template(
            "index.html",
            image_size=metadata.image_size,
            checkpoint_path=metadata.checkpoint_path,
            device=metadata.device,
            model_kind=metadata.model_kind,
            latent_dim=metadata.latent_dim,
            coord_space=metadata.coord_space,
            prior_sample_temperature=metadata.prior_sample_temperature,
        )

    @app.get("/api/health")
    def health() -> tuple[dict[str, object], int]:
        engine = engine_manager.get_engine()
        metadata = engine.metadata
        return (
            {
                "ok": True,
                "device": metadata.device,
                "image_size": metadata.image_size,
                "model_kind": metadata.model_kind,
                "base_channels": metadata.base_channels,
                "latent_channels": metadata.latent_channels,
                "latent_dim": metadata.latent_dim,
                "coord_space": metadata.coord_space,
                "prior_sample_temperature": metadata.prior_sample_temperature,
                "checkpoint_path": metadata.checkpoint_path,
            },
            200,
        )

    @app.post("/api/infer")
    def infer() -> tuple[object, int]:
        engine = engine_manager.get_engine()
        payload = request.get_json(silent=True) or {}
        try:
            x_norm = float(payload["x_norm"])
            y_norm = float(payload["y_norm"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "Expected JSON body with numeric x_norm and y_norm."}), 400

        try:
            sample_count = _parse_optional_sample_count(payload)
            temperature = _parse_optional_temperature(payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            result = engine.generate(
                x_norm,
                y_norm,
                sample_count=sample_count,
                temperature=temperature,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return (
            jsonify(
                {
                    "x_norm": result.x_norm,
                    "y_norm": result.y_norm,
                    "model_x": result.model_x,
                    "model_y": result.model_y,
                    "latency_ms": round(result.latency_ms, 2),
                    "image_base64": result.image_base64,
                    "default_gallery_key": result.default_gallery_key,
                    "gallery": result.gallery,
                    "model_kind": result.model_kind,
                    "coord_space": result.coord_space,
                    "supports_latent_exploration": result.supports_latent_exploration,
                    "sample_count": result.sample_count,
                    "temperature": result.sampling_temperature,
                    "prior_distribution": result.prior_distribution,
                    "uncertainty_heatmap_base64": result.uncertainty_heatmap_base64,
                }
            ),
            200,
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the coordinate-to-image inference web server."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional path to a specific checkpoint. "
            f"Default: latest *.pt file in {DEFAULT_CHECKPOINT_DIR_ABS}, preferring the highest checkpoint ID."
        ),
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind. Use 0.0.0.0 for remote port forwarding.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the inference web server to.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Torch device to use, for example "cuda", "cuda:0", or "cpu". Default: auto.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.checkpoint, device=args.device)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
