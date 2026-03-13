from __future__ import annotations

import argparse
from pathlib import Path
import sys

from flask import Flask, jsonify, render_template, request

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.engine import CoordinateToImageInference


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "model" / "checkpoints" / "coord_to_image_unet.pt"
WEBAPP_ROOT = REPO_ROOT / "inference" / "webapp"


def create_app(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    *,
    device: str | None = None,
) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(WEBAPP_ROOT / "templates"),
        static_folder=str(WEBAPP_ROOT / "static"),
    )
    engine = CoordinateToImageInference(checkpoint_path, device=device)
    engine.warmup()

    @app.get("/")
    def index() -> str:
        metadata = engine.metadata
        return render_template(
            "index.html",
            image_size=metadata.image_size,
            checkpoint_path=metadata.checkpoint_path,
            device=metadata.device,
        )

    @app.get("/api/health")
    def health() -> tuple[dict[str, object], int]:
        metadata = engine.metadata
        return (
            {
                "ok": True,
                "device": metadata.device,
                "image_size": metadata.image_size,
                "checkpoint_path": metadata.checkpoint_path,
            },
            200,
        )

    @app.post("/api/infer")
    def infer() -> tuple[object, int]:
        payload = request.get_json(silent=True) or {}
        try:
            x_norm = float(payload["x_norm"])
            y_norm = float(payload["y_norm"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "Expected JSON body with numeric x_norm and y_norm."}), 400

        result = engine.generate(x_norm, y_norm)
        return (
            jsonify(
                {
                    "x_norm": result.x_norm,
                    "y_norm": result.y_norm,
                    "latency_ms": round(result.latency_ms, 2),
                    "image_base64": result.image_base64,
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
        default=DEFAULT_CHECKPOINT,
        help=f"Path to the trained checkpoint. Default: {DEFAULT_CHECKPOINT}",
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
