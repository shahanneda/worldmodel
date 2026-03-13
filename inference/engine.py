from __future__ import annotations

import base64
import io
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from model.model import CoordinateToImageUNet


@dataclass(frozen=True)
class InferenceMetadata:
    checkpoint_path: str
    device: str
    image_size: int
    training_extra: dict[str, Any]


@dataclass(frozen=True)
class InferenceResult:
    x_norm: float
    y_norm: float
    latency_ms: float
    image_base64: str


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class CoordinateToImageInference:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = requested_device

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        model_state_dict, extra = self._extract_checkpoint_parts(checkpoint)
        image_size = self._resolve_image_size(extra)

        self.model = CoordinateToImageUNet(image_size=image_size)
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.metadata = InferenceMetadata(
            checkpoint_path=str(self.checkpoint_path),
            device=self.device,
            image_size=image_size,
            training_extra=extra,
        )
        self._lock = threading.Lock()

    @staticmethod
    def _extract_checkpoint_parts(checkpoint: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            extra = checkpoint.get("extra")
            return checkpoint["model_state_dict"], dict(extra) if isinstance(extra, dict) else {}

        if isinstance(checkpoint, dict):
            return checkpoint, {}

        raise TypeError("Unsupported checkpoint format.")

    @staticmethod
    def _resolve_image_size(extra: dict[str, Any]) -> int:
        image_size = extra.get("image_size", 128)
        if isinstance(image_size, (tuple, list)):
            if len(image_size) != 2 or image_size[0] != image_size[1]:
                raise ValueError("Only square image sizes are currently supported.")
            image_size = image_size[0]
        return int(image_size)

    def warmup(self) -> None:
        self.generate(0.5, 0.5)

    def generate(self, x_norm: float, y_norm: float) -> InferenceResult:
        x_norm = _clamp_unit(x_norm)
        y_norm = _clamp_unit(y_norm)

        coords = torch.tensor([[x_norm, y_norm]], dtype=torch.float32, device=self.device)
        start_time = time.perf_counter()

        with self._lock:
            with torch.inference_mode():
                prediction = self.model(coords)[0].detach().cpu().clamp(0.0, 1.0)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        image_base64 = self._tensor_to_base64_png(prediction)
        return InferenceResult(
            x_norm=x_norm,
            y_norm=y_norm,
            latency_ms=latency_ms,
            image_base64=image_base64,
        )

    @staticmethod
    def _tensor_to_base64_png(image_tensor: torch.Tensor) -> str:
        image_uint8 = (
            image_tensor.permute(1, 2, 0).mul(255.0).round().to(torch.uint8).numpy()
        )
        image = Image.fromarray(image_uint8, mode="RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
