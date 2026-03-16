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
from torch import nn

from model.cvae import PointingCVAE, clamp_logvar
from model.model import CoordinateToImageUNet


DEFAULT_PRIOR_SAMPLE_COUNT = 4
MAX_PRIOR_SAMPLE_COUNT = 8


@dataclass(frozen=True)
class InferenceMetadata:
    checkpoint_path: str
    device: str
    image_size: int
    model_kind: str
    base_channels: int
    latent_channels: int
    latent_dim: int
    coord_space: str
    prior_sample_temperature: float
    training_extra: dict[str, Any]


@dataclass(frozen=True)
class InferenceResult:
    x_norm: float
    y_norm: float
    model_x: float
    model_y: float
    latency_ms: float
    image_base64: str
    default_gallery_key: str
    gallery: list[dict[str, Any]]
    model_kind: str
    coord_space: str
    supports_latent_exploration: bool
    sample_count: int
    sampling_temperature: float
    prior_distribution: dict[str, Any] | None
    uncertainty_heatmap_base64: str | None


class InferenceModelWrapper(nn.Module):
    def __init__(self, model: nn.Module, *, coord_space: str) -> None:
        super().__init__()
        self.model = model
        self.coord_space = coord_space

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if self.coord_space == "minus_one_to_one":
            coords = coords * 2.0 - 1.0
        return self.model(coords)


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clamp_sample_count(value: int) -> int:
    return max(0, min(MAX_PRIOR_SAMPLE_COUNT, int(value)))


def _clamp_temperature(value: float) -> float:
    return max(0.05, min(2.0, float(value)))


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
        model_state_dict, extra, training_config = self._extract_checkpoint_parts(checkpoint)
        model_config = (
            dict(training_config.get("model", {}))
            if isinstance(training_config, dict)
            else {}
        )
        data_config = (
            dict(training_config.get("data", {}))
            if isinstance(training_config, dict)
            else {}
        )
        training_runtime_config = (
            dict(training_config.get("training", {}))
            if isinstance(training_config, dict)
            else {}
        )
        model_kind = str(model_config.get("kind", extra.get("model_kind", "coord_to_image_unet")))
        image_size = self._resolve_image_size(extra, model_config=model_config)
        base_channels = int(model_config.get("base_channels", extra.get("base_channels", 32)))
        latent_channels = int(model_config.get("latent_channels", extra.get("latent_channels", 256)))
        coord_dim = int(model_config.get("coord_dim", extra.get("coord_dim", 2)))
        latent_dim = int(model_config.get("latent_dim", extra.get("latent_dim", 32)))
        posterior_base_channels = int(
            model_config.get(
                "posterior_base_channels",
                extra.get("posterior_base_channels", 32),
            )
        )
        prior_hidden_dim = int(
            model_config.get(
                "prior_hidden_dim",
                extra.get("prior_hidden_dim", 128),
            )
        )
        bottleneck_size = int(model_config.get("bottleneck_size", extra.get("bottleneck_size", 8)))
        coord_space = str(data_config.get("coord_space", extra.get("coord_space", "zero_to_one")))
        prior_sample_temperature = float(
            training_runtime_config.get(
                "prior_sample_temperature",
                extra.get("prior_sample_temperature", 0.7),
            )
        )

        core_model: nn.Module
        if model_kind == "pointing_cvae":
            core_model = PointingCVAE(
                image_size=image_size,
                coord_dim=coord_dim,
                latent_dim=latent_dim,
                base_channels=base_channels,
                latent_channels=latent_channels,
                bottleneck_size=bottleneck_size,
                posterior_base_channels=posterior_base_channels,
                prior_hidden_dim=prior_hidden_dim,
            )
        else:
            core_model = CoordinateToImageUNet(
                image_size=image_size,
                coord_dim=coord_dim,
                base_channels=base_channels,
                latent_channels=latent_channels,
                bottleneck_size=bottleneck_size,
            )

        core_model.load_state_dict(model_state_dict)
        self.core_model = core_model
        self.model = InferenceModelWrapper(core_model, coord_space=coord_space)
        self.model.to(self.device)
        self.model.eval()

        self.model_kind = model_kind
        self.coord_space = coord_space
        self.supports_latent_exploration = model_kind == "pointing_cvae"
        self.prior_sample_temperature = prior_sample_temperature

        self.metadata = InferenceMetadata(
            checkpoint_path=str(self.checkpoint_path),
            device=self.device,
            image_size=image_size,
            model_kind=model_kind,
            base_channels=base_channels,
            latent_channels=latent_channels,
            latent_dim=latent_dim,
            coord_space=coord_space,
            prior_sample_temperature=prior_sample_temperature,
            training_extra=extra,
        )
        self._lock = threading.Lock()

    @staticmethod
    def _extract_checkpoint_parts(
        checkpoint: Any,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            extra = checkpoint.get("extra")
            training_config = checkpoint.get("training_config")
            return (
                checkpoint["model_state_dict"],
                dict(extra) if isinstance(extra, dict) else {},
                dict(training_config) if isinstance(training_config, dict) else {},
            )

        if isinstance(checkpoint, dict):
            return checkpoint, {}, {}

        raise TypeError("Unsupported checkpoint format.")

    @staticmethod
    def _resolve_image_size(
        extra: dict[str, Any],
        *,
        model_config: dict[str, Any],
    ) -> int:
        image_size = model_config.get("image_size", extra.get("image_size", 128))
        if isinstance(image_size, (tuple, list)):
            if len(image_size) != 2 or image_size[0] != image_size[1]:
                raise ValueError("Only square image sizes are currently supported.")
            image_size = image_size[0]
        return int(image_size)

    def warmup(self) -> None:
        self.generate(0.5, 0.5, sample_count=0)

    def _to_model_coord_space(self, unit_coords: torch.Tensor) -> torch.Tensor:
        if self.coord_space == "minus_one_to_one":
            return unit_coords * 2.0 - 1.0
        return unit_coords

    def generate(
        self,
        x_norm: float,
        y_norm: float,
        *,
        sample_count: int = DEFAULT_PRIOR_SAMPLE_COUNT,
        temperature: float | None = None,
    ) -> InferenceResult:
        x_norm = _clamp_unit(x_norm)
        y_norm = _clamp_unit(y_norm)
        sample_count = _clamp_sample_count(sample_count)
        sampling_temperature = _clamp_temperature(
            self.prior_sample_temperature if temperature is None else temperature
        )

        coords = torch.tensor([[x_norm, y_norm]], dtype=torch.float32, device=self.device)
        model_coords = self._to_model_coord_space(coords)
        start_time = time.perf_counter()

        with self._lock:
            with torch.inference_mode():
                if self.supports_latent_exploration:
                    payload = self._generate_cvae_payload(
                        model_coords,
                        sample_count=sample_count,
                        temperature=sampling_temperature,
                    )
                else:
                    payload = self._generate_deterministic_payload(coords)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return InferenceResult(
            x_norm=x_norm,
            y_norm=y_norm,
            model_x=float(model_coords[0, 0].detach().cpu().item()),
            model_y=float(model_coords[0, 1].detach().cpu().item()),
            latency_ms=latency_ms,
            image_base64=payload["image_base64"],
            default_gallery_key=payload["default_gallery_key"],
            gallery=payload["gallery"],
            model_kind=self.model_kind,
            coord_space=self.coord_space,
            supports_latent_exploration=self.supports_latent_exploration,
            sample_count=payload["sample_count"],
            sampling_temperature=sampling_temperature,
            prior_distribution=payload["prior_distribution"],
            uncertainty_heatmap_base64=payload["uncertainty_heatmap_base64"],
        )

    def _generate_deterministic_payload(self, coords: torch.Tensor) -> dict[str, Any]:
        prediction = self.model(coords)[0].detach().cpu().clamp(0.0, 1.0)
        image_base64 = self._tensor_to_base64_png(prediction)
        return {
            "image_base64": image_base64,
            "default_gallery_key": "deterministic",
            "sample_count": 0,
            "prior_distribution": None,
            "uncertainty_heatmap_base64": None,
            "gallery": [
                {
                    "key": "deterministic",
                    "label": "Deterministic output",
                    "kind": "deterministic",
                    "image_base64": image_base64,
                    "mean_absolute_delta": None,
                    "latent_l2": None,
                }
            ],
        }

    def _generate_cvae_payload(
        self,
        model_coords: torch.Tensor,
        *,
        sample_count: int,
        temperature: float,
    ) -> dict[str, Any]:
        cvae = self.core_model
        if not isinstance(cvae, PointingCVAE):
            raise TypeError("Latent exploration requested for a non-CVAE checkpoint.")

        mu_prior, logvar_prior = cvae.prior(model_coords)
        prior_std = torch.exp(0.5 * clamp_logvar(logvar_prior))
        prior_mean_image = cvae.decode(model_coords, mu_prior)[0].detach().cpu().clamp(0.0, 1.0)
        prior_mean_base64 = self._tensor_to_base64_png(prior_mean_image)

        mu_cpu = mu_prior[0].detach().cpu()
        std_cpu = prior_std[0].detach().cpu()
        gallery: list[dict[str, Any]] = [
            {
                "key": "prior_mean",
                "label": "Prior mean",
                "kind": "mean",
                "image_base64": prior_mean_base64,
                "mean_absolute_delta": 0.0,
                "latent_l2": float(mu_cpu.norm().item()),
            }
        ]
        prior_samples: list[dict[str, Any]] = []
        uncertainty_heatmap_base64: str | None = None
        default_gallery_key = "prior_mean"

        if sample_count > 0:
            expanded_mu = mu_prior.expand(sample_count, -1)
            expanded_std = prior_std.expand(sample_count, -1)
            z_samples = expanded_mu + torch.randn_like(expanded_std) * expanded_std * temperature
            coord_batch = model_coords.expand(sample_count, -1)
            decoded_samples = cvae.decode(coord_batch, z_samples).detach().cpu().clamp(0.0, 1.0)
            sample_latents = z_samples.detach().cpu()
            sample_deltas = (decoded_samples - prior_mean_image.unsqueeze(0)).abs().mean(dim=(1, 2, 3))

            for index in range(sample_count):
                key = f"prior_sample_{index + 1}"
                gallery.append(
                    {
                        "key": key,
                        "label": f"Sample {index + 1}",
                        "kind": "sample",
                        "image_base64": self._tensor_to_base64_png(decoded_samples[index]),
                        "mean_absolute_delta": float(sample_deltas[index].item()),
                        "latent_l2": float(sample_latents[index].norm().item()),
                    }
                )
                prior_samples.append(
                    {
                        "key": key,
                        "values": [float(value) for value in sample_latents[index].tolist()],
                    }
                )

            default_gallery_key = gallery[1]["key"]
            if sample_count > 1:
                uncertainty_map = decoded_samples.std(dim=0, unbiased=False).mean(dim=0)
                uncertainty_max = float(uncertainty_map.max().item())
                if uncertainty_max > 0.0:
                    uncertainty_map = uncertainty_map / uncertainty_max
                    uncertainty_heatmap_base64 = self._scalar_map_to_base64_png(uncertainty_map)

        top_uncertainty_dims = torch.argsort(std_cpu, descending=True)[: min(5, std_cpu.numel())]
        prior_distribution = {
            "mean": [float(value) for value in mu_cpu.tolist()],
            "std": [float(value) for value in std_cpu.tolist()],
            "samples": prior_samples,
            "avg_std": float(std_cpu.mean().item()),
            "max_std": float(std_cpu.max().item()),
            "temperature": float(temperature),
            "top_uncertainty_dims": [
                {
                    "dim": int(index.item()),
                    "mean": float(mu_cpu[index].item()),
                    "std": float(std_cpu[index].item()),
                }
                for index in top_uncertainty_dims
            ],
        }
        return {
            "image_base64": prior_mean_base64,
            "default_gallery_key": default_gallery_key,
            "sample_count": sample_count,
            "prior_distribution": prior_distribution,
            "uncertainty_heatmap_base64": uncertainty_heatmap_base64,
            "gallery": gallery,
        }

    @staticmethod
    def _tensor_to_base64_png(image_tensor: torch.Tensor) -> str:
        image_uint8 = (
            image_tensor.permute(1, 2, 0).mul(255.0).round().to(torch.uint8).numpy()
        )
        image = Image.fromarray(image_uint8, mode="RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @classmethod
    def _scalar_map_to_base64_png(cls, value_map: torch.Tensor) -> str:
        value_map = value_map.detach().cpu().clamp(0.0, 1.0)
        red = value_map.sqrt()
        green = value_map * 0.9
        blue = (1.0 - value_map) * 0.28
        rgb = torch.stack([red, green, blue], dim=0)
        return cls._tensor_to_base64_png(rgb)
