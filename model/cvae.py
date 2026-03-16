from __future__ import annotations

import torch
from torch import Tensor, nn

from model.model import CoordinateToImageUNet


def clamp_logvar(logvar: Tensor) -> Tensor:
    return torch.clamp(logvar, min=-8.0, max=4.0)


def reparameterize(mu: Tensor, logvar: Tensor, temperature: float = 1.0) -> Tensor:
    logvar = clamp_logvar(logvar)
    std = torch.exp(0.5 * logvar) * temperature
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_diag_gaussians(
    mu_q: Tensor,
    logvar_q: Tensor,
    mu_p: Tensor,
    logvar_p: Tensor,
) -> Tensor:
    logvar_q = clamp_logvar(logvar_q)
    logvar_p = clamp_logvar(logvar_p)

    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)

    kl_per_dim = 0.5 * (
        logvar_p
        - logvar_q
        + (var_q + (mu_q - mu_p) ** 2) / var_p
        - 1.0
    )
    return kl_per_dim.sum(dim=1)


def kl_beta(
    global_step: int,
    *,
    warmup_steps: int = 5000,
    max_beta: float = 1e-3,
) -> float:
    if warmup_steps <= 0:
        return max_beta
    frac = min(1.0, global_step / float(warmup_steps))
    return max_beta * frac


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        latent_dim: int = 32,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        def gn(channels: int) -> nn.GroupNorm:
            groups = 8 if channels % 8 == 0 else 4 if channels % 4 == 0 else 1
            return nn.GroupNorm(groups, channels)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            gn(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            gn(base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            gn(base_channels * 4),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            gn(base_channels * 8),
            nn.SiLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        flat_dim = base_channels * 8 * 4 * 4
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.SiLU(),
        )
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, img: Tensor) -> tuple[Tensor, Tensor]:
        h = self.conv(img)
        h = self.pool(h)
        h = self.fc(h)
        return self.mu(h), self.logvar(h)


class PriorNet(nn.Module):
    def __init__(
        self,
        *,
        coord_dim: int = 2,
        latent_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, coord: Tensor) -> tuple[Tensor, Tensor]:
        h = self.net(coord)
        return self.mu(h), self.logvar(h)


class PointingCVAE(nn.Module):
    def __init__(
        self,
        *,
        image_size: int = 64,
        coord_dim: int = 2,
        latent_dim: int = 32,
        base_channels: int = 16,
        latent_channels: int = 128,
        bottleneck_size: int = 8,
        posterior_base_channels: int = 32,
        prior_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim

        self.posterior = PosteriorEncoder(
            in_channels=3,
            latent_dim=latent_dim,
            base_channels=posterior_base_channels,
        )
        self.prior = PriorNet(
            coord_dim=coord_dim,
            latent_dim=latent_dim,
            hidden_dim=prior_hidden_dim,
        )
        self.decoder = CoordinateToImageUNet(
            image_size=image_size,
            coord_dim=coord_dim + latent_dim,
            base_channels=base_channels,
            latent_channels=latent_channels,
            bottleneck_size=bottleneck_size,
        )

    def decode(self, coord: Tensor, z: Tensor) -> Tensor:
        conditioning = torch.cat([coord, z], dim=1)
        return self.decoder(conditioning)

    def forward_train(
        self,
        coord: Tensor,
        img: Tensor,
        *,
        temperature: float = 1.0,
    ) -> dict[str, Tensor]:
        mu_post, logvar_post = self.posterior(img)
        mu_prior, logvar_prior = self.prior(coord)
        z = reparameterize(mu_post, logvar_post, temperature=temperature)
        img_hat = self.decode(coord, z)
        return {
            "img_hat": img_hat,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "z": z,
        }

    def reconstruct_from_posterior_mean(
        self,
        coord: Tensor,
        img: Tensor,
    ) -> dict[str, Tensor]:
        mu_post, logvar_post = self.posterior(img)
        img_hat = self.decode(coord, mu_post)
        return {
            "img_hat": img_hat,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "z": mu_post,
        }

    def sample_from_prior(
        self,
        coord: Tensor,
        *,
        temperature: float = 1.0,
    ) -> dict[str, Tensor]:
        mu_prior, logvar_prior = self.prior(coord)
        z = reparameterize(mu_prior, logvar_prior, temperature=temperature)
        img_hat = self.decode(coord, z)
        return {
            "img_hat": img_hat,
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "z": z,
        }

    def sample_prior_mean(self, coord: Tensor) -> dict[str, Tensor]:
        mu_prior, logvar_prior = self.prior(coord)
        img_hat = self.decode(coord, mu_prior)
        return {
            "img_hat": img_hat,
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "z": mu_prior,
        }

    def forward(self, coord: Tensor) -> Tensor:
        return self.sample_prior_mean(coord)["img_hat"]
