from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CoordinateToImageUNet(nn.Module):
    """
    A small coordinate-conditioned U-Net style generator.

    Input:
    - coords: (B, 2) normalized x/y fingertip location

    Output:
    - image: (B, 3, H, W) in [0, 1]
    """

    def __init__(
        self,
        image_size: int = 128,
        coord_dim: int = 2,
        base_channels: int = 32,
        latent_channels: int = 256,
        bottleneck_size: int = 8,
    ) -> None:
        super().__init__()
        if image_size % bottleneck_size != 0:
            raise ValueError("image_size must be divisible by bottleneck_size")

        num_upsamples = int(math.log2(image_size // bottleneck_size))
        if 2**num_upsamples != image_size // bottleneck_size:
            raise ValueError("image_size / bottleneck_size must be a power of 2")

        self.image_size = image_size
        self.bottleneck_size = bottleneck_size
        self.num_upsamples = num_upsamples

        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, latent_channels),
            nn.SiLU(),
        )

        self.bottleneck_proj = nn.Linear(
            latent_channels, latent_channels * bottleneck_size * bottleneck_size
        )
        self.bottleneck_conv = ConvBlock(latent_channels, latent_channels)

        skip_sizes = [bottleneck_size * (2 ** (idx + 1)) for idx in range(num_upsamples)]
        skip_channels = [
            max(base_channels, latent_channels // (2 ** (idx + 1)))
            for idx in range(num_upsamples)
        ]

        self.skip_projections = nn.ModuleList(
            [nn.Linear(latent_channels, channels * size * size) for size, channels in zip(skip_sizes, skip_channels)]
        )

        decoder_out_channels = skip_channels.copy()
        decoder_in_channels = [latent_channels] + decoder_out_channels[:-1]
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(in_ch, skip_ch, out_ch)
                for in_ch, skip_ch, out_ch in zip(
                    decoder_in_channels, skip_channels, decoder_out_channels
                )
            ]
        )

        self.to_rgb = nn.Sequential(
            nn.Conv2d(decoder_out_channels[-1], base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, coords: Tensor) -> Tensor:
        embedding = self.coord_mlp(coords)
        batch_size = coords.shape[0]

        x = self.bottleneck_proj(embedding).view(
            batch_size, -1, self.bottleneck_size, self.bottleneck_size
        )
        x = self.bottleneck_conv(x)

        skips = []
        for size, proj in zip(
            [self.bottleneck_size * (2 ** (idx + 1)) for idx in range(self.num_upsamples)],
            self.skip_projections,
        ):
            skip = proj(embedding).view(batch_size, -1, size, size)
            skips.append(skip)

        for up_block, skip in zip(self.up_blocks, skips):
            x = up_block(x, skip)

        return self.to_rgb(x)
