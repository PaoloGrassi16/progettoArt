from __future__ import annotations

import torch
from torch import nn

from musicalai.config import GanConfig


class SpectrogramGenerator(nn.Module):
    """DCGAN generator that maps latent noise to normalized Log-Mel images."""

    def __init__(self, config: GanConfig) -> None:
        super().__init__()
        c = config.base_channels
        self.config = config
        self.project = nn.Sequential(
            nn.Linear(config.latent_dim, c * 8 * 8 * 8),
            nn.BatchNorm1d(c * 8 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, 4, 2, 1),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(c, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z).view(z.size(0), self.config.base_channels * 8, 8, 8)
        x = self.net(x)
        return x[..., : self.config.n_mels, : self.config.time_steps]


class SpectrogramDiscriminator(nn.Module):
    """CNN discriminator for real/fake Log-Mel spectrograms."""

    def __init__(self, config: GanConfig) -> None:
        super().__init__()
        c = config.base_channels
        self.net = nn.Sequential(
            nn.Conv2d(1, c, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 8, 4, 2, 1),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c * 8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

