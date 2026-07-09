from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from musicalai.config import GanConfig, MusicalAIConfig
from musicalai.data.dataset import LogMelDataset
from musicalai.models.dcgan import SpectrogramDiscriminator, SpectrogramGenerator


def _resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _crop_or_pad_time(batch: torch.Tensor, target_steps: int) -> torch.Tensor:
    current = batch.size(-1)
    if current == target_steps:
        return batch
    if current < target_steps:
        return nn.functional.pad(batch, (0, target_steps - current))
    start = torch.randint(0, current - target_steps + 1, (1,), device=batch.device).item()
    return batch[..., start : start + target_steps]


def _match_gan_shape(batch: torch.Tensor, config: GanConfig) -> torch.Tensor:
    batch = batch[..., : config.n_mels, :]
    if batch.size(-2) < config.n_mels:
        batch = nn.functional.pad(batch, (0, 0, 0, config.n_mels - batch.size(-2)))
    return _crop_or_pad_time(batch, config.time_steps)


def train_gan(
    feature_root: str | Path,
    output: str | Path,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 2e-4,
    device: str | None = None,
    config: MusicalAIConfig | None = None,
) -> Path:
    """Train a DCGAN over normalized Log-Mel spectrogram crops."""
    config = config or MusicalAIConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device_obj = _resolve_device(device)

    dataset = LogMelDataset(feature_root, augment=False)
    if not dataset.items:
        raise ValueError(f"No .npy features found under {feature_root}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    generator = SpectrogramGenerator(config.gan).to(device_obj)
    discriminator = SpectrogramDiscriminator(config.gan).to(device_obj)
    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    fixed_noise = torch.randn(8, config.gan.latent_dim, device=device_obj)
    generator_path = output / "dcgan_generator.pt"

    for epoch in range(1, epochs + 1):
        for real, _ in tqdm(loader, desc=f"GAN epoch {epoch}"):
            real = _match_gan_shape(real.to(device_obj), config.gan)
            batch_n = real.size(0)
            real_labels = torch.ones(batch_n, device=device_obj)
            fake_labels = torch.zeros(batch_n, device=device_obj)

            noise = torch.randn(batch_n, config.gan.latent_dim, device=device_obj)
            fake = generator(noise).detach()
            opt_d.zero_grad(set_to_none=True)
            d_real = criterion(discriminator(real), real_labels)
            d_fake = criterion(discriminator(fake), fake_labels)
            d_loss = d_real + d_fake
            d_loss.backward()
            opt_d.step()

            noise = torch.randn(batch_n, config.gan.latent_dim, device=device_obj)
            opt_g.zero_grad(set_to_none=True)
            fake = generator(noise)
            g_loss = criterion(discriminator(fake), real_labels)
            g_loss.backward()
            opt_g.step()

        torch.save(
            {
                "generator_state": generator.state_dict(),
                "discriminator_state": discriminator.state_dict(),
                "config": asdict(config.gan),
                "fixed_preview": generator(fixed_noise).detach().cpu(),
                "epoch": epoch,
            },
            generator_path,
        )

    return generator_path

