from __future__ import annotations

import random

import numpy as np

from musicalai.config import SpecAugmentConfig


def _mask_value(spec: np.ndarray, replace_with_mean: bool) -> float:
    return float(spec.mean()) if replace_with_mean else 0.0


def frequency_mask(
    spec: np.ndarray,
    max_width: int,
    replace_with_mean: bool = True,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Apply one horizontal frequency mask to a Log-Mel spectrogram."""
    rng = rng or random
    augmented = spec.copy()
    n_mels = augmented.shape[-2]
    width = rng.randint(0, min(max_width, n_mels))
    if width == 0:
        return augmented
    start = rng.randint(0, n_mels - width)
    augmented[..., start : start + width, :] = _mask_value(spec, replace_with_mean)
    return augmented


def time_mask(
    spec: np.ndarray,
    max_width: int,
    replace_with_mean: bool = True,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Apply one vertical time mask to a Log-Mel spectrogram."""
    rng = rng or random
    augmented = spec.copy()
    time_steps = augmented.shape[-1]
    width = rng.randint(0, min(max_width, time_steps))
    if width == 0:
        return augmented
    start = rng.randint(0, time_steps - width)
    augmented[..., :, start : start + width] = _mask_value(spec, replace_with_mean)
    return augmented


def specaugment(
    spec: np.ndarray,
    config: SpecAugmentConfig,
    seed: int | None = None,
) -> np.ndarray:
    """Apply SpecAugment frequency and time masking to a spectrogram."""
    rng = random.Random(seed)
    augmented = spec.copy()
    for _ in range(config.num_freq_masks):
        augmented = frequency_mask(
            augmented,
            config.freq_mask_param,
            config.replace_with_mean,
            rng,
        )
    for _ in range(config.num_time_masks):
        augmented = time_mask(
            augmented,
            config.time_mask_param,
            config.replace_with_mean,
            rng,
        )
    return augmented.astype(np.float32)

