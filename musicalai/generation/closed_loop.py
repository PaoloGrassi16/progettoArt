from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from musicalai.audio.reconstruct import griffin_lim_from_log_mel, save_wave
from musicalai.config import AudioConfig, ClosedLoopConfig, GanConfig
from musicalai.models.dcgan import SpectrogramGenerator
from musicalai.models.perception import CNNBiGRUAttentionClassifier


@dataclass(frozen=True)
class GenerationResult:
    approved: bool
    target: str
    predicted: str
    confidence: float
    attempts: int
    spectrogram: np.ndarray


class ClosedLoopSoundscapeGenerator:
    """Generate candidates and keep only spectrograms approved by the validator."""

    def __init__(
        self,
        generator: SpectrogramGenerator,
        validator: CNNBiGRUAttentionClassifier,
        classes: list[str],
        gan_config: GanConfig,
        loop_config: ClosedLoopConfig,
    ) -> None:
        self.generator = generator.eval()
        self.validator = validator.eval()
        self.classes = classes
        self.gan_config = gan_config
        self.loop_config = loop_config

    @property
    def device(self) -> torch.device:
        return next(self.generator.parameters()).device

    def generate(self, target: str) -> GenerationResult:
        if target not in self.classes:
            raise ValueError(f"Unknown target '{target}'. Available: {self.classes}")
        target_idx = self.classes.index(target)
        best: GenerationResult | None = None

        with torch.no_grad():
            for attempt in range(1, self.loop_config.max_attempts + 1):
                z = torch.randn(1, self.gan_config.latent_dim, device=self.device)
                candidate = self.generator(z)
                logits = self.validator(candidate)["logits"]
                probs = F.softmax(logits, dim=-1).squeeze(0)
                confidence = float(probs[target_idx].item())
                predicted_idx = int(torch.argmax(probs).item())
                predicted = self.classes[predicted_idx]
                spec = candidate.squeeze(0).squeeze(0).detach().cpu().numpy()
                result = GenerationResult(
                    approved=confidence >= self.loop_config.confidence_threshold,
                    target=target,
                    predicted=predicted,
                    confidence=confidence,
                    attempts=attempt,
                    spectrogram=spec,
                )
                if best is None or result.confidence > best.confidence:
                    best = result
                if result.approved:
                    return result

        assert best is not None
        return best


def save_generation_outputs(
    result: GenerationResult,
    output_dir: str | Path,
    audio_config: AudioConfig,
    griffin_lim_iterations: int = 64,
) -> dict[str, Path]:
    """Save `.npy` spectrogram and `.wav` audio for a generation result."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{result.target}_attempts-{result.attempts}_conf-{result.confidence:.2f}"
    spec_path = output / f"{stem}.npy"
    wav_path = output / f"{stem}.wav"
    np.save(spec_path, result.spectrogram)
    audio = griffin_lim_from_log_mel(result.spectrogram, audio_config, griffin_lim_iterations)
    save_wave(wav_path, audio, audio_config)
    return {"spectrogram": spec_path, "audio": wav_path}

