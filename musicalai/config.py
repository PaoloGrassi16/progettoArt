from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


GTZAN_GENRES = (
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
)

EXTENDED_SCENARIO_TARGETS = GTZAN_GENRES + (
    "ambient",
    "electronic",
    "lofi",
    "lofi_glitch",
)


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    duration_seconds: float = 30.0
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    f_min: float = 20.0
    f_max: float | None = None
    top_db: float = 80.0

    @property
    def target_samples(self) -> int:
        return int(self.sample_rate * self.duration_seconds)


@dataclass(frozen=True)
class SpecAugmentConfig:
    freq_mask_param: int = 18
    time_mask_param: int = 32
    num_freq_masks: int = 2
    num_time_masks: int = 2
    replace_with_mean: bool = True


@dataclass(frozen=True)
class PerceptionConfig:
    n_mels: int = 128
    num_classes: int = len(GTZAN_GENRES)
    cnn_channels: tuple[int, ...] = (32, 64, 128)
    gru_hidden_size: int = 128
    gru_layers: int = 2
    attention_dim: int = 128
    dropout: float = 0.25
    focal_gamma: float = 2.0


@dataclass(frozen=True)
class GanConfig:
    latent_dim: int = 128
    n_mels: int = 128
    time_steps: int = 128
    base_channels: int = 64
    spectrogram_min: float = -1.0
    spectrogram_max: float = 1.0


@dataclass(frozen=True)
class ClosedLoopConfig:
    confidence_threshold: float = 0.85
    max_attempts: int = 64
    device: str = "cuda"


@dataclass(frozen=True)
class ProjectPaths:
    data_dir: Path = Path("data")
    feature_dir: Path = Path("data/features")
    run_dir: Path = Path("runs")
    checkpoint_dir: Path = Path("checkpoints")
    generated_dir: Path = Path("outputs/generated")

    def ensure(self) -> None:
        for path in (
            self.data_dir,
            self.feature_dir,
            self.run_dir,
            self.checkpoint_dir,
            self.generated_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class MusicalAIConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    specaugment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    gan: GanConfig = field(default_factory=GanConfig)
    closed_loop: ClosedLoopConfig = field(default_factory=ClosedLoopConfig)
    paths: ProjectPaths = field(default_factory=ProjectPaths)

