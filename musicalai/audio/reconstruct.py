from __future__ import annotations

from pathlib import Path

import numpy as np

from musicalai.audio.features import denormalize_log_mel
from musicalai.config import AudioConfig


def _require_audio_stack():
    try:
        import librosa
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "librosa and soundfile are required for reconstruction. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return librosa, sf


def griffin_lim_from_log_mel(
    normalized_log_mel: np.ndarray,
    config: AudioConfig,
    iterations: int = 64,
) -> np.ndarray:
    """Invert a normalized Log-Mel spectrogram using Griffin-Lim."""
    librosa, _ = _require_audio_stack()
    log_mel_db = denormalize_log_mel(normalized_log_mel, config.top_db)
    mel_power = librosa.db_to_power(log_mel_db)
    audio = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        fmin=config.f_min,
        fmax=config.f_max,
        n_iter=iterations,
    )
    peak = np.max(np.abs(audio)) or 1.0
    return (audio / peak * 0.95).astype(np.float32)


def save_wave(path: str | Path, audio: np.ndarray, config: AudioConfig) -> None:
    _, sf = _require_audio_stack()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, config.sample_rate)

