from __future__ import annotations

from pathlib import Path

import numpy as np

from musicalai.config import AudioConfig


def _require_librosa():
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "librosa is required for audio feature extraction. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return librosa


def load_audio(path: str | Path, config: AudioConfig) -> np.ndarray:
    """Load mono audio and force the duration expected by the neural pipeline."""
    librosa = _require_librosa()
    y, _ = librosa.load(path, sr=config.sample_rate, mono=True)
    target = config.target_samples
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    return y[:target].astype(np.float32)


def log_mel_spectrogram(audio: np.ndarray, config: AudioConfig) -> np.ndarray:
    """Convert a waveform into a normalized Log-Mel spectrogram."""
    librosa = _require_librosa()
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.f_min,
        fmax=config.f_max,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=config.top_db)
    return normalize_log_mel(log_mel, top_db=config.top_db)


def normalize_log_mel(log_mel: np.ndarray, top_db: float = 80.0) -> np.ndarray:
    """Map decibel Log-Mel values from [-top_db, 0] to [-1, 1]."""
    clipped = np.clip(log_mel, -top_db, 0.0)
    return ((clipped + top_db) / top_db * 2.0 - 1.0).astype(np.float32)


def denormalize_log_mel(normalized: np.ndarray, top_db: float = 80.0) -> np.ndarray:
    """Map normalized Log-Mel values from [-1, 1] back to decibels."""
    clipped = np.clip(normalized, -1.0, 1.0)
    return ((clipped + 1.0) / 2.0 * top_db - top_db).astype(np.float32)


def extract_file(path: str | Path, config: AudioConfig) -> np.ndarray:
    audio = load_audio(path, config)
    return log_mel_spectrogram(audio, config)

