from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from musicalai.audio.features import extract_file
from musicalai.config import AudioConfig

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"}


def iter_audio_files(audio_root: str | Path):
    root = Path(audio_root)
    for genre_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(genre_dir.rglob("*")):
            if path.suffix.lower() in AUDIO_EXTENSIONS:
                yield genre_dir.name, path


def prepare_feature_dataset(
    audio_root: str | Path,
    output_root: str | Path,
    config: AudioConfig,
) -> dict[str, int]:
    """Extract Log-Mel features into class folders and emit metadata."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    metadata = []

    for genre, path in tqdm(list(iter_audio_files(audio_root)), desc="Extracting Log-Mel"):
      try:
        genre_out = output / genre
        genre_out.mkdir(parents=True, exist_ok=True)

        feature = extract_file(path, config)

        destination = genre_out / f"{path.stem}.npy"
        np.save(destination, feature)

        counts[genre] = counts.get(genre, 0) + 1

        metadata.append(
            {
                "genre": genre,
                "source": str(path),
                "feature": str(destination),
                "shape": list(feature.shape),
            }
        )

      except Exception as exc:
        print(f"\n[WARNING] File ignorato: {path}")
        print(f"          Motivo: {exc}")
        continue

    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return counts

