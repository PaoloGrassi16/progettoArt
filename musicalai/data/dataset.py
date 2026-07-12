from __future__ import annotations

from pathlib import Path

import numpy as np

from musicalai.audio.specaugment import specaugment
from musicalai.config import SpecAugmentConfig

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


def _require_torch():
    if torch is None:
        raise ImportError(
            "PyTorch is required for datasets and training. "
            "Install dependencies with `pip install -r requirements.txt`."
        )
    return torch


class LogMelDataset(Dataset):
    """Dataset backed by `.npy` Log-Mel spectrogram files."""

    def __init__(
        self,
        feature_root: str | Path,
        augment: bool = False,
        specaugment_config: SpecAugmentConfig | None = None,
    ) -> None:
        self.torch = _require_torch()
        self.feature_root = Path(feature_root)
        self.augment = augment
        self.specaugment_config = specaugment_config or SpecAugmentConfig()
        self.classes = sorted(p.name for p in self.feature_root.iterdir() if p.is_dir())
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.items = []
        for cls in self.classes:
            for path in sorted((self.feature_root / cls).glob("*.npy")):
                self.items.append((path, self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        feature = np.load(path).astype(np.float32)
        if self.augment:
            feature = specaugment(feature, self.specaugment_config)
        tensor = self.torch.from_numpy(feature).unsqueeze(0)
        return tensor, self.torch.tensor(label, dtype=self.torch.long)
