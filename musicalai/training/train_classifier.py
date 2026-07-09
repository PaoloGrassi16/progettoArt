from __future__ import annotations

import csv
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from musicalai.config import MusicalAIConfig, PerceptionConfig
from musicalai.data.dataset import LogMelDataset
from musicalai.models.losses import FocalLoss
from musicalai.models.perception import CNNBiGRUAttentionClassifier


def _resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return float((predictions == targets).float().mean().item())


def train_classifier(
    feature_root: str | Path,
    output: str | Path,
    epochs: int = 25,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    validation_fraction: float = 0.2,
    device: str | None = None,
    config: MusicalAIConfig | None = None,
) -> Path:
    """Train the CNN-BiGRU-Attention perceptual validator."""
    config = config or MusicalAIConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device_obj = _resolve_device(device)

    full_dataset = LogMelDataset(
        feature_root,
        augment=True,
        specaugment_config=config.specaugment,
    )
    if not full_dataset.items:
        raise ValueError(f"No .npy features found under {feature_root}")

    num_classes = len(full_dataset.classes)
    perception_config = replace(config.perception, num_classes=num_classes)
    val_size = max(1, int(len(full_dataset) * validation_fraction))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CNNBiGRUAttentionClassifier(perception_config).to(device_obj)
    criterion = FocalLoss(gamma=perception_config.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    history_path = output / "classifier_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()

        best_val = 0.0
        best_path = output / "perception_classifier.pt"
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            train_acc = 0.0
            for inputs, targets in tqdm(train_loader, desc=f"Classifier epoch {epoch}"):
                inputs = inputs.to(device_obj)
                targets = targets.to(device_obj)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)["logits"]
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                train_acc += _accuracy(logits.detach(), targets)

            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device_obj)
                    targets = targets.to(device_obj)
                    logits = model(inputs)["logits"]
                    loss = criterion(logits, targets)
                    val_loss += float(loss.item())
                    val_acc += _accuracy(logits, targets)

            row = {
                "epoch": epoch,
                "train_loss": train_loss / max(1, len(train_loader)),
                "train_acc": train_acc / max(1, len(train_loader)),
                "val_loss": val_loss / max(1, len(val_loader)),
                "val_acc": val_acc / max(1, len(val_loader)),
            }
            writer.writerow(row)
            history_file.flush()

            if row["val_acc"] >= best_val:
                best_val = row["val_acc"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "classes": full_dataset.classes,
                        "config": asdict(perception_config),
                    },
                    best_path,
                )

    return best_path

