from pathlib import Path
import sys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.utils.data import random_split, DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from musicalai.data.dataset import LogMelDataset
from musicalai.models.perception import CNNBiGRUAttentionClassifier
from musicalai.config import PerceptionConfig


CHECKPOINT = "runs/perception/perception_classifier.pt"
FEATURE_ROOT = "data/features"

VALIDATION_FRACTION = 0.20
BATCH_SIZE = 16


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    classes = checkpoint["classes"]

    config = PerceptionConfig(
        **checkpoint["config"]
    )

    model = CNNBiGRUAttentionClassifier(config)
    model.load_state_dict(
        checkpoint["model_state"]
    )
    model.to(device)
    model.eval()

    dataset = LogMelDataset(
        FEATURE_ROOT,
        augment=False,
    )

    val_size = max(
        1,
        int(len(dataset) * VALIDATION_FRACTION)
    )

    train_size = len(dataset) - val_size

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    y_true = []
    y_pred = []

    with torch.no_grad():

        for inputs, targets in val_loader:

            inputs = inputs.to(device)

            logits = model(inputs)["logits"]

            predictions = logits.argmax(dim=1)

            y_true.extend(
                targets.cpu().numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        output_dict=True,
        digits=4,
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        "classification_report.csv"
    )

    print("\nClassification Report\n")
    print(report_df)

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    cm_df = pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )

    cm_df.to_csv(
        "confusion_matrix.csv"
    )

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Genre"
    )

    plt.ylabel(
        "True Genre"
    )

    plt.tight_layout()

    plt.savefig(
        "confusion_matrix.png",
        dpi=300
    )

    print(
        "\nSaved:"
        "\n- classification_report.csv"
        "\n- confusion_matrix.csv"
        "\n- confusion_matrix.png"
    )


if __name__ == "__main__":
    main()