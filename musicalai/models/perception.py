from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from musicalai.config import PerceptionConfig


class SelfAttentionPooling(nn.Module):
    """Scaled dot-product self-attention followed by salience pooling."""

    def __init__(self, input_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.salience = nn.Linear(attention_dim, 1)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query(sequence)
        k = self.key(sequence)
        v = self.value(sequence)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        attention_map = F.softmax(scores, dim=-1)
        contextual = torch.matmul(attention_map, v)
        salience = F.softmax(self.salience(contextual).squeeze(-1), dim=-1)
        pooled = torch.sum(contextual * salience.unsqueeze(-1), dim=1)
        return pooled, attention_map


class CNNBiGRUAttentionClassifier(nn.Module):
    """Hybrid perceptual model: 2D-CNN -> BiGRU -> Self-Attention -> genre logits."""

    def __init__(self, config: PerceptionConfig) -> None:
        super().__init__()
        channels = [1, *config.cnn_channels]
        blocks = []
        for in_channels, out_channels in zip(channels, channels[1:]):
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(2, 2)),
                    nn.Dropout2d(config.dropout),
                ]
            )
        self.cnn = nn.Sequential(*blocks)
        freq_after_pool = config.n_mels // (2 ** len(config.cnn_channels))
        gru_input_size = config.cnn_channels[-1] * freq_after_pool
        self.bigru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=config.gru_hidden_size,
            num_layers=config.gru_layers,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        recurrent_dim = config.gru_hidden_size * 2
        self.attention = SelfAttentionPooling(recurrent_dim, config.attention_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.attention_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.attention_dim, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.cnn(x)
        # B, C, F, T -> B, T, C*F
        sequence = features.permute(0, 3, 1, 2).flatten(2)
        sequence, _ = self.bigru(sequence)
        pooled, attention_map = self.attention(sequence)
        logits = self.classifier(pooled)
        return {"logits": logits, "attention": attention_map}

