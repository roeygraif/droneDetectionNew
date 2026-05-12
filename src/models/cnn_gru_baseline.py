"""CropCNNGRU baseline.

Per-crop CNN to flat features, GRU over time, and a single track-classification
head. Exists so the thesis can answer "does the U-Net decoder actually help?"
— if this baseline is competitive, the recurrent U-Net's extra structure isn't
buying us much.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CropCNNGRUConfig:
    """Architecture hyperparameters for CropCNNGRU."""

    in_channels: int = 5
    base_channels: int = 32
    feature_dim: int = 128
    gru_hidden: int = 128
    crop_size: int = 31
    use_visibility_head: bool = True


class _ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CropCNNGRU(nn.Module):
    """Per-crop CNN + temporal GRU + track head (+ optional visibility head).

    Input shape: ``(B, T, C, S, S)``. Output dict matches TrackletRecurrentUNet
    where it overlaps, but it does NOT emit a per-frame heatmap (set to None).
    """

    def __init__(self, cfg: CropCNNGRUConfig | None = None):
        super().__init__()
        self.cfg = cfg or CropCNNGRUConfig()
        c1 = self.cfg.base_channels
        c2 = c1 * 2
        c3 = c1 * 4

        self.block1 = _ConvBlock(self.cfg.in_channels, c1)
        self.block2 = _ConvBlock(c1, c2)
        self.block3 = _ConvBlock(c2, c3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(c3, self.cfg.feature_dim),
            nn.ReLU(inplace=True),
        )

        self.gru = nn.GRU(
            input_size=self.cfg.feature_dim,
            hidden_size=self.cfg.gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.track_head = nn.Sequential(
            nn.Linear(self.cfg.gru_hidden, self.cfg.gru_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.gru_hidden // 2, 1),
        )

        if self.cfg.use_visibility_head:
            self.visibility_head = nn.Linear(self.cfg.gru_hidden, 1)
        else:
            self.visibility_head = None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        if x.ndim != 5:
            raise ValueError(f"CropCNNGRU expects (B,T,C,S,S); got {tuple(x.shape)}")
        B, T, C, S, _ = x.shape
        x_flat = x.reshape(B * T, C, S, S)
        f = self.block1(x_flat)
        f = self.block2(f)
        f = self.block3(f)
        f = self.pool(f)
        f = self.fc(f).reshape(B, T, -1)

        out, h = self.gru(f)  # out: (B,T,hidden), h: (1,B,hidden)
        track_logit = self.track_head(h[-1]).reshape(B)

        visibility_logits = None
        if self.visibility_head is not None:
            visibility_logits = self.visibility_head(out).reshape(B, T)

        return {
            "track_logit": track_logit,
            "visibility_logits": visibility_logits,
            "heatmap_logits": None,
        }


__all__ = ["CropCNNGRU", "CropCNNGRUConfig"]
