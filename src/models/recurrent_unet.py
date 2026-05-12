"""TrackletRecurrentUNet.

A small U-Net applied per-frame on the (C,S,S) crop, with a ConvLSTM at the
bottleneck that integrates information across the T frames of the tube. Heads:

  - heatmap_head:    per-frame (S,S) UAV-location heatmap
  - visibility_head: per-frame scalar — is the target visible in this frame
  - track_head:      one scalar per tube — is this tube a real UAV at all

The ``use_convlstm=False`` variant is the non-recurrent ablation; the model
then degenerates to a shared-weights per-frame U-Net with a temporal pooling
aggregation for the track head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convlstm import ConvLSTM


@dataclass
class TrackletRecurrentUNetConfig:
    """Architecture hyperparameters for TrackletRecurrentUNet."""

    in_channels: int = 5
    base_channels: int = 32
    bottleneck_channels: int = 128
    use_convlstm: bool = True
    convlstm_kernel: int = 3
    track_pool: str = "mean"           # "mean" | "max"
    crop_size: int = 31


class _DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrackletRecurrentUNet(nn.Module):
    """Per-frame U-Net + bottleneck ConvLSTM + multi-head outputs.

    Input:  ``x`` shape ``(B, T, C, S, S)``.
    Output: dict with
        - ``track_logit``:        ``(B,)``
        - ``visibility_logits``:  ``(B, T)``
        - ``heatmap_logits``:     ``(B, T, 1, S, S)``
    """

    def __init__(self, cfg: TrackletRecurrentUNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or TrackletRecurrentUNetConfig()
        c1 = self.cfg.base_channels
        c2 = c1 * 2
        c3 = self.cfg.bottleneck_channels

        # Encoder (per-frame, weights shared across T).
        self.enc1 = _DoubleConv(self.cfg.in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _DoubleConv(c2, c3)

        # Bottleneck temporal block.
        if self.cfg.use_convlstm:
            self.temporal = ConvLSTM(c3, c3, kernel_size=self.cfg.convlstm_kernel)
        else:
            self.temporal = None

        # Decoder (per-frame).
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(c1 + c1, c1)

        # Heads.
        self.heatmap_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(c3, c3 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(c3 // 2, 1),
        )
        self.track_head = nn.Sequential(
            nn.Linear(c3, c3 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(c3 // 2, 1),
        )

    # ------------------------------------------------------------------
    def _encode_per_frame(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the encoder on a (B*T, C, S, S) tensor, return (e1, e2, e3)."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        return e1, e2, e3

    def _decode_per_frame(
        self,
        bottleneck: torch.Tensor,
        e1: torch.Tensor,
        e2: torch.Tensor,
    ) -> torch.Tensor:
        d2 = self.up2(bottleneck)
        d2 = _crop_to(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = _crop_to(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return d1

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"TrackletRecurrentUNet expects (B,T,C,S,S); got {tuple(x.shape)}")
        B, T, C, S, _ = x.shape

        x_flat = x.reshape(B * T, C, S, S)
        e1, e2, e3 = self._encode_per_frame(x_flat)

        c3 = e3.shape[1]
        Hb, Wb = e3.shape[-2:]

        if self.temporal is not None:
            e3_seq = e3.reshape(B, T, c3, Hb, Wb)
            temporal_out, _ = self.temporal(e3_seq)
            bottleneck_seq = temporal_out
        else:
            bottleneck_seq = e3.reshape(B, T, c3, Hb, Wb)

        bottleneck_flat = bottleneck_seq.reshape(B * T, c3, Hb, Wb)
        d1 = self._decode_per_frame(bottleneck_flat, e1, e2)
        heatmap_logits = self.heatmap_head(d1).reshape(B, T, 1, S, S)

        visibility_flat = self.visibility_head(bottleneck_flat).reshape(B, T)

        # Aggregate temporal features for the track head.
        pooled_spatial = F.adaptive_avg_pool2d(bottleneck_flat, 1).reshape(B, T, c3)
        if self.cfg.track_pool == "max":
            pooled, _ = pooled_spatial.max(dim=1)
        else:
            pooled = pooled_spatial.mean(dim=1)
        track_logit = self.track_head(pooled).reshape(B)

        return {
            "track_logit": track_logit,
            "visibility_logits": visibility_flat,
            "heatmap_logits": heatmap_logits,
        }


def _crop_to(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Center-crop or pad ``x`` so its (H,W) match ``target``.

    Necessary because for odd crop sizes (e.g. 31) the down/up sampling can
    introduce a 1-pixel mismatch.
    """
    if x.shape[-2:] == target.shape[-2:]:
        return x
    _, _, hx, wx = x.shape
    _, _, ht, wt = target.shape
    # Pad if too small
    pad_h = max(0, ht - hx)
    pad_w = max(0, wt - wx)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
        hx += pad_h
        wx += pad_w
    # Crop if too large
    dh = hx - ht
    dw = wx - wt
    if dh > 0 or dw > 0:
        top = dh // 2
        left = dw // 2
        x = x[:, :, top : top + ht, left : left + wt]
    return x


__all__ = ["TrackletRecurrentUNet", "TrackletRecurrentUNetConfig"]
