"""LIAFTrackletUNet — V3 hybrid SNN (analog fire / LIAF) version.

Topology identical to SpikingTrackletUNet, but every hidden layer uses LIAF
(continuous analog output) instead of LIF (binary spike output). This
preserves amplitude information through the network, addressing the
matched-filter information-loss diagnosed in the V1/V2 experiments at the
−6 dB band where the residual gap to ConvLSTM lives.

Trade-off: V3 gives up the binary-spike / energy-efficiency property of true
SNNs. It's a "spike-free SNN" — keeps the LIF leaky-integrator temporal
dynamics but replaces firing with bounded continuous activation. The temporal
integration mechanism (matched-filter-like) is preserved; the binary
thresholding (information-discarding) is removed.

I/O contract matches the baseline TrackletRecurrentUNet exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikeNN.models.liaf_blocks import ConvLIAFBlock, AnalogRecurrentBlock


@dataclass
class LIAFTrackletUNetConfig:
    in_channels: int = 5
    base_channels: int = 24            # matches V2 / baseline param budget
    bottleneck_channels: int = 48
    crop_size: int = 15
    beta_encoder: float = 0.9
    beta_recurrent: float = 0.95
    threshold: float = 1.0
    sigmoid_slope: float = 4.0


class LIAFTrackletUNet(nn.Module):
    def __init__(self, cfg: LIAFTrackletUNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or LIAFTrackletUNetConfig()
        c1 = self.cfg.base_channels
        c2 = c1 * 2
        c3 = self.cfg.bottleneck_channels

        # Encoder — analog-fire per frame.
        self.enc1 = ConvLIAFBlock(self.cfg.in_channels, c1, beta=self.cfg.beta_encoder,
                                  threshold=self.cfg.threshold, slope=self.cfg.sigmoid_slope)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvLIAFBlock(c1, c2, beta=self.cfg.beta_encoder,
                                  threshold=self.cfg.threshold, slope=self.cfg.sigmoid_slope)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvLIAFBlock(c2, c3, beta=self.cfg.beta_encoder,
                                  threshold=self.cfg.threshold, slope=self.cfg.sigmoid_slope)

        # Recurrent bottleneck — analog with persistent membrane.
        self.temporal = AnalogRecurrentBlock(c3, beta=self.cfg.beta_recurrent,
                                             threshold=self.cfg.threshold,
                                             slope=self.cfg.sigmoid_slope)

        # Decoder — analog per frame.
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvLIAFBlock(c2 + c2, c2, beta=self.cfg.beta_encoder,
                                  threshold=self.cfg.threshold, slope=self.cfg.sigmoid_slope)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvLIAFBlock(c1 + c1, c1, beta=self.cfg.beta_encoder,
                                  threshold=self.cfg.threshold, slope=self.cfg.sigmoid_slope)

        # Heads (linear/conv on analog outputs).
        self.heatmap_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1),
            nn.Linear(c3, c3 // 2), nn.ReLU(inplace=True),
            nn.Linear(c3 // 2, 1),
        )
        self.track_head = nn.Sequential(
            nn.Linear(c3, c3 // 2), nn.ReLU(inplace=True),
            nn.Linear(c3 // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"LIAFTrackletUNet expects (B,T,C,S,S); got {tuple(x.shape)}")
        B, T, C, S, _ = x.shape

        per_frame_dec1: list[torch.Tensor] = []
        per_frame_pooled_rec: list[torch.Tensor] = []
        rec_mem: torch.Tensor | None = None
        rec_prev_y: torch.Tensor | None = None

        for t in range(T):
            x_t = x[:, t]

            # Encoder.
            mem_e1 = self.enc1.init_mem(x_t)
            y_e1, _ = self.enc1(x_t, mem_e1)
            p1 = self.pool1(y_e1)
            mem_e2 = self.enc2.init_mem(p1)
            y_e2, _ = self.enc2(p1, mem_e2)
            p2 = self.pool2(y_e2)
            mem_e3 = self.enc3.init_mem(p2)
            y_e3, _ = self.enc3(p2, mem_e3)

            # Recurrent bottleneck (persistent state).
            if rec_mem is None:
                rec_mem, rec_prev_y = self.temporal.init_state(y_e3)
            y_r, rec_mem = self.temporal(y_e3, rec_mem, rec_prev_y)
            rec_prev_y = y_r

            # Decoder.
            u2 = self.up2(y_r)
            u2 = _crop_to(u2, y_e2)
            cat2 = torch.cat([u2, y_e2], dim=1)
            mem_d2 = self.dec2.init_mem(cat2)
            y_d2, _ = self.dec2(cat2, mem_d2)

            u1 = self.up1(y_d2)
            u1 = _crop_to(u1, y_e1)
            cat1 = torch.cat([u1, y_e1], dim=1)
            mem_d1 = self.dec1.init_mem(cat1)
            y_d1, _ = self.dec1(cat1, mem_d1)

            per_frame_dec1.append(y_d1)
            pooled = F.adaptive_avg_pool2d(y_r, 1).flatten(1)
            per_frame_pooled_rec.append(pooled)

        dec1_stack = torch.stack(per_frame_dec1, dim=1)
        rec_pool_stack = torch.stack(per_frame_pooled_rec, 1)

        dec1_flat = dec1_stack.reshape(B * T, -1, S, S)
        heatmap_logits = self.heatmap_head(dec1_flat).reshape(B, T, 1, S, S)

        vis_in = rec_pool_stack.reshape(B * T, -1, 1, 1)
        visibility_logits = self.visibility_head(vis_in).reshape(B, T)

        pooled_track = rec_pool_stack.mean(dim=1)
        track_logit = self.track_head(pooled_track).reshape(B)

        return {
            "track_logit": track_logit,
            "visibility_logits": visibility_logits,
            "heatmap_logits": heatmap_logits,
        }


def _crop_to(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == target.shape[-2:]:
        return x
    _, _, hx, wx = x.shape
    _, _, ht, wt = target.shape
    pad_h = max(0, ht - hx); pad_w = max(0, wt - wx)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
        hx += pad_h; wx += pad_w
    dh = hx - ht; dw = wx - wt
    if dh > 0 or dw > 0:
        top = dh // 2; left = dw // 2
        x = x[:, :, top:top + ht, left:left + wt]
    return x


__all__ = ["LIAFTrackletUNet", "LIAFTrackletUNetConfig"]
