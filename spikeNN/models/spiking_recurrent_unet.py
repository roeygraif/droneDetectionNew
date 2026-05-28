"""SpikingTrackletUNet — SNN mirror of TrackletRecurrentUNet.

Drop-in alternative to the existing detector. Same I/O contract:
  Input:  x of shape (B, T, C, S, S), float32.
  Output: dict with
    - track_logit:        (B,)
    - visibility_logits:  (B, T)
    - heatmap_logits:     (B, T, 1, S, S)

Architecture (mirrors recurrent_unet.py topology):
  - enc1: ConvLIFBlock(C_in -> c1)                  [membrane resets per frame]
  - pool -> enc2: ConvLIFBlock(c1 -> c2)            [membrane resets per frame]
  - pool -> enc3: ConvLIFBlock(c2 -> c3)            [membrane resets per frame]
  - SpikingRecurrentBlock(c3)                       [membrane PERSISTS across T]
  - up   -> dec2: ConvLIFBlock(c3 -> c2) + skip e2  [resets]
  - up   -> dec1: ConvLIFBlock(c2 -> c1) + skip e1  [resets]
  - heatmap_head:    Conv2d(c1 -> 1)  applied to dec1 spike output
  - visibility_head: Linear over global-avg-pooled recurrent spikes
  - track_head:      Linear over mean-over-T of pooled recurrent spikes

Encoding: direct (raw float pixels into enc1). The recurrent block carries
membrane potential across all T frames so temporal integration is native to
the LIF dynamics rather than gated by explicit LSTM cells.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikeNN.models.lif_blocks import ConvLIFBlock, SpikingRecurrentBlock


@dataclass
class SpikingTrackletUNetConfig:
    in_channels: int = 5
    base_channels: int = 16
    bottleneck_channels: int = 32
    crop_size: int = 15
    beta_encoder: float = 0.9       # decay for per-frame blocks
    beta_recurrent: float = 0.95    # longer decay for temporal bottleneck
    threshold: float = 1.0
    surrogate_slope: float = 25.0
    # "spike_rate" (V1) feeds the heads the binary spike outputs of the
    # recurrent block + dec1 (information is lost at the threshold).
    # "membrane" (V2) feeds the heads the continuous membrane potentials,
    # preserving amplitude info at the output while keeping spiking dynamics
    # inside the network — this is the principled fix for the matched-filter
    # information-loss diagnosed in the V1 null result.
    readout_mode: str = "spike_rate"


class SpikingTrackletUNet(nn.Module):
    def __init__(self, cfg: SpikingTrackletUNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or SpikingTrackletUNetConfig()
        c1 = self.cfg.base_channels
        c2 = c1 * 2
        c3 = self.cfg.bottleneck_channels

        # Encoder (per-frame; membrane resets every timestep).
        self.enc1 = ConvLIFBlock(self.cfg.in_channels, c1,
                                 beta=self.cfg.beta_encoder,
                                 threshold=self.cfg.threshold,
                                 slope=self.cfg.surrogate_slope)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvLIFBlock(c1, c2,
                                 beta=self.cfg.beta_encoder,
                                 threshold=self.cfg.threshold,
                                 slope=self.cfg.surrogate_slope)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvLIFBlock(c2, c3,
                                 beta=self.cfg.beta_encoder,
                                 threshold=self.cfg.threshold,
                                 slope=self.cfg.surrogate_slope)

        # Recurrent bottleneck (membrane and prev-spike carry across all T frames).
        self.temporal = SpikingRecurrentBlock(c3,
                                              beta=self.cfg.beta_recurrent,
                                              threshold=self.cfg.threshold,
                                              slope=self.cfg.surrogate_slope)

        # Decoder (per-frame; membrane resets every timestep).
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvLIFBlock(c2 + c2, c2,
                                 beta=self.cfg.beta_encoder,
                                 threshold=self.cfg.threshold,
                                 slope=self.cfg.surrogate_slope)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvLIFBlock(c1 + c1, c1,
                                 beta=self.cfg.beta_encoder,
                                 threshold=self.cfg.threshold,
                                 slope=self.cfg.surrogate_slope)

        # Heads (analog conv/linear on spike outputs — NOT spiking, since we
        # want logits not binary outputs for the loss).
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
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(
                f"SpikingTrackletUNet expects (B,T,C,S,S); got {tuple(x.shape)}"
            )
        B, T, C, S, _ = x.shape
        device = x.device

        # Initialise recurrent block state (membrane + prev-spike). Both
        # persist across the whole T-step loop, giving the bottleneck temporal
        # memory.
        # We need a dummy tensor with the bottleneck spatial shape — easiest
        # is to do a single dry-run encode and read the shape, but cheaper to
        # compute analytically: two pool/2 reduce 15 -> 7 -> 3 (floor div).
        # Use that to allocate the recurrent state. We verify with a real
        # forward at first frame.
        # Approach: run frame 0 through encoder to find shapes, then allocate.

        # Choose which signal feeds the heads.
        # V1 ("spike_rate"): binary spike outputs of the recurrent block + dec1
        #   — information is thresholded away at every layer.
        # V2 ("membrane"):   continuous membrane potentials — preserves amplitude
        #   info at the output while keeping spiking dynamics inside.
        use_membrane_readout = (self.cfg.readout_mode == "membrane")

        per_frame_dec1_signal: list[torch.Tensor] = []
        per_frame_pooled_recurrent: list[torch.Tensor] = []

        rec_mem: torch.Tensor | None = None
        rec_prev_spk: torch.Tensor | None = None

        for t in range(T):
            x_t = x[:, t]                                    # (B, C, S, S)

            # Encoder — fresh membrane each frame.
            mem_e1 = self.enc1.init_mem(x_t)
            spk_e1, _ = self.enc1(x_t, mem_e1)              # (B, c1, S, S)

            p1 = self.pool1(spk_e1)                          # (B, c1, S/2, S/2)
            mem_e2 = self.enc2.init_mem(p1)
            spk_e2, _ = self.enc2(p1, mem_e2)                # (B, c2, S/2, S/2)

            p2 = self.pool2(spk_e2)                          # (B, c2, S/4, S/4)
            mem_e3 = self.enc3.init_mem(p2)
            spk_e3, _ = self.enc3(p2, mem_e3)                # (B, c3, S/4, S/4)

            # Recurrent bottleneck — state persists across frames.
            if rec_mem is None:
                rec_mem, rec_prev_spk = self.temporal.init_state(spk_e3)
            spk_r, rec_mem = self.temporal(spk_e3, rec_mem, rec_prev_spk)
            rec_prev_spk = spk_r                              # carry to next t

            # Decoder — fresh membrane each frame, skip-connections from encoder spikes.
            u2 = self.up2(spk_r)                              # (B, c2, S/2-ish, ..)
            u2 = _crop_to(u2, spk_e2)
            cat2 = torch.cat([u2, spk_e2], dim=1)            # (B, 2*c2, S/2, S/2)
            mem_d2 = self.dec2.init_mem(cat2)
            spk_d2, _ = self.dec2(cat2, mem_d2)              # (B, c2, S/2, S/2)

            u1 = self.up1(spk_d2)                             # (B, c1, S, S)
            u1 = _crop_to(u1, spk_e1)
            cat1 = torch.cat([u1, spk_e1], dim=1)            # (B, 2*c1, S, S)
            mem_d1 = self.dec1.init_mem(cat1)
            spk_d1, mem_d1_out = self.dec1(cat1, mem_d1)     # (B, c1, S, S)

            # Per-frame head inputs.
            if use_membrane_readout:
                # Continuous membrane preserves amplitude information.
                per_frame_dec1_signal.append(mem_d1_out)
                pooled = F.adaptive_avg_pool2d(rec_mem, 1).flatten(1)
            else:
                # Spike-rate (V1 behaviour).
                per_frame_dec1_signal.append(spk_d1)
                pooled = F.adaptive_avg_pool2d(spk_r, 1).flatten(1)
            per_frame_pooled_recurrent.append(pooled)

        # Stack across time.
        dec1_stack = torch.stack(per_frame_dec1_signal, dim=1)        # (B, T, c1, S, S)
        rec_pool_stack = torch.stack(per_frame_pooled_recurrent, 1)    # (B, T, c3)

        # heatmap_logits: per-frame Conv on dec1 spikes (which are 0/1)
        # → reshape to (B*T, c1, S, S) for the Conv head
        dec1_flat = dec1_stack.reshape(B * T, -1, S, S)
        heatmap_logits = self.heatmap_head(dec1_flat).reshape(B, T, 1, S, S)

        # visibility_logits: per-frame Linear on (B, T, c3) pooled recurrent spikes
        # The visibility_head expects (B, c3, 1, 1) shape; adapt.
        # We reshape rec_pool_stack -> (B*T, c3, 1, 1) and apply head.
        vis_in = rec_pool_stack.reshape(B * T, -1, 1, 1)
        visibility_logits = self.visibility_head(vis_in).reshape(B, T)

        # track_logit: mean over T then Linear
        pooled_track = rec_pool_stack.mean(dim=1)                      # (B, c3)
        track_logit = self.track_head(pooled_track).reshape(B)

        return {
            "track_logit": track_logit,
            "visibility_logits": visibility_logits,
            "heatmap_logits": heatmap_logits,
        }


def _crop_to(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Center-crop or pad x to match target's H,W (handles odd crop sizes)."""
    if x.shape[-2:] == target.shape[-2:]:
        return x
    _, _, hx, wx = x.shape
    _, _, ht, wt = target.shape
    pad_h = max(0, ht - hx)
    pad_w = max(0, wt - wx)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
        hx += pad_h
        wx += pad_w
    dh = hx - ht
    dw = wx - wt
    if dh > 0 or dw > 0:
        top = dh // 2
        left = dw // 2
        x = x[:, :, top:top + ht, left:left + wt]
    return x


__all__ = ["SpikingTrackletUNet", "SpikingTrackletUNetConfig"]
