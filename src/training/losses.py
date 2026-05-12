"""Composite loss for the tracklet model.

    total_loss = lambda_track * L_track
               + lambda_heatmap * L_heatmap
               + lambda_visibility * L_visibility

L_heatmap and L_visibility are masked to *positive tracklets only* by default;
otherwise the all-zero targets on negatives dwarf any learning signal from
positives. The track loss can use focal loss to handle class imbalance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrackletLossConfig:
    """Loss weights and switches."""

    lambda_track: float = 1.0
    lambda_heatmap: float = 1.0
    lambda_visibility: float = 0.5
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    heatmap_negative_weight: float = 0.0  # set >0 to weakly supervise neg tubes
    visibility_negative_weight: float = 0.1


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Binary focal loss (Lin et al.). Shapes must match."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * ce
    return loss.mean()


class TrackletLoss(nn.Module):
    """Combined track / heatmap / visibility loss."""

    def __init__(self, cfg: TrackletLossConfig | None = None):
        super().__init__()
        self.cfg = cfg or TrackletLossConfig()

    def forward(
        self,
        outputs: dict[str, torch.Tensor | None],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        track_logit = outputs["track_logit"]                   # (B,)
        visibility_logits = outputs.get("visibility_logits")    # (B,T) or None
        heatmap_logits = outputs.get("heatmap_logits")          # (B,T,1,S,S) or None

        track_label = batch["track_label"].float().reshape(track_logit.shape)
        visibility_label = batch["visibility_label"].float()      # (B,T)
        heatmap_label = batch["heatmap_label"].float()            # (B,T,S,S)
        valid_mask = batch["valid_mask"].float()                  # (B,T)

        # --- track loss ----------------------------------------------------
        if self.cfg.use_focal_loss:
            l_track = focal_loss_with_logits(
                track_logit, track_label,
                alpha=self.cfg.focal_alpha, gamma=self.cfg.focal_gamma,
            )
        else:
            l_track = F.binary_cross_entropy_with_logits(track_logit, track_label)

        # Mask of positive tubes (B,) -> broadcast over T or T,S,S.
        is_pos = (track_label >= 0.5).float()
        pos_t = is_pos.unsqueeze(-1)                                # (B,1)
        # Soft per-frame weighting that down-weights negatives but doesn't zero
        # them out entirely (configurable).
        frame_weight = pos_t + (1.0 - pos_t) * 1.0  # placeholder, overridden below

        # --- visibility loss -----------------------------------------------
        l_visibility = track_logit.new_zeros(())
        if visibility_logits is not None:
            vis_target = visibility_label
            vis_logits = visibility_logits
            # weight: 1 on positive tubes, ``visibility_negative_weight`` else.
            vis_w = pos_t + (1.0 - pos_t) * self.cfg.visibility_negative_weight  # (B,1)
            vis_w = vis_w.expand_as(vis_logits)
            l_vis_per = F.binary_cross_entropy_with_logits(
                vis_logits, vis_target, reduction="none"
            )
            # Only count frames where the tracklet actually saw a candidate.
            l_vis_per = l_vis_per * vis_w * valid_mask
            denom = (vis_w * valid_mask).sum().clamp(min=1.0)
            l_visibility = l_vis_per.sum() / denom

        # --- heatmap loss --------------------------------------------------
        l_heatmap = track_logit.new_zeros(())
        if heatmap_logits is not None:
            hm_logits = heatmap_logits.squeeze(2)                  # (B,T,S,S)
            hm_target = heatmap_label
            hm_w = pos_t + (1.0 - pos_t) * self.cfg.heatmap_negative_weight  # (B,1)
            hm_w = hm_w.unsqueeze(-1).unsqueeze(-1).expand_as(hm_logits)
            valid_w = valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(hm_logits)
            per = F.binary_cross_entropy_with_logits(hm_logits, hm_target, reduction="none")
            per = per * hm_w * valid_w
            denom = (hm_w * valid_w).sum().clamp(min=1.0)
            l_heatmap = per.sum() / denom

        total = (
            self.cfg.lambda_track * l_track
            + self.cfg.lambda_heatmap * l_heatmap
            + self.cfg.lambda_visibility * l_visibility
        )
        return {
            "loss": total,
            "loss_track": l_track.detach(),
            "loss_heatmap": l_heatmap.detach(),
            "loss_visibility": l_visibility.detach(),
        }


__all__ = ["TrackletLoss", "TrackletLossConfig", "focal_loss_with_logits"]
