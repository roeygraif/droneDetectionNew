"""Crop-tube extraction.

A *crop tube* is a small spatial window followed across frames along a
tracklet. The model classifies each tube as UAV or false alarm. We stack
multiple evidence channels per frame so the model gets both the raw signal
and pre-computed weak features for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from tbd.tracklets import Tracklet


@dataclass
class CropTubeConfig:
    """Knobs for crop-tube extraction."""

    crop_size: int = 31
    channels: tuple[str, ...] = ("raw", "local_z", "matched", "temporal_diff", "evidence")


def _extract_one_crop(
    canvas: np.ndarray,
    cy: float,
    cx: float,
    crop_size: int,
) -> np.ndarray:
    """Extract a (S,S) crop centered at sub-pixel ``(cy,cx)`` with zero padding.

    We snap the center to the nearest pixel — the model is small enough that
    sub-pixel re-sampling is not worth the complexity at this stage.
    """
    H, W = canvas.shape
    S = int(crop_size)
    half = S // 2
    # For even S we still want a deterministic crop window.
    cy_i = int(round(cy))
    cx_i = int(round(cx))
    y0 = cy_i - half
    x0 = cx_i - half
    y1 = y0 + S
    x1 = x0 + S

    out = np.zeros((S, S), dtype=np.float32)

    src_y0 = max(0, y0)
    src_x0 = max(0, x0)
    src_y1 = min(H, y1)
    src_x1 = min(W, x1)
    if src_y0 >= src_y1 or src_x0 >= src_x1:
        return out

    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = canvas[src_y0:src_y1, src_x0:src_x1]
    return out


def _resolve_centers(tracklet: Tracklet, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (centers, valid_mask) with shape (T,2) and (T,).

    Missing positions are predicted by constant velocity from the last
    observed position when possible; otherwise we hold the last known
    position (or copy the next future one, scanning backward at the end).
    ``valid_mask`` is False on frames where the tracklet did not observe a
    candidate (so the model can down-weight them and the dataset can ignore
    them in the heatmap loss).
    """
    centers = np.full((T, 2), np.nan, dtype=np.float32)
    valid = np.zeros(T, dtype=bool)

    positions = tracklet.positions
    # First pass: copy observed positions.
    observed_idx: list[int] = []
    for t in range(T):
        if t < tracklet.start_t or t >= tracklet.end_t:
            continue
        local = t - tracklet.start_t
        if local >= positions.shape[0]:
            continue
        py = positions[local, 0]
        px = positions[local, 1]
        if np.isfinite(py) and np.isfinite(px):
            centers[t] = (py, px)
            valid[t] = True
            observed_idx.append(t)

    if not observed_idx:
        return centers, valid

    # Estimate velocity from the observed points (last two if available).
    if len(observed_idx) >= 2:
        a = observed_idx[-2]
        b = observed_idx[-1]
        vy = (centers[b, 0] - centers[a, 0]) / max(1, b - a)
        vx = (centers[b, 1] - centers[a, 1]) / max(1, b - a)
    else:
        vy = 0.0
        vx = 0.0

    # Forward-fill / extrapolate using constant velocity from the last observed.
    last_t = observed_idx[-1]
    last_pos = centers[last_t].copy()
    for t in range(last_t + 1, T):
        dt = t - last_t
        centers[t, 0] = last_pos[0] + vy * dt
        centers[t, 1] = last_pos[1] + vx * dt

    # Back-fill before the first observed (use velocity if 2+ points, else hold).
    first_t = observed_idx[0]
    first_pos = centers[first_t].copy()
    for t in range(first_t - 1, -1, -1):
        dt = first_t - t
        centers[t, 0] = first_pos[0] - vy * dt
        centers[t, 1] = first_pos[1] - vx * dt

    # Interior gaps between observations: linear interpolation.
    for i in range(1, len(observed_idx)):
        a = observed_idx[i - 1]
        b = observed_idx[i]
        if b - a <= 1:
            continue
        for t in range(a + 1, b):
            alpha = (t - a) / (b - a)
            centers[t, 0] = centers[a, 0] * (1.0 - alpha) + centers[b, 0] * alpha
            centers[t, 1] = centers[a, 1] * (1.0 - alpha) + centers[b, 1] * alpha

    return centers, valid


def extract_crop_tube(
    frames: np.ndarray,
    evidence_maps: dict[str, np.ndarray],
    tracklet: Tracklet,
    cfg: CropTubeConfig | None = None,
) -> dict:
    """Extract a (T,C,S,S) crop tube along ``tracklet``.

    Returns:
        - ``crop_tube``: torch.FloatTensor (T,C,S,S)
        - ``valid_mask``: torch.FloatTensor (T,)  -- 1 where the tracklet had
          an observed candidate at that frame, 0 where the center was
          interpolated/extrapolated
        - ``crop_centers``: torch.FloatTensor (T,2)  -- (y, x) center used for
          each frame
    """
    if cfg is None:
        cfg = CropTubeConfig()

    frames_np = np.asarray(frames, dtype=np.float32)
    if frames_np.ndim != 3:
        raise ValueError(f"frames must be (T,H,W); got {frames_np.shape}")
    T, H, W = frames_np.shape

    # Map "raw" channel to the input frames if the evidence dict doesn't
    # already include it (it should, but be permissive).
    sources: dict[str, np.ndarray] = {"raw": frames_np}
    sources.update({k: np.asarray(v, dtype=np.float32) for k, v in evidence_maps.items()})

    for ch in cfg.channels:
        if ch not in sources:
            raise KeyError(f"Channel {ch!r} not present in evidence_maps")
        if sources[ch].shape != (T, H, W):
            raise ValueError(
                f"Channel {ch!r} has shape {sources[ch].shape}, expected {(T, H, W)}"
            )

    centers, valid = _resolve_centers(tracklet, T)

    C = len(cfg.channels)
    S = int(cfg.crop_size)
    tube = np.zeros((T, C, S, S), dtype=np.float32)

    for t in range(T):
        cy = float(centers[t, 0]) if np.isfinite(centers[t, 0]) else float(H // 2)
        cx = float(centers[t, 1]) if np.isfinite(centers[t, 1]) else float(W // 2)
        for ci, name in enumerate(cfg.channels):
            tube[t, ci] = _extract_one_crop(sources[name][t], cy, cx, S)

    return {
        "crop_tube": torch.from_numpy(tube),
        "valid_mask": torch.from_numpy(valid.astype(np.float32)),
        "crop_centers": torch.from_numpy(centers),
    }


def extract_crop_tubes_batch(
    frames: np.ndarray,
    evidence_maps: dict[str, np.ndarray],
    tracklets: Iterable[Tracklet],
    cfg: CropTubeConfig | None = None,
) -> list[dict]:
    """Extract crop tubes for every tracklet in one go (no batched IO trickery)."""
    return [extract_crop_tube(frames, evidence_maps, tr, cfg) for tr in tracklets]


__all__ = ["CropTubeConfig", "extract_crop_tube", "extract_crop_tubes_batch"]
