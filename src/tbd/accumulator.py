"""Track-length-aware evidence accumulation (classical TBD).

The per-frame candidate extractor processes each frame independently — at very
low SNR the true UAV is statistically indistinguishable from a strong noise
speckle in any single frame, so it almost never makes the per-frame top-K cut
and the beam-search tracklet builder never gets a chance.

This module takes the opposite approach: for each motion hypothesis
``(start_y, start_x, vy, vx)`` on a small grid, sum the evidence along the
predicted trajectory:

    acc(y, x, v) = sum_{t=0..T-1}  evidence[t, y + vy*t, x + vx*t]

The signal at the true motion hypothesis accumulates coherently across T
frames (~ T·A); noise accumulates incoherently (~ sqrt(T)·σ). At T=8 this is
worth ~9 dB of effective SNR — enough to dig the target out of the noise.

The output of the accumulator is a stack of ``V`` score maps (one per velocity
hypothesis). We take per-pixel max over V to collapse to a (H, W) map, run
NMS, and convert each peak into a :class:`tbd.tracklets.Tracklet` with
``positions[t] = (y0 + vy*t, x0 + vx*t)``. From there the rest of the pipeline
(crop tubes + recurrent U-Net) is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tbd.candidates import _nms_local_maxima
from tbd.tracklets import Tracklet


@dataclass
class AccumulatorConfig:
    """Knobs for the TBD accumulator."""

    speed_max_px_per_frame: float = 2.0
    n_speeds: int = 5                    # including the v=0 hypothesis
    n_directions: int = 8                # uniform headings
    accumulator_top_k: int = 100
    accumulator_nms_radius: int = 3
    accumulator_min_observed_points: int = 3
    crop_size: int = 31                  # for border filtering
    velocity_grid: tuple[tuple[float, float], ...] | None = None  # if set, overrides speeds/dirs
    # When True (default), per-frame shifts use bilinear interpolation instead
    # of integer rounding. Recovers most of the 1-3 frames of integration we'd
    # otherwise lose to velocity-grid quantization at off-grid trajectories.
    # Has no effect when (vy, vx) happens to be on the integer grid.
    bilinear: bool = True


def build_velocity_grid(cfg: AccumulatorConfig) -> np.ndarray:
    """Polar grid of velocity hypotheses; shape (V, 2) in (vy, vx)."""
    if cfg.velocity_grid:
        return np.asarray(cfg.velocity_grid, dtype=np.float32)
    n_speeds = max(2, int(cfg.n_speeds))
    n_dirs = max(1, int(cfg.n_directions))
    speeds = np.linspace(0.0, float(cfg.speed_max_px_per_frame), n_speeds)
    dirs = np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False)
    vels: list[tuple[float, float]] = [(0.0, 0.0)]
    for s in speeds[1:]:
        for d in dirs:
            vy = float(s * np.sin(d))
            vx = float(s * np.cos(d))
            vels.append((vy, vx))
    return np.asarray(vels, dtype=np.float32)


def _shift_integer_add(out: np.ndarray, frame: np.ndarray, dy: int, dx: int, weight: float) -> None:
    """In-place ``out += weight * frame_shifted_by_(dy, dx)`` with zero out-of-bounds."""
    H, W = frame.shape
    y_src_start = max(0, dy)
    y_src_end = min(H, H + dy)
    x_src_start = max(0, dx)
    x_src_end = min(W, W + dx)
    if y_src_start >= y_src_end or x_src_start >= x_src_end:
        return
    y_dst_start = y_src_start - dy
    x_dst_start = x_src_start - dx
    h = y_src_end - y_src_start
    w = x_src_end - x_src_start
    if weight == 1.0:
        out[y_dst_start : y_dst_start + h, x_dst_start : x_dst_start + w] += (
            frame[y_src_start:y_src_end, x_src_start:x_src_end]
        )
    else:
        out[y_dst_start : y_dst_start + h, x_dst_start : x_dst_start + w] += (
            weight * frame[y_src_start:y_src_end, x_src_start:x_src_end]
        )


def _shift_and_sum(evidence: np.ndarray, vy: float, vx: float, bilinear: bool = True) -> np.ndarray:
    """Velocity-aligned cumulative sum over the time axis.

    Returns ``(H, W)`` where ``out[y, x] = sum_t evidence[t, y + vy*t, x + vx*t]``.
    Out-of-bound pixels contribute zero (so partial tracks score lower than
    fully-in-bound ones).

    With ``bilinear=True`` the per-frame shift is sub-pixel, computed as a
    weighted sum over the four integer-shifted neighbors:

        I[y + δy, x + δx] = (1-α)(1-β) I[y+i_y, x+i_x]
                          + α    (1-β) I[y+i_y+1, x+i_x]
                          + (1-α)β    I[y+i_y, x+i_x+1]
                          + α     β    I[y+i_y+1, x+i_x+1]

    where i_y = floor(δy), α = δy - i_y, etc. For integer offsets this reduces
    to the original nearest-pixel shift exactly (α=β=0 → only the first term).
    """
    T, H, W = evidence.shape
    out = np.zeros((H, W), dtype=np.float32)
    for t in range(T):
        dy = vy * t
        dx = vx * t
        if not bilinear:
            _shift_integer_add(out, evidence[t], int(round(dy)), int(round(dx)), 1.0)
            continue
        dy_i = int(np.floor(dy))
        dx_i = int(np.floor(dx))
        alpha = dy - dy_i
        beta = dx - dx_i
        w00 = (1.0 - alpha) * (1.0 - beta)
        w10 = alpha         * (1.0 - beta)
        w01 = (1.0 - alpha) * beta
        w11 = alpha         * beta
        if w00 > 0.0:
            _shift_integer_add(out, evidence[t], dy_i,     dx_i,     w00)
        if w10 > 0.0:
            _shift_integer_add(out, evidence[t], dy_i + 1, dx_i,     w10)
        if w01 > 0.0:
            _shift_integer_add(out, evidence[t], dy_i,     dx_i + 1, w01)
        if w11 > 0.0:
            _shift_integer_add(out, evidence[t], dy_i + 1, dx_i + 1, w11)
    return out


def accumulate_tracks(
    evidence: np.ndarray,
    cfg: AccumulatorConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """For each velocity hypothesis, compute the velocity-aligned evidence sum.

    Returns ``(scores, velocity_grid)``:
      - ``scores``: ``(V, H, W)`` float32
      - ``velocity_grid``: ``(V, 2)`` float32
    """
    if cfg is None:
        cfg = AccumulatorConfig()
    ev = np.asarray(evidence, dtype=np.float32)
    if ev.ndim != 3:
        raise ValueError(f"evidence must be (T,H,W); got {ev.shape}")
    T, H, W = ev.shape
    vels = build_velocity_grid(cfg)
    V = vels.shape[0]
    scores = np.empty((V, H, W), dtype=np.float32)
    for i in range(V):
        scores[i] = _shift_and_sum(ev, float(vels[i, 0]), float(vels[i, 1]),
                                   bilinear=cfg.bilinear)
    return scores, vels


def _border_mask(shape: tuple[int, int], crop_size: int) -> np.ndarray:
    H, W = shape
    half = max(0, int(crop_size) // 2)
    mask = np.zeros((H, W), dtype=bool)
    if H > 2 * half and W > 2 * half:
        mask[half : H - half, half : W - half] = True
    return mask


def extract_seed_tracklets(
    evidence: np.ndarray,
    cfg: AccumulatorConfig | None = None,
) -> list[Tracklet]:
    """Run the accumulator and return one :class:`Tracklet` per top peak.

    Each tracklet's positions are determined by the constant-velocity model
    that won at that pixel: ``positions[t] = (y0 + vy*t, x0 + vx*t)``.
    Positions that leave the image are set to NaN (the crop-tube extractor
    will then pad with zeros at those frames).
    """
    if cfg is None:
        cfg = AccumulatorConfig()
    ev = np.asarray(evidence, dtype=np.float32)
    T, H, W = ev.shape

    scores, vels = accumulate_tracks(ev, cfg)
    V = vels.shape[0]

    best_v_idx = scores.argmax(axis=0)             # (H, W) int64
    best_score = scores.max(axis=0)                # (H, W) float32

    peaks = _nms_local_maxima(best_score, cfg.accumulator_nms_radius)
    border = _border_mask((H, W), cfg.crop_size)
    valid = peaks & border
    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        return []

    peak_scores = best_score[ys, xs]
    order = np.argsort(-peak_scores, kind="stable")
    k = max(0, int(cfg.accumulator_top_k))
    order = order[:k]

    tracklets: list[Tracklet] = []
    for i in order:
        y0 = float(ys[i])
        x0 = float(xs[i])
        vy = float(vels[best_v_idx[ys[i], xs[i]], 0])
        vx = float(vels[best_v_idx[ys[i], xs[i]], 1])

        positions = np.full((T, 2), np.nan, dtype=np.float32)
        candidate_scores = np.zeros(T, dtype=np.float32)
        for t in range(T):
            py = y0 + vy * t
            px = x0 + vx * t
            yi = int(round(py))
            xi = int(round(px))
            if 0 <= yi < H and 0 <= xi < W:
                positions[t, 0] = np.float32(py)
                positions[t, 1] = np.float32(px)
                candidate_scores[t] = np.float32(ev[t, yi, xi])

        n_observed = int(np.isfinite(positions[:, 0]).sum())
        if n_observed < int(cfg.accumulator_min_observed_points):
            continue
        miss_count = T - n_observed

        tracklets.append(Tracklet(
            positions=positions,
            candidate_scores=candidate_scores,
            start_t=0,
            end_t=T,
            score=float(peak_scores[i]),
            miss_count=miss_count,
            velocity=np.array([vy, vx], dtype=np.float32),
            metadata={"source": "accumulator", "n_observed": n_observed},
        ))
    return tracklets


__all__ = [
    "AccumulatorConfig",
    "accumulate_tracks",
    "build_velocity_grid",
    "extract_seed_tracklets",
]
