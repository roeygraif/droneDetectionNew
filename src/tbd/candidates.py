"""Per-frame weak-candidate extraction.

Given an evidence map (T,H,W), each frame is reduced to a short list of local
maxima after non-maximum suppression. The goal here is recall, not precision —
at the SNRs the thesis targets, the real UAV is often *not* the brightest
local maximum, so we keep a generous top-K and let the tracklet builder /
classifier sort signal from clutter downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Candidate:
    """One weak candidate point in one frame."""

    t: int
    y: float
    x: float
    score: float


@dataclass
class CandidateConfig:
    """Knobs for candidate extraction."""

    candidate_top_k: int = 200
    candidate_nms_radius: int = 3
    candidate_min_score: float | None = None
    crop_size: int = 31


def _nms_local_maxima(score_map: np.ndarray, radius: int) -> np.ndarray:
    """Return a boolean mask of strict local maxima within ``radius``.

    Implementation: per-pixel max over the (2r+1)x(2r+1) neighborhood via a
    sliding-window view with reflect padding, then equality with the original.
    This is O(H*W*(2r+1)^2) but ``radius`` is tiny (default 3) so it's fast.
    """
    if score_map.ndim != 2:
        raise ValueError(f"score_map must be 2D, got {score_map.shape}")
    r = max(0, int(radius))
    if r == 0:
        return np.ones_like(score_map, dtype=bool)

    H, W = score_map.shape
    w = 2 * r + 1
    pad = np.pad(score_map, ((r, r), (r, r)), mode="reflect")
    shape = (H, W, w, w)
    s0, s1 = pad.strides
    strides = (s0, s1, s0, s1)
    windows = np.lib.stride_tricks.as_strided(pad, shape=shape, strides=strides, writeable=False)
    nb_max = windows.max(axis=(2, 3))
    return score_map >= nb_max - 1e-9


def _border_mask(shape: tuple[int, int], crop_size: int) -> np.ndarray:
    """True where a centered crop of size ``crop_size`` fits within ``shape``."""
    H, W = shape
    half = max(0, int(crop_size) // 2)
    mask = np.zeros((H, W), dtype=bool)
    if H > 2 * half and W > 2 * half:
        mask[half : H - half, half : W - half] = True
    return mask


def extract_candidates(
    evidence: np.ndarray,
    cfg: CandidateConfig | None = None,
) -> list[list[Candidate]]:
    """Extract top-K NMS local maxima per frame.

    Returns a list (length T) of per-frame candidate lists, each sorted by
    descending score.
    """
    if cfg is None:
        cfg = CandidateConfig()

    evidence = np.asarray(evidence, dtype=np.float32)
    if evidence.ndim != 3:
        raise ValueError(f"evidence must be (T,H,W); got {evidence.shape}")
    T, H, W = evidence.shape

    border = _border_mask((H, W), cfg.crop_size)
    out: list[list[Candidate]] = []

    for t in range(T):
        frame = evidence[t]
        peaks = _nms_local_maxima(frame, cfg.candidate_nms_radius)
        valid = peaks & border
        if cfg.candidate_min_score is not None:
            valid &= frame >= float(cfg.candidate_min_score)

        ys, xs = np.nonzero(valid)
        if ys.size == 0:
            out.append([])
            continue

        scores = frame[ys, xs]
        # Sort descending, then truncate.
        order = np.argsort(-scores, kind="stable")
        k = max(0, int(cfg.candidate_top_k))
        order = order[:k]

        cands = [
            Candidate(t=int(t), y=float(ys[i]), x=float(xs[i]), score=float(scores[i]))
            for i in order
        ]
        out.append(cands)

    return out


__all__ = ["Candidate", "CandidateConfig", "extract_candidates"]
