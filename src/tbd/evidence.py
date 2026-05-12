"""Per-frame weak evidence maps.

Each input frame is transformed into several intermediate maps that highlight
plausible UAV-like signal while still being permissive enough to keep recall
high at low SNR. The combined ``evidence`` map is what the candidate extractor
consumes; the intermediate maps are kept around because the crop-tube stage
re-uses them as additional input channels for the model.

All maps are computed in numpy (cheap; well below the model's compute cost)
and returned with shape (T,H,W) float32. Inputs may be torch tensors or numpy
arrays — they are normalized to numpy float32 internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

try:  # torch is required by the project; keep the import optional only for static analysis.
    import torch
    _ArrayLike = Union[np.ndarray, "torch.Tensor"]
except Exception:  # pragma: no cover - torch is in pyproject.toml deps
    torch = None  # type: ignore
    _ArrayLike = np.ndarray  # type: ignore


_EPS = 1e-6


@dataclass
class EvidenceConfig:
    """Knobs for evidence-map computation."""

    evidence_local_window: int = 9
    evidence_gaussian_sigma: float = 1.0
    evidence_w_local_z: float = 0.4
    evidence_w_matched: float = 0.4
    evidence_w_temporal_diff: float = 0.2


def _to_numpy(frames: _ArrayLike) -> np.ndarray:
    if torch is not None and isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"frames must have shape (T,H,W); got {arr.shape}")
    return arr


def _box_blur(img: np.ndarray, window: int) -> np.ndarray:
    """Separable box blur with reflect padding. Window forced odd, >= 1."""
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    if w == 1:
        return img.copy()
    r = w // 2

    pad = np.pad(img, ((r, r), (r, r)), mode="reflect").astype(np.float64, copy=False)
    H_pad, W_pad = pad.shape

    cs0 = np.zeros((H_pad + 1, W_pad), dtype=np.float64)
    cs0[1:] = np.cumsum(pad, axis=0)
    row_sum = cs0[w:] - cs0[:-w]  # (H, W_pad)

    cs1 = np.zeros((row_sum.shape[0], W_pad + 1), dtype=np.float64)
    cs1[:, 1:] = np.cumsum(row_sum, axis=1)
    out = cs1[:, w:] - cs1[:, :-w]  # (H, W)
    return (out / (w * w)).astype(np.float32, copy=False)


def _gaussian_kernel_2d(sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-3)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    g1 = np.exp(-0.5 * (ax / sigma) ** 2)
    g1 /= g1.sum()
    k = np.outer(g1, g1).astype(np.float32)
    return k


def _conv2_reflect(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Direct 2D correlation with reflect padding (small kernels only)."""
    kh, kw = kernel.shape
    rh, rw = kh // 2, kw // 2
    pad = np.pad(img, ((rh, rh), (rw, rw)), mode="reflect")
    H, W = img.shape
    # Build a 4D windowed view: (H, W, kh, kw)
    shape = (H, W, kh, kw)
    s0, s1 = pad.strides
    strides = (s0, s1, s0, s1)
    windows = np.lib.stride_tricks.as_strided(pad, shape=shape, strides=strides, writeable=False)
    return np.einsum("ijkl,kl->ij", windows, kernel).astype(np.float32, copy=False)


def _local_z(frame: np.ndarray, window: int) -> np.ndarray:
    mean = _box_blur(frame, window)
    mean_sq = _box_blur(frame * frame, window)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var + _EPS)
    return ((frame - mean) / (std + _EPS)).astype(np.float32, copy=False)


def _matched_filter(frame: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    resp = _conv2_reflect(frame, kernel)
    # Normalize to roughly unit RMS so downstream weighting is stable across
    # frame energies. Use robust stats so a single hot pixel can't blow it up.
    med = float(np.median(resp))
    mad = float(np.median(np.abs(resp - med)) + _EPS)
    scale = 1.4826 * mad  # MAD -> sigma estimate for gaussian-ish noise
    return ((resp - med) / (scale + _EPS)).astype(np.float32, copy=False)


def _temporal_diff(stack: np.ndarray) -> np.ndarray:
    T = stack.shape[0]
    out = np.zeros_like(stack)
    if T >= 2:
        out[1:] = np.abs(stack[1:] - stack[:-1])
    return out


def _normalize_raw(stack: np.ndarray) -> np.ndarray:
    """Robust z-score over the whole stack so 'raw' is in a stable range."""
    med = float(np.median(stack))
    mad = float(np.median(np.abs(stack - med)) + _EPS)
    scale = 1.4826 * mad
    return ((stack - med) / (scale + _EPS)).astype(np.float32, copy=False)


def compute_evidence_maps(
    frames: _ArrayLike,
    cfg: EvidenceConfig | None = None,
) -> dict[str, np.ndarray]:
    """Compute raw / local_z / matched / temporal_diff / combined evidence maps.

    Returns a dict of numpy arrays each shaped (T,H,W) float32.
    """
    if cfg is None:
        cfg = EvidenceConfig()

    stack = _to_numpy(frames)
    T, H, W = stack.shape

    raw = _normalize_raw(stack)

    local_z = np.empty_like(stack)
    for t in range(T):
        local_z[t] = _local_z(stack[t], cfg.evidence_local_window)

    kernel = _gaussian_kernel_2d(cfg.evidence_gaussian_sigma)
    matched = np.empty_like(stack)
    for t in range(T):
        matched[t] = _matched_filter(stack[t], kernel)

    temporal_diff = _temporal_diff(stack).astype(np.float32, copy=False)

    evidence = (
        cfg.evidence_w_local_z * local_z
        + cfg.evidence_w_matched * matched
        + cfg.evidence_w_temporal_diff * temporal_diff
    ).astype(np.float32, copy=False)

    return {
        "raw": raw,
        "local_z": local_z,
        "matched": matched,
        "temporal_diff": temporal_diff,
        "evidence": evidence,
    }


__all__ = ["EvidenceConfig", "compute_evidence_maps"]
