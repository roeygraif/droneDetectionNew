"""Generalized Likelihood Ratio Test (GLRT) for unknown-trajectory detection.

The realistic NP-optimal detector when the target's initial position and
velocity are *unknown* nuisance parameters (composite hypothesis, Kay 1998).
The GLRT plugs in the maximum-likelihood estimates of these parameters:

  T_GLRT = max_{p₀, v}  Σ_t  Σ_{y,x}  frame[t,y,x] · g(y − (p₀_y + v_y·t),
                                                       x − (p₀_x + v_x·t); σ_PSF)

We evaluate by:
  1. FFT-correlating each frame with the unit-peak Gaussian template once.
     This gives a "matched-filter response map" per frame — the value at
     pixel (y, x) is the score if the target were at (y, x) on that frame.
  2. For each candidate velocity (v_y, v_x) on the grid, shift each frame's
     response map by (-v_y·t, -v_x·t) and sum across t. The peak of the
     summed map is the GLRT score for that velocity.
  3. Take the maximum over all velocities.

Equivalent to the existing ``src/tbd/accumulator.py`` BUT on RAW frames,
not on derived evidence channels (raw matched filter is NP-optimal under
AWGN; the evidence-channel version trades optimality for clutter-robustness).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from npceiling.theory.matched_filter import render_unit_template


def _fft_correlate_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Cross-correlate ``image`` with ``kernel``; return same-size output.

    Pure numpy. For a symmetric Gaussian kernel, correlation == convolution,
    so we use FFT-based convolution and align centers.
    """
    H, W = image.shape
    Kh, Kw = kernel.shape
    fft_h = H + Kh - 1
    fft_w = W + Kw - 1
    img_fft = np.fft.rfft2(image, s=(fft_h, fft_w))
    ker_fft = np.fft.rfft2(kernel, s=(fft_h, fft_w))
    full = np.fft.irfft2(img_fft * ker_fft, s=(fft_h, fft_w))
    # Extract 'same'-mode region.
    start_h = (Kh - 1) // 2
    start_w = (Kw - 1) // 2
    return full[start_h:start_h + H, start_w:start_w + W]


@dataclass
class GLRTConfig:
    """Velocity-grid hyperparameters for the GLRT search.

    Defaults match the analytical motion range of the synthetic data
    (speed_min=0.3, speed_max=1.5 px/frame from SequenceConfig). We extend
    speed_max to 2.0 to be safe for Brownian-noise excursions.

    Total hypotheses = 1 + n_speeds * n_directions  (one v=0 + the rest).
    Default: 1 + 9 × 16 = 145 hypotheses — nearly 2× the accumulator's 73.
    """
    n_speeds: int = 9
    n_directions: int = 16
    speed_max_px_per_frame: float = 2.0
    sigma_psf: float = 1.0


def _build_velocity_grid(cfg: GLRTConfig) -> np.ndarray:
    """Return an (N, 2) array of (v_y, v_x) hypotheses on a polar grid.

    Always includes v=0 (stationary hypothesis).
    """
    speeds = np.linspace(0.0, cfg.speed_max_px_per_frame, cfg.n_speeds + 1)[1:]  # skip 0
    angles = np.linspace(0.0, 2.0 * np.pi, cfg.n_directions, endpoint=False)
    vs = [(0.0, 0.0)]  # stationary
    for s in speeds:
        for a in angles:
            vs.append((s * np.sin(a), s * np.cos(a)))  # (vy, vx)
    return np.asarray(vs, dtype=np.float64)


def _kernel_for_sigma(sigma_psf: float) -> np.ndarray:
    """Compact PSF-shaped kernel (peak=1.0) centered on its own grid.

    Mirrors the renderer in src/synthetic/targets.py for exact-shape match.
    """
    radius = int(np.ceil(3.0 * sigma_psf)) + 1
    side = 2 * radius + 1
    center = radius
    return render_unit_template((side, side), center, center, sigma_psf)


def _shift_accumulate(
    responses: np.ndarray,  # (T, H, W) per-frame matched-filter response
    v_y: float,
    v_x: float,
) -> np.ndarray:
    """Shift each frame by (-v_y·t, -v_x·t) and sum along t — zero-padded.

    np.roll wraps around the canvas, which would create spurious correlations
    at the edges. We use boundary-correct slicing via np.pad → np.roll →
    crop, but a simpler safe alternative is to slice-based shift:

      shifted[y, x] = responses[t, y + v_y·t, x + v_x·t]  if in bounds, else 0

    For the trajectories in our canvas (target stays within bounds across all
    T frames by construction), the zero-padded version gives identical results
    to np.roll for valid hypotheses, but penalizes hypotheses that "would have
    walked off the canvas" — which we want.
    """
    T, H, W = responses.shape
    acc = np.zeros((H, W), dtype=responses.dtype)
    for t in range(T):
        dy = int(round(-v_y * t))
        dx = int(round(-v_x * t))
        # Source indices in responses[t]: y - dy, x - dx (so destination
        # y + (target_y - source_y) where target_y = source_y - v_y·t).
        # Use slicing to avoid wraparound.
        src_y_start = max(0, -dy)
        src_y_end = min(H, H - dy)
        src_x_start = max(0, -dx)
        src_x_end = min(W, W - dx)
        dst_y_start = max(0, dy)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = max(0, dx)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        if src_y_end > src_y_start and src_x_end > src_x_start:
            acc[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += \
                responses[t, src_y_start:src_y_end, src_x_start:src_x_end]
    return acc


def glrt_score(
    frames: np.ndarray,
    cfg: GLRTConfig | None = None,
) -> float:
    """GLRT test statistic — max over velocity grid of the matched-filter
    sum along the corresponding straight-line trajectory.

    Args:
        frames: (T, H, W) observed frames.
        cfg:    GLRT configuration (velocity grid).

    Returns:
        Scalar GLRT score. Higher → stronger evidence for a target on
        *some* trajectory in the search space.
    """
    if frames.ndim != 3:
        raise ValueError(f"frames must be (T, H, W); got {frames.shape}")
    cfg = cfg or GLRTConfig()

    T, H, W = frames.shape
    kernel = _kernel_for_sigma(cfg.sigma_psf)

    # 1) FFT-correlate each frame with the template.
    #    For a symmetric Gaussian kernel, correlation == convolution.
    #    Output[y, x] = matched-filter score for hypothesis "target at (y, x)".
    responses = np.empty((T, H, W), dtype=np.float64)
    for t in range(T):
        responses[t] = _fft_correlate_same(frames[t], kernel)

    # 2) Search velocity grid.
    velocities = _build_velocity_grid(cfg)
    best = -np.inf
    for (v_y, v_x) in velocities:
        acc = _shift_accumulate(responses, float(v_y), float(v_x))
        peak = float(acc.max())
        if peak > best:
            best = peak
    return best


__all__ = ["GLRTConfig", "glrt_score"]
