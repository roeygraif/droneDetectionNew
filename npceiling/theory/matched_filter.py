"""Clairvoyant Matched Filter (CMF).

The matched filter is the Neyman-Pearson optimal detector for a known signal
in additive white Gaussian noise (Marcum 1948, Kay 1998 vol. 2). For our
synthetic generator the "known signal" includes the trajectory — the
matched filter therefore needs ground-truth positions, which we get from
``SequenceSample.positions`` and ``SequenceSample.target_visible``.

Since the ground truth is used, this is strictly the **upper bound** — no
practical detector can beat it. We call it Clairvoyant Matched Filter (CMF).

The template must match the rendered PSF shape **exactly**, including the
sub-pixel normalization compensation done in ``src/synthetic/targets.py``
(line 52: ``canvas += (peak / gauss_peak) * gauss``). The math:

  rendered_signal(y, x; cy, cx, σ_PSF, A_peak) =
      A_peak · exp(-r²/(2σ_PSF²)) / max_yx exp(-r²/(2σ_PSF²))

where the max is over the *sampled* grid (sub-pixel rolloff compensation).

For the matched filter test statistic, the optimal template is proportional
to the signal shape. We use the unit-peak Gaussian (peak=1.0) and normalize
the same way the renderer does so the inner product matches expectation.

Test statistic per sequence:

  T_CMF = Σ_{t : target_visible[t]} Σ_{y,x} frame[t,y,x] · template_t(y,x)

Under H₀ (target absent):
  T_CMF ~ N(0, σ_noise² · ||template||²)
Under H₁ (target present):
  T_CMF ~ N(A_peak · ||template||², σ_noise² · ||template||²)

so the test-statistic SNR (mean separation / std) is
  SNR_test = (A_peak / σ_noise) · sqrt(||template||²)
            = 10^(SNR_dB/20) · sqrt(||template||²).
"""
from __future__ import annotations

import numpy as np


def render_unit_template(
    canvas_shape: tuple[int, int],
    cy: float,
    cx: float,
    sigma: float,
) -> np.ndarray:
    """Render a unit-peak Gaussian template matching the project's PSF.

    Mirrors ``src/synthetic/targets.render_gaussian`` with peak=1.0 — including
    the sub-pixel sampling-rolloff normalization so the rendered peak equals
    exactly 1.0 (as the data generator does for amplitude). The template
    is zero outside the 3-sigma support window.

    Args:
        canvas_shape: (H, W) — must match the frame size.
        cy, cx:       sub-pixel target center (float pixels).
        sigma:        PSF width (same as ``SequenceConfig.target_sigma``).

    Returns:
        (H, W) float64 array. Most pixels are zero; non-zero only in the
        3σ+1 box around (cy, cx). Peak value is exactly 1.0.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    H, W = canvas_shape
    template = np.zeros((H, W), dtype=np.float64)
    radius = int(np.ceil(3.0 * sigma)) + 1
    cy_i = int(np.round(cy))
    cx_i = int(np.round(cx))
    y_min = max(0, cy_i - radius)
    y_max = min(H, cy_i + radius + 1)
    x_min = max(0, cx_i - radius)
    x_max = min(W, cx_i + radius + 1)
    if y_min >= y_max or x_min >= x_max:
        return template
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max].astype(np.float64)
    gauss = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
    gauss_peak = float(gauss.max())
    if gauss_peak <= 0.0:
        return template
    template[y_min:y_max, x_min:x_max] = gauss / gauss_peak  # peak = 1.0 exactly
    return template


def cmf_score(
    frames: np.ndarray,
    positions: np.ndarray,
    target_visible: np.ndarray,
    sigma_psf: float,
) -> float:
    """Clairvoyant matched-filter test statistic.

    Sums template·frame inner products over all frames where the target is
    visible (Bernoulli visibility dropout from the data generator).

    Args:
        frames:         (T, H, W) observed frames (signal + noise).
        positions:      (T, 2) ground-truth target centers (y, x). NaN entries
                        are treated as "not visible" regardless of
                        ``target_visible``.
        target_visible: (T,) bool array — per-frame visibility flag.
        sigma_psf:      PSF width.

    Returns:
        Scalar matched-filter score. Higher → more evidence for target.
    """
    if frames.ndim != 3:
        raise ValueError(f"frames must be (T, H, W); got {frames.shape}")
    if positions.shape != (frames.shape[0], 2):
        raise ValueError(
            f"positions must be (T, 2); got {positions.shape} for T={frames.shape[0]}"
        )
    if target_visible.shape != (frames.shape[0],):
        raise ValueError(
            f"target_visible must be (T,); got {target_visible.shape}"
        )

    T, H, W = frames.shape
    score = 0.0
    for t in range(T):
        if not bool(target_visible[t]):
            continue
        cy, cx = float(positions[t, 0]), float(positions[t, 1])
        if not (np.isfinite(cy) and np.isfinite(cx)):
            continue
        template = render_unit_template((H, W), cy, cx, sigma_psf)
        score += float(np.sum(frames[t] * template))
    return score


def cmf_score_with_fake_trajectory(
    frames: np.ndarray,
    fake_positions: np.ndarray,
    fake_visible: np.ndarray,
    sigma_psf: float,
) -> float:
    """CMF score using a *given* trajectory — for the noise-only paired baseline.

    Identical math to ``cmf_score``, but takes the trajectory as input directly
    (the caller borrows a positive sequence's trajectory and applies it to a
    negative sequence's frames to get the noise-only test statistic
    distribution).
    """
    return cmf_score(frames, fake_positions, fake_visible, sigma_psf)


def template_norm_squared(
    canvas_shape: tuple[int, int],
    cy: float,
    cx: float,
    sigma: float,
) -> float:
    """||template||² for the unit-peak Gaussian — used by analytical Pd checks."""
    template = render_unit_template(canvas_shape, cy, cx, sigma)
    return float(np.sum(template * template))


__all__ = [
    "render_unit_template",
    "cmf_score",
    "cmf_score_with_fake_trajectory",
    "template_norm_squared",
]
