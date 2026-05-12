"""Single source of truth for SNR math.

Convention used throughout the project (matches the thesis brief):

    SNR_dB = 10 * log10( A_peak^2 / sigma_noise^2 )
           = 20 * log10( A_peak / sigma_noise )

where A_peak is the peak intensity of the rendered target signal and
sigma_noise is the std-dev of the AWGN field. Clutter (when present) is
treated separately and reported as a clutter-to-noise ratio elsewhere.
"""

from __future__ import annotations

import math

import numpy as np


def amplitude_for_snr(snr_db: float, sigma_noise: float) -> float:
    """Return the peak target amplitude that yields ``snr_db`` against AWGN of std ``sigma_noise``.

    Inverse of :func:`measure_peak_snr_db`.
    """
    if sigma_noise <= 0.0:
        raise ValueError(f"sigma_noise must be > 0, got {sigma_noise}")
    return float(sigma_noise * 10.0 ** (snr_db / 20.0))


def measure_peak_snr_db(target_field: np.ndarray, noise_field: np.ndarray) -> float:
    """Measure peak-amplitude SNR in dB from rendered fields.

    ``target_field`` is the noise-free signal (what was added on top of the noise).
    ``noise_field`` is a sample of the AWGN background only.
    """
    peak = float(np.max(target_field))
    sigma = float(np.std(noise_field))
    if peak <= 0.0:
        return -math.inf
    if sigma <= 0.0:
        return math.inf
    return 20.0 * math.log10(peak / sigma)


def effective_scnr_db(
    target_peak: float,
    bg_frame: np.ndarray,
    cy: float,
    cx: float,
    target_sigma: float,
) -> tuple[float, float]:
    """Local SCNR around ``(cy, cx)`` in one frame.

    Measures ``std`` of the background components (AWGN + clutter + distractors
    — i.e. everything except the target PSF) inside a square neighborhood
    around the target center, with an inner disc of radius ~3 sigma excluded
    so the rendered PSF doesn't bias the background estimate. Returns
    ``(scnr_db, local_bg_std)``; ``scnr_db = 20 log10(target_peak / local_bg_std)``.

    If the neighborhood is too small (target near canvas edge), returns ``NaN``
    for both. With zero background, returns ``+inf`` for the SCNR.
    """
    H, W = bg_frame.shape
    inner = max(1, int(np.ceil(3.0 * target_sigma)))
    outer = max(int(np.ceil(5.0 * target_sigma)), inner + 2, 8)
    cy_i = int(round(cy))
    cx_i = int(round(cx))
    y0, y1 = max(0, cy_i - outer), min(H, cy_i + outer + 1)
    x0, x1 = max(0, cx_i - outer), min(W, cx_i + outer + 1)
    if y1 - y0 < 3 or x1 - x0 < 3:
        return math.nan, math.nan
    patch = bg_frame[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    annulus = ((yy - cy) ** 2 + (xx - cx) ** 2) >= (inner * inner)
    pixels = patch[annulus]
    if pixels.size < 4:
        pixels = patch.ravel()
    s = float(np.std(pixels))
    if s <= 0.0:
        return (math.inf if target_peak > 0 else math.nan), s
    if target_peak <= 0.0:
        return -math.inf, s
    return 20.0 * math.log10(target_peak / s), s
