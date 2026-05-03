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
