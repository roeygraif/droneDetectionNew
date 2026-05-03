"""2D Gaussian PSF rendering with sub-pixel support.

The rendered peak of a Gaussian centered between pixels is strictly less than
its analytical peak (sub-pixel sampling rolloff). For tight SNR control we
*normalize* every render so that the rendered peak equals the requested
``peak`` exactly. This decouples the SNR sweep from sub-pixel rendering
artifacts — which would otherwise add ~2 dB of jitter to every measurement
and confound the waterfall curves.
"""

from __future__ import annotations

import numpy as np


def render_gaussian(
    canvas: np.ndarray,
    cy: float,
    cx: float,
    sigma: float,
    peak: float,
) -> None:
    """Add a 2D isotropic Gaussian to ``canvas`` in-place at sub-pixel center ``(cy, cx)``.

    The contribution's peak pixel value equals ``peak`` exactly (we rescale to
    compensate for the sub-pixel sampling rolloff).
    """
    if canvas.ndim != 2:
        raise ValueError(f"canvas must be 2D, got shape {canvas.shape}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    H, W = canvas.shape
    radius = int(np.ceil(3.0 * sigma)) + 1

    cy_i = int(np.round(cy))
    cx_i = int(np.round(cx))
    y_min = max(0, cy_i - radius)
    y_max = min(H, cy_i + radius + 1)
    x_min = max(0, cx_i - radius)
    x_max = min(W, cx_i + radius + 1)

    if y_min >= y_max or x_min >= x_max:
        return

    yy, xx = np.mgrid[y_min:y_max, x_min:x_max].astype(np.float64)
    gauss = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
    gauss_peak = float(gauss.max())
    if gauss_peak <= 0.0:
        return

    canvas[y_min:y_max, x_min:x_max] += (peak / gauss_peak) * gauss.astype(canvas.dtype)
