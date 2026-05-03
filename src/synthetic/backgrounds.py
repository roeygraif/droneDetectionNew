"""AWGN and 1/f-shaped Gaussian clutter (FFT-based, no Perlin dependency).

Clutter is parameterized by a single spectral exponent ``alpha`` controlling
how steeply the 2D power spectrum falls off:

  alpha = 0  -> white noise (no spatial structure)
  alpha = 1  -> "pink" noise (1/f, fractal texture)
  alpha = 2  -> "brown"/cloud-like smooth structure (default)

For a sequence with temporally-correlated clutter (e.g., drifting clouds), we
generate one field per sequence and shift it sub-pixel-precisely each frame
using the Fourier shift theorem. Temporally-correlated clutter is what makes
the deep-temporal vs. classical separation in the thesis non-trivial —
uncorrelated clutter would be averaged out trivially.
"""

from __future__ import annotations

import numpy as np


def awgn(
    shape: tuple[int, ...],
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pure additive white Gaussian noise of shape ``shape``."""
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    return rng.normal(0.0, sigma, size=shape).astype(np.float32)


def clutter_field(
    shape: tuple[int, int],
    spectral_exponent: float,
    rms: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a 2D Gaussian random field with 1/f^alpha power spectrum, scaled to ``rms``."""
    H, W = shape
    white = rng.normal(0.0, 1.0, size=(H, W))

    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    f_radial = np.sqrt(fy * fy + fx * fx)
    f_radial[0, 0] = 1.0  # avoid divide-by-zero; DC zeroed below

    spectrum_filter = 1.0 / (f_radial**spectral_exponent)
    spectrum_filter[0, 0] = 0.0  # zero DC component

    F = np.fft.fft2(white) * spectrum_filter
    field = np.fft.ifft2(F).real

    current_rms = float(np.sqrt(np.mean(field * field)))
    if current_rms > 0.0:
        field = field * (rms / current_rms)
    return field.astype(np.float32)


def _fourier_shift(field: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Sub-pixel translate ``field`` by ``(dy, dx)`` using the Fourier shift theorem (periodic)."""
    H, W = field.shape
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    F = np.fft.fft2(field)
    F = F * np.exp(-2j * np.pi * (dy * fy + dx * fx))
    return np.fft.ifft2(F).real.astype(field.dtype)


def clutter_sequence(
    shape: tuple[int, int],
    n_frames: int,
    spectral_exponent: float,
    rms: float,
    drift_px_per_frame: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a temporally-coherent clutter sequence.

    A single 2D clutter field is drawn, then translated by a fixed sub-pixel
    drift each frame. Returns shape ``(n_frames, H, W)``.
    """
    base = clutter_field(shape, spectral_exponent, rms, rng)
    if drift_px_per_frame == 0.0:
        return np.broadcast_to(base, (n_frames, *shape)).copy()

    angle = rng.uniform(0.0, 2.0 * np.pi)
    dy = drift_px_per_frame * np.sin(angle)
    dx = drift_px_per_frame * np.cos(angle)

    out = np.zeros((n_frames, *shape), dtype=np.float32)
    for t in range(n_frames):
        out[t] = _fourier_shift(base, dy * t, dx * t)
    return out
