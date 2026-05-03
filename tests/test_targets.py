import numpy as np
import pytest

from synthetic.targets import render_gaussian


def test_pixel_centered_peak_exact():
    canvas = np.zeros((50, 50), dtype=np.float32)
    render_gaussian(canvas, 25, 25, sigma=1.0, peak=3.0)
    assert canvas.max() == pytest.approx(3.0, abs=1e-6)


def test_subpixel_centered_peak_normalized():
    """Peak normalization should hold even when the Gaussian is between pixels."""
    canvas = np.zeros((50, 50), dtype=np.float32)
    render_gaussian(canvas, 25.5, 25.5, sigma=1.0, peak=3.0)
    assert canvas.max() == pytest.approx(3.0, abs=1e-5)


def test_gaussian_integral_on_pixel():
    """Integral of a Gaussian rendered on a pixel ~= 2*pi*sigma^2 * peak (peak == analytical peak when on-pixel)."""
    canvas = np.zeros((80, 80), dtype=np.float32)
    render_gaussian(canvas, 40, 40, sigma=2.0, peak=1.0)
    expected = 2.0 * np.pi * 2.0 * 2.0
    assert canvas.sum() == pytest.approx(expected, rel=0.02)


def test_off_canvas_no_op():
    canvas = np.zeros((50, 50), dtype=np.float32)
    render_gaussian(canvas, -100, -100, sigma=1.0, peak=1.0)
    assert canvas.sum() == 0.0


def test_canvas_unchanged_when_peak_zero():
    canvas = np.zeros((50, 50), dtype=np.float32)
    render_gaussian(canvas, 25, 25, sigma=1.0, peak=0.0)
    assert canvas.sum() == 0.0


def test_invalid_sigma():
    canvas = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        render_gaussian(canvas, 5, 5, sigma=0.0, peak=1.0)
    with pytest.raises(ValueError):
        render_gaussian(canvas, 5, 5, sigma=-1.0, peak=1.0)


def test_canvas_must_be_2d():
    canvas = np.zeros((10, 10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        render_gaussian(canvas, 5, 5, sigma=1.0, peak=1.0)
