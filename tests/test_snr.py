import math

import numpy as np
import pytest

from synthetic.snr import amplitude_for_snr, measure_peak_snr_db


def test_amplitude_at_zero_db():
    assert amplitude_for_snr(0.0, 1.0) == pytest.approx(1.0)
    assert amplitude_for_snr(0.0, 2.5) == pytest.approx(2.5)


def test_amplitude_db_doubling():
    a0 = amplitude_for_snr(0.0, 1.0)
    a6 = amplitude_for_snr(6.0, 1.0)
    a12 = amplitude_for_snr(12.0, 1.0)
    assert a6 / a0 == pytest.approx(2.0, rel=0.005)
    assert a12 / a0 == pytest.approx(4.0, rel=0.005)


@pytest.mark.parametrize("snr_db", [-25.0, -15.0, -5.0, 0.0, 5.0])
def test_round_trip_snr(snr_db):
    rng = np.random.default_rng(123)
    sigma = 1.0
    A = amplitude_for_snr(snr_db, sigma)

    H, W = 256, 256
    target_field = np.zeros((H, W), dtype=np.float32)
    target_field[H // 2, W // 2] = A
    noise_field = rng.normal(0.0, sigma, (H, W)).astype(np.float32)

    measured = measure_peak_snr_db(target_field, noise_field)
    assert measured == pytest.approx(snr_db, abs=0.3)


def test_invalid_sigma():
    with pytest.raises(ValueError):
        amplitude_for_snr(0.0, 0.0)
    with pytest.raises(ValueError):
        amplitude_for_snr(0.0, -1.0)


def test_zero_target_returns_neg_inf():
    target = np.zeros((10, 10), dtype=np.float32)
    noise = np.random.default_rng(0).normal(0, 1, (10, 10)).astype(np.float32)
    assert measure_peak_snr_db(target, noise) == -math.inf


def test_zero_noise_returns_inf():
    target = np.zeros((10, 10), dtype=np.float32)
    target[5, 5] = 1.0
    noise = np.zeros((10, 10), dtype=np.float32)
    assert measure_peak_snr_db(target, noise) == math.inf
