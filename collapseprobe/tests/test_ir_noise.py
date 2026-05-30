"""Validation tests for the IR 3-D noise model.

These check the *physics* we rely on:
  - fixed-pattern noise is identical on every frame (does NOT integrate away),
  - random noise decorrelates across frames (integrates as 1/sqrt(T)),
  - each component carries the configured variance,
  - the headline property: temporal averaging kills the random part but leaves
    the fixed pattern as a hard floor,
  - scintillation has unit mean and the prescribed index,
  - everything is deterministic given the rng seed.
"""
from __future__ import annotations

import numpy as np
import pytest

from collapseprobe.ir_noise import (
    IR3DNoiseConfig,
    ScintillationConfig,
    apply_scintillation,
    make_fixed_pattern,
    sample_random_noise,
    total_noise,
    total_noise_variance,
)


def test_fixed_pattern_is_identical_every_frame():
    cfg = IR3DNoiseConfig(sigma_tvh=0.0, sigma_t=0.0, sigma_vh=0.5, sigma_v=0.2, sigma_h=0.1)
    rng = np.random.default_rng(0)
    fixed = make_fixed_pattern((64, 64), cfg, rng)
    noise = total_noise((16, 64, 64), fixed, cfg, rng)  # random part is zero here
    # Every frame must equal frame 0 exactly (it is a fixed pattern).
    for t in range(noise.shape[0]):
        assert np.allclose(noise[t], noise[0], atol=1e-6)


def test_random_part_decorrelates_across_frames():
    cfg = IR3DNoiseConfig(sigma_tvh=1.0, sigma_t=0.0, sigma_vh=0.0, sigma_v=0.0, sigma_h=0.0)
    rng = np.random.default_rng(1)
    n = sample_random_noise((40, 64, 64), cfg, rng)
    # Lag-1 temporal correlation should be ~0 for white random noise.
    a = n[:-1].ravel()
    b = n[1:].ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    assert abs(corr) < 0.05


def test_component_variances_match_config():
    # Pixel FPN only.
    cfg = IR3DNoiseConfig(sigma_tvh=0.0, sigma_t=0.0, sigma_vh=0.7, sigma_v=0.0, sigma_h=0.0)
    rng = np.random.default_rng(2)
    fixed = make_fixed_pattern((256, 256), cfg, rng)
    assert np.std(fixed) == pytest.approx(0.7, rel=0.1)

    # Random floor only.
    cfg2 = IR3DNoiseConfig(sigma_tvh=1.3, sigma_t=0.0, sigma_vh=0.0, sigma_v=0.0, sigma_h=0.0)
    r = sample_random_noise((32, 128, 128), cfg2, np.random.default_rng(3))
    assert np.std(r) == pytest.approx(1.3, rel=0.1)


def test_temporal_averaging_floor_is_fixed_pattern():
    """The headline thesis property: averaging T frames removes the random part
    (1/sqrt(T)) but leaves the fixed pattern untouched as a hard floor."""
    cfg = IR3DNoiseConfig(sigma_tvh=1.0, sigma_t=0.0, sigma_vh=0.3, sigma_v=0.0, sigma_h=0.0)
    rng = np.random.default_rng(4)
    fixed = make_fixed_pattern((96, 96), cfg, rng)

    residual_std = []
    Ts = [1, 4, 16, 64]
    for T in Ts:
        n = total_noise((T, 96, 96), fixed, cfg, np.random.default_rng(100 + T))
        time_avg = n.mean(axis=0)            # average over frames
        # Subtract the (known) fixed pattern -> what's left is the random residual.
        residual = time_avg - fixed
        residual_std.append(float(np.std(residual)))

    # Random residual must shrink ~1/sqrt(T).
    assert residual_std[0] > residual_std[-1] * 3
    assert residual_std[-1] == pytest.approx(1.0 / np.sqrt(64), rel=0.25)

    # The fixed pattern itself never shrinks: the time-average is dominated by it
    # once T is large.
    n_large = total_noise((256, 96, 96), fixed, cfg, np.random.default_rng(7))
    assert np.std(n_large.mean(axis=0)) == pytest.approx(np.std(fixed), rel=0.2)


def test_total_noise_variance_matches_empirical():
    cfg = IR3DNoiseConfig(sigma_tvh=1.0, sigma_t=0.1, sigma_vh=0.3, sigma_v=0.12, sigma_h=0.08)
    rng = np.random.default_rng(5)
    fixed = make_fixed_pattern((128, 128), cfg, rng)
    n = total_noise((64, 128, 128), fixed, cfg, rng)
    assert np.var(n) == pytest.approx(total_noise_variance(cfg), rel=0.1)


def test_temporal_ar1_introduces_correlation():
    cfg = IR3DNoiseConfig(sigma_tvh=1.0, sigma_t=0.0, sigma_vh=0.0, temporal_1f_rho=0.7)
    n = sample_random_noise((64, 48, 48), cfg, np.random.default_rng(6))
    a, b = n[:-1].ravel(), n[1:].ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    assert corr == pytest.approx(0.7, abs=0.1)


def test_scintillation_unit_mean_and_index():
    cfg = ScintillationConfig(enabled=True, scintillation_index=0.2, temporal_rho=0.5)
    rng = np.random.default_rng(8)
    target = np.ones((20000, 1, 1), dtype=np.float32)  # unit target -> gain distribution
    out = apply_scintillation(target, cfg, rng).ravel()
    assert out.mean() == pytest.approx(1.0, abs=0.02)
    assert out.std() == pytest.approx(0.2, rel=0.15)


def test_scintillation_disabled_is_identity():
    cfg = ScintillationConfig(enabled=False)
    target = np.random.default_rng(9).normal(size=(8, 4, 4)).astype(np.float32)
    assert np.allclose(apply_scintillation(target, cfg, np.random.default_rng(0)), target)


def test_determinism_by_seed():
    cfg = IR3DNoiseConfig()
    a = total_noise((8, 32, 32), make_fixed_pattern((32, 32), cfg, np.random.default_rng(11)), cfg, np.random.default_rng(12))
    b = total_noise((8, 32, 32), make_fixed_pattern((32, 32), cfg, np.random.default_rng(11)), cfg, np.random.default_rng(12))
    assert np.array_equal(a, b)
