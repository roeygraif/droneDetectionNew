"""Sanity tests for the matched-filter / GLRT ceiling.

Validates the implementation against analytical predictions and basic
ordering invariants. If any of these fail, the ceiling is buggy and any
results computed from it are unreliable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from synthetic.sequence import SequenceConfig, generate_sequence  # noqa: E402

from npceiling.theory.glrt import GLRTConfig, glrt_score  # noqa: E402
from npceiling.theory.matched_filter import (  # noqa: E402
    cmf_score,
    render_unit_template,
    template_norm_squared,
)


# ---------------------------------------------------------------------------
# 0. Template-shape correctness
# ---------------------------------------------------------------------------

def test_template_peak_is_unity():
    """The rendered template must have peak = 1.0 exactly, including sub-pixel."""
    for cy, cx in [(32.0, 32.0), (32.3, 32.7), (32.5, 32.5), (32.999, 32.001)]:
        t = render_unit_template((64, 64), cy, cx, sigma=1.0)
        assert abs(t.max() - 1.0) < 1e-9, f"peak != 1.0 at ({cy}, {cx})"


def test_template_norm_squared_matches_theory():
    """For a unit-peak Gaussian, ||template||² ≈ π σ² (the integral of e^{-r²/σ²})."""
    for sigma in [0.7, 1.0, 1.5]:
        norm_sq = template_norm_squared((64, 64), 32.0, 32.0, sigma)
        expected = np.pi * sigma ** 2
        assert abs(norm_sq - expected) / expected < 0.05, (
            f"||template||² = {norm_sq:.4f}, expected ~{expected:.4f} for sigma={sigma}"
        )


# ---------------------------------------------------------------------------
# 1. CMF on pure noise has the right distribution
# ---------------------------------------------------------------------------

def test_cmf_noise_distribution():
    """CMF on pure Gaussian noise: mean ≈ 0, std ≈ σ_noise · sqrt(||template||²)."""
    H, W, T = 64, 64, 32
    sigma_noise = 1.0
    sigma_psf = 1.0
    rng = np.random.default_rng(42)
    positions = np.tile([32.0, 32.0], (T, 1))
    visible = np.ones(T, dtype=bool)

    scores = []
    for _ in range(200):
        frames = rng.standard_normal((T, H, W)) * sigma_noise
        scores.append(cmf_score(frames, positions, visible, sigma_psf))
    scores = np.asarray(scores)

    norm_sq = template_norm_squared((H, W), 32.0, 32.0, sigma_psf)
    expected_std = sigma_noise * np.sqrt(T * norm_sq)
    assert abs(scores.mean()) < 0.2 * expected_std, (
        f"CMF noise mean {scores.mean():.3f} >> 0 (expected_std={expected_std:.3f})"
    )
    # Std should be within ~10% with 200 Monte-Carlo trials.
    assert abs(scores.std() - expected_std) / expected_std < 0.15, (
        f"CMF noise std {scores.std():.3f} != expected {expected_std:.3f}"
    )


# ---------------------------------------------------------------------------
# 2. Pd-vs-SNR: at high SNR Pd ≈ 1, at very low SNR Pd ≈ FAR
# ---------------------------------------------------------------------------

def _generate_pair(snr_db: float, seed: int, n_frames: int = 32):
    """One positive and one empty sequence at the same SNR (AWGN-only)."""
    cfg_pos = SequenceConfig(
        canvas_shape=(64, 64), n_frames=n_frames,
        target_sigma=1.0, noise_sigma=1.0,
        motion="cv", speed_min_px_per_frame=0.3,
        speed_max_px_per_frame=1.5, boundary_margin_px=4.0,
        snr_db=snr_db, sequence_type="positive_uav", seed=seed,
        clutter_rms=0.0,
    )
    cfg_neg = SequenceConfig(
        canvas_shape=(64, 64), n_frames=n_frames,
        target_sigma=1.0, noise_sigma=1.0,
        motion="cv", boundary_margin_px=4.0,
        snr_db=snr_db, sequence_type="empty_background", seed=seed + 10_000,
        clutter_rms=0.0,
    )
    return generate_sequence(cfg_pos), generate_sequence(cfg_neg)


def test_cmf_pd_high_snr_near_one():
    """At SNR = +10 dB with T=32, the CMF detector at FAR=1e-2 should have Pd ≈ 1.0."""
    pos_scores, neg_scores = [], []
    for i in range(30):
        s_pos, s_neg = _generate_pair(snr_db=10.0, seed=1000 + i)
        pos_scores.append(cmf_score(
            s_pos.frames, s_pos.positions, s_pos.target_visible, sigma_psf=1.0
        ))
        neg_scores.append(cmf_score(
            s_neg.frames, s_pos.positions, s_pos.target_visible, sigma_psf=1.0
        ))
    pos = np.asarray(pos_scores)
    neg = np.asarray(neg_scores)
    # At FAR=1e-2 with 30 negatives, threshold = the max negative score.
    threshold = neg.max()
    pd = float((pos >= threshold).mean())
    assert pd > 0.95, f"Pd at +10 dB = {pd:.2f}, expected > 0.95"


def test_cmf_pd_very_low_snr_near_far():
    """At SNR = -30 dB the signal is buried; Pd at FAR=1e-2 should be near FAR (~0.01-0.05)."""
    pos_scores, neg_scores = [], []
    for i in range(30):
        s_pos, s_neg = _generate_pair(snr_db=-30.0, seed=2000 + i)
        pos_scores.append(cmf_score(
            s_pos.frames, s_pos.positions, s_pos.target_visible, sigma_psf=1.0
        ))
        neg_scores.append(cmf_score(
            s_neg.frames, s_pos.positions, s_pos.target_visible, sigma_psf=1.0
        ))
    pos = np.asarray(pos_scores)
    neg = np.asarray(neg_scores)
    threshold = neg.max()
    pd = float((pos >= threshold).mean())
    # Loose tolerance: with only 30 trials Pd can range up to ~0.2 by chance.
    assert pd < 0.30, f"Pd at -30 dB = {pd:.2f}, expected ≈ FAR (loose: < 0.30)"


# ---------------------------------------------------------------------------
# 3. CMF ≥ GLRT — clairvoyant always wins (or ties)
# ---------------------------------------------------------------------------

def test_cmf_at_least_as_good_as_glrt():
    """At every SNR, CMF Pd should be ≥ GLRT Pd. Clairvoyant cannot lose."""
    cfg_glrt = GLRTConfig()
    for snr_db in [-15.0, -6.0, 0.0]:
        pos_cmf, neg_cmf, pos_glrt, neg_glrt = [], [], [], []
        for i in range(25):
            s_pos, s_neg = _generate_pair(snr_db=snr_db, seed=3000 + i)
            pos_cmf.append(cmf_score(s_pos.frames, s_pos.positions,
                                      s_pos.target_visible, sigma_psf=1.0))
            neg_cmf.append(cmf_score(s_neg.frames, s_pos.positions,
                                      s_pos.target_visible, sigma_psf=1.0))
            pos_glrt.append(glrt_score(s_pos.frames, cfg_glrt))
            neg_glrt.append(glrt_score(s_neg.frames, cfg_glrt))

        cmf_thresh = float(np.max(neg_cmf))
        glrt_thresh = float(np.max(neg_glrt))
        cmf_pd = float((np.asarray(pos_cmf) >= cmf_thresh).mean())
        glrt_pd = float((np.asarray(pos_glrt) >= glrt_thresh).mean())
        # Allow small MC slack — equality up to ±0.10 with 25 trials.
        assert cmf_pd + 0.10 >= glrt_pd, (
            f"At SNR={snr_db} dB: CMF Pd={cmf_pd:.2f} < GLRT Pd={glrt_pd:.2f} "
            "(clairvoyant should never be worse)"
        )


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------

def test_determinism():
    """Same seed → identical scores."""
    s1, _ = _generate_pair(snr_db=-6.0, seed=99)
    s2, _ = _generate_pair(snr_db=-6.0, seed=99)
    np.testing.assert_array_equal(s1.frames, s2.frames)
    cmf_a = cmf_score(s1.frames, s1.positions, s1.target_visible, sigma_psf=1.0)
    cmf_b = cmf_score(s2.frames, s2.positions, s2.target_visible, sigma_psf=1.0)
    assert abs(cmf_a - cmf_b) < 1e-9, f"CMF non-deterministic: {cmf_a} vs {cmf_b}"
    glrt_a = glrt_score(s1.frames)
    glrt_b = glrt_score(s2.frames)
    assert abs(glrt_a - glrt_b) < 1e-9, f"GLRT non-deterministic: {glrt_a} vs {glrt_b}"
