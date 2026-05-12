"""Tests for the false-alarm-aware extensions to generate_sequence.

Covers:
  - target visibility dropout (no PSF leakage on dropped frames),
  - empty_background / hard_negative / mixed_uav_and_distractors sequence types,
  - determinism across sequence types and seeds,
  - SNR / effective-SCNR semantics for target-absent and target-dropped frames.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from synthetic.distractors import DISTRACTOR_TYPES
from synthetic.sequence import SequenceConfig, generate_sequence


def _base(**kw) -> SequenceConfig:
    return SequenceConfig(
        n_frames=kw.pop("n_frames", 10),
        canvas_shape=kw.pop("canvas_shape", (96, 96)),
        snr_db=kw.pop("snr_db", 0.0),
        noise_sigma=kw.pop("noise_sigma", 1.0),
        seed=kw.pop("seed", 42),
        **kw,
    )


# ---------------------------------------------------------------------------
# A. Target dropout
# ---------------------------------------------------------------------------

def test_target_dropout_zeroes_target_signal_completely():
    cfg = _base(
        n_frames=20, target_dropout_enabled=True, target_dropout_prob=0.5,
        target_min_visible_frames=1, snr_db=10.0, seed=1,
    )
    s = generate_sequence(cfg)
    # At least one frame must be visible (min_visible_frames guarantee).
    assert int(s.target_visible.sum()) >= 1
    # And at least one frame must be dropped for this test to be meaningful.
    assert int((~s.target_visible).sum()) >= 1
    # Every dropped frame: target_signal is *exactly* zero everywhere (no PSF tails).
    for t in range(cfg.n_frames):
        if not s.target_visible[t]:
            assert np.all(s.target_signal[t] == 0.0)
            # mask_soft and mask_hard also zero on dropped frames.
            assert np.all(s.mask_soft[t] == 0.0)
            assert not s.mask_hard[t].any()


def test_target_dropout_does_not_remove_background():
    cfg = _base(
        n_frames=20, target_dropout_enabled=True, target_dropout_prob=0.5,
        snr_db=-3.0, noise_sigma=1.0, seed=2,
    )
    s = generate_sequence(cfg)
    for t in range(cfg.n_frames):
        if not s.target_visible[t]:
            # frames[t] = awgn (no target, no clutter, no distractor) -> nonzero std
            assert float(np.std(s.frames[t])) > 0.1


def test_target_min_visible_frames_respected():
    cfg = _base(
        n_frames=10, target_dropout_enabled=True, target_dropout_prob=0.95,
        target_min_visible_frames=3, seed=3,
    )
    s = generate_sequence(cfg)
    assert int(s.target_visible.sum()) >= 3


def test_no_target_dropout_when_disabled():
    cfg = _base(target_dropout_enabled=False, target_dropout_prob=0.9)
    s = generate_sequence(cfg)
    assert s.target_visible.all()


def test_target_signal_clean_full_render_even_when_dropped():
    """target_signal_clean is the *pre-dropout* render — it should be populated on every frame."""
    cfg = _base(
        n_frames=10, target_dropout_enabled=True, target_dropout_prob=0.5,
        snr_db=10.0, seed=4,
    )
    s = generate_sequence(cfg)
    # Every frame in clean has a target rendered.
    for t in range(cfg.n_frames):
        assert s.target_signal_clean[t].max() > 0.0


# ---------------------------------------------------------------------------
# B. Empty background
# ---------------------------------------------------------------------------

def test_empty_background_no_target_no_distractors():
    cfg = _base(sequence_type="empty_background", snr_db=10.0)
    s = generate_sequence(cfg)
    assert s.has_target is False
    assert s.sequence_type == "empty_background"
    assert np.all(s.target_signal == 0.0)
    assert np.all(s.target_signal_clean == 0.0)
    assert np.all(s.mask_soft == 0.0)
    assert not s.mask_hard.any()
    assert np.all(np.isnan(s.positions))
    assert not s.target_visible.any()
    assert np.all(s.distractor_signal == 0.0)
    assert s.distractor_tracks == []
    # AWGN still present.
    assert float(np.std(s.frames)) > 0.1
    # SNR / SCNR NaN for target-absent.
    assert math.isnan(s.snr_db_measured)
    assert np.all(np.isnan(s.effective_scnr_db_per_frame))
    assert math.isnan(s.effective_scnr_db_mean)


def test_empty_background_keeps_clutter():
    cfg = _base(sequence_type="empty_background", clutter_rms=0.5, seed=5)
    s = generate_sequence(cfg)
    assert float(np.std(s.clutter_signal)) > 0.0


# ---------------------------------------------------------------------------
# C. Hard negative
# ---------------------------------------------------------------------------

def test_hard_negative_no_target_with_distractors():
    cfg = _base(
        sequence_type="hard_negative",
        distractor_count_min=1, distractor_count_max=3,
        distractor_types=DISTRACTOR_TYPES,
        distractor_peak_snr_db_min=0.0, distractor_peak_snr_db_max=5.0,
        seed=6,
    )
    s = generate_sequence(cfg)
    assert s.has_target is False
    assert s.sequence_type == "hard_negative"
    # No target labels.
    assert np.all(s.target_signal == 0.0)
    assert np.all(s.mask_soft == 0.0)
    assert not s.mask_hard.any()
    assert np.all(np.isnan(s.positions))
    # Distractors present.
    assert float(s.distractor_signal.sum()) != 0.0
    assert len(s.distractor_tracks) >= 1


def test_hard_negative_forces_at_least_one_distractor_even_with_zero_count():
    cfg = _base(
        sequence_type="hard_negative",
        distractor_count_min=0, distractor_count_max=0,
        distractor_peak_snr_db_min=3.0, distractor_peak_snr_db_max=3.0,
        seed=7,
    )
    s = generate_sequence(cfg)
    assert len(s.distractor_tracks) >= 1
    assert float(s.distractor_signal.sum()) != 0.0


# ---------------------------------------------------------------------------
# D. Mixed
# ---------------------------------------------------------------------------

def test_mixed_has_target_and_distractors_but_labels_only_target():
    cfg = _base(
        sequence_type="mixed_uav_and_distractors",
        snr_db=3.0,
        distractor_count_min=1, distractor_count_max=2,
        distractor_peak_snr_db_min=3.0, distractor_peak_snr_db_max=3.0,
        seed=8,
    )
    s = generate_sequence(cfg)
    assert s.has_target is True
    assert s.target_visible.all()
    assert float(s.target_signal.sum()) > 0.0
    assert float(s.distractor_signal.sum()) > 0.0
    # Each visible target frame has exactly one hard-mask pixel — distractors
    # do NOT contribute to the label.
    for t in range(cfg.n_frames):
        assert int(s.mask_hard[t].sum()) == 1
    assert len(s.distractor_tracks) >= 1


# ---------------------------------------------------------------------------
# E. Determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq_type", [
    "positive_uav", "empty_background", "hard_negative", "mixed_uav_and_distractors",
])
def test_determinism_per_sequence_type(seq_type):
    cfg = _base(
        sequence_type=seq_type,
        distractor_count_min=1, distractor_count_max=2,
        clutter_rms=0.3, target_dropout_enabled=True, target_dropout_prob=0.3,
        seed=11,
    )
    s1 = generate_sequence(cfg)
    s2 = generate_sequence(cfg)
    np.testing.assert_array_equal(s1.frames, s2.frames)
    np.testing.assert_array_equal(s1.target_signal, s2.target_signal)
    np.testing.assert_array_equal(s1.distractor_signal, s2.distractor_signal)
    np.testing.assert_array_equal(s1.target_visible, s2.target_visible)
    np.testing.assert_array_equal(
        np.nan_to_num(s1.positions), np.nan_to_num(s2.positions)
    )


def test_different_seeds_differ_for_negative():
    cfg_a = _base(sequence_type="hard_negative", distractor_count_min=2, distractor_count_max=2, seed=1)
    cfg_b = _base(sequence_type="hard_negative", distractor_count_min=2, distractor_count_max=2, seed=2)
    s_a = generate_sequence(cfg_a)
    s_b = generate_sequence(cfg_b)
    assert not np.allclose(s_a.frames, s_b.frames)


# ---------------------------------------------------------------------------
# F. SNR
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("snr_db", [-10.0, 0.0, 5.0])
def test_positive_visible_target_matches_prescribed_snr(snr_db):
    cfg = _base(snr_db=snr_db, seed=13)
    s = generate_sequence(cfg)
    assert s.has_target is True
    assert s.snr_db_measured == pytest.approx(snr_db, abs=0.3)


def test_dropped_target_frames_not_counted_in_visible_snr():
    """When a frame is target-dropout'd, its target_signal peak is 0 — but the
    measured SNR (from the clean render) should still match prescribed."""
    cfg = _base(
        snr_db=3.0,
        target_dropout_enabled=True, target_dropout_prob=0.5,
        seed=14,
    )
    s = generate_sequence(cfg)
    # Measured SNR is from the clean signal, unaffected by visibility dropout.
    assert s.snr_db_measured == pytest.approx(3.0, abs=0.3)
    # And the dropped frames contribute zero peak to target_signal.
    for t in range(cfg.n_frames):
        if not s.target_visible[t]:
            assert float(s.target_signal[t].max()) == 0.0


def test_snr_nan_for_target_absent():
    cfg_a = _base(sequence_type="empty_background", seed=15)
    cfg_b = _base(sequence_type="hard_negative", seed=15)
    for cfg in (cfg_a, cfg_b):
        s = generate_sequence(cfg)
        assert math.isnan(s.snr_db_measured)
        assert math.isnan(s.effective_scnr_db_mean)
        assert np.all(np.isnan(s.effective_scnr_db_per_frame))


# ---------------------------------------------------------------------------
# G. No leakage on dropped frames
# ---------------------------------------------------------------------------

def test_no_psf_leakage_on_dropped_frames():
    """Strong-target dropout: the full PSF contribution must be exactly zero."""
    cfg = _base(
        snr_db=20.0,  # huge target — any leakage would show up
        target_sigma=2.0,
        target_dropout_enabled=True, target_dropout_prob=0.8,
        target_min_visible_frames=1,
        n_frames=15,
        seed=16,
    )
    s = generate_sequence(cfg)
    # At least one dropped frame.
    assert int((~s.target_visible).sum()) >= 1
    for t in range(cfg.n_frames):
        if not s.target_visible[t]:
            # No nonzero pixel anywhere on a dropped frame's target_signal.
            assert np.count_nonzero(s.target_signal[t]) == 0


# ---------------------------------------------------------------------------
# Effective SCNR
# ---------------------------------------------------------------------------

def test_effective_scnr_finite_on_visible_target_frames():
    cfg = _base(snr_db=5.0, clutter_rms=0.3, seed=17)
    s = generate_sequence(cfg)
    finite = np.isfinite(s.effective_scnr_db_per_frame)
    # All visible frames should yield a finite SCNR (assuming the target is far enough from the edge).
    assert (finite == s.target_visible).any()
    assert math.isfinite(s.effective_scnr_db_mean)


def test_effective_scnr_drops_with_clutter():
    """SCNR should be lower when clutter is added (same target SNR)."""
    cfg_clean = _base(snr_db=5.0, clutter_rms=0.0, seed=18)
    cfg_clut  = _base(snr_db=5.0, clutter_rms=1.0, seed=18)
    s_clean = generate_sequence(cfg_clean)
    s_clut  = generate_sequence(cfg_clut)
    assert s_clut.effective_scnr_db_mean < s_clean.effective_scnr_db_mean


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_sequence_type_raises():
    cfg = _base()
    object.__setattr__(cfg, "sequence_type", "not_a_real_type")
    with pytest.raises(ValueError):
        generate_sequence(cfg)


def test_invalid_distractor_type_raises():
    cfg = _base(
        sequence_type="hard_negative",
        distractor_types=("not_a_real_type",),
    )
    with pytest.raises(ValueError):
        generate_sequence(cfg)
