"""Tests for the distractor renderers."""

from __future__ import annotations

import numpy as np
import pytest

from synthetic.distractors import (
    DISTRACTOR_TYPES,
    DistractorTrack,
    render_distractors,
)
from synthetic.sequence import SequenceConfig


def _cfg(**overrides) -> SequenceConfig:
    """A SequenceConfig with distractors switched on."""
    return SequenceConfig(
        n_frames=overrides.pop("n_frames", 12),
        canvas_shape=overrides.pop("canvas_shape", (96, 96)),
        sequence_type=overrides.pop("sequence_type", "hard_negative"),
        snr_db=overrides.pop("snr_db", 0.0),
        noise_sigma=overrides.pop("noise_sigma", 1.0),
        distractors_enabled=True,
        distractor_count_min=overrides.pop("distractor_count_min", 1),
        distractor_count_max=overrides.pop("distractor_count_max", 1),
        distractor_types=overrides.pop("distractor_types", DISTRACTOR_TYPES),
        distractor_peak_snr_db_min=overrides.pop("distractor_peak_snr_db_min", -5.0),
        distractor_peak_snr_db_max=overrides.pop("distractor_peak_snr_db_max", 5.0),
        distractor_sigma_min=overrides.pop("distractor_sigma_min", 0.5),
        distractor_sigma_max=overrides.pop("distractor_sigma_max", 2.0),
        distractor_lifetime_min=overrides.pop("distractor_lifetime_min", 2),
        distractor_lifetime_max=overrides.pop("distractor_lifetime_max", 8),
        distractor_dropout_prob=overrides.pop("distractor_dropout_prob", 0.0),
        distractor_motion_max_px=overrides.pop("distractor_motion_max_px", 2.0),
        seed=overrides.pop("seed", 7),
        **overrides,
    )


@pytest.mark.parametrize("dtype", list(DISTRACTOR_TYPES))
def test_each_distractor_type_produces_signal(dtype):
    cfg = _cfg(distractor_types=(dtype,))
    rng = np.random.default_rng(123)
    sig, tracks = render_distractors(
        cfg, cfg.n_frames, cfg.canvas_shape, cfg.noise_sigma, rng, force_min_one=True
    )
    assert sig.shape == (cfg.n_frames, *cfg.canvas_shape)
    assert sig.dtype == np.float32
    assert len(tracks) >= 1
    tr = tracks[0]
    assert isinstance(tr, DistractorTrack)
    assert tr.type == dtype
    assert tr.visible.shape == (cfg.n_frames,)
    assert tr.positions.shape == (cfg.n_frames, 2)
    # Non-visible frames have NaN positions.
    np.testing.assert_array_equal(np.isnan(tr.positions[:, 0]), ~tr.visible)
    # At least one frame is visible (otherwise the distractor would be invisible).
    assert int(tr.visible.sum()) >= 1
    # The signal is nonzero on at least one visible frame.
    visible_signal_sum = float(sig[tr.visible].sum())
    assert visible_signal_sum > 0.0


def test_render_distractors_deterministic():
    cfg = _cfg(distractor_count_min=2, distractor_count_max=3)
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    s1, t1 = render_distractors(cfg, cfg.n_frames, cfg.canvas_shape, 1.0, rng1, force_min_one=True)
    s2, t2 = render_distractors(cfg, cfg.n_frames, cfg.canvas_shape, 1.0, rng2, force_min_one=True)
    np.testing.assert_array_equal(s1, s2)
    assert len(t1) == len(t2)
    for a, b in zip(t1, t2):
        assert a.type == b.type
        np.testing.assert_array_equal(a.visible, b.visible)
        np.testing.assert_array_equal(np.nan_to_num(a.positions), np.nan_to_num(b.positions))


def test_render_distractors_different_seeds_differ():
    cfg = _cfg(distractor_count_min=2, distractor_count_max=2)
    s1, _ = render_distractors(cfg, cfg.n_frames, cfg.canvas_shape, 1.0,
                               np.random.default_rng(1), force_min_one=True)
    s2, _ = render_distractors(cfg, cfg.n_frames, cfg.canvas_shape, 1.0,
                               np.random.default_rng(2), force_min_one=True)
    assert not np.allclose(s1, s2)


def test_force_min_one_overrides_zero_count():
    cfg = _cfg(distractor_count_min=0, distractor_count_max=0)
    _, tracks = render_distractors(
        cfg, cfg.n_frames, cfg.canvas_shape, 1.0, np.random.default_rng(3), force_min_one=True
    )
    assert len(tracks) >= 1


def test_zero_count_returns_empty_when_not_forced():
    cfg = _cfg(distractor_count_min=0, distractor_count_max=0)
    sig, tracks = render_distractors(
        cfg, cfg.n_frames, cfg.canvas_shape, 1.0, np.random.default_rng(3), force_min_one=False
    )
    assert tracks == []
    assert float(sig.sum()) == 0.0


def test_distractor_count_in_range():
    cfg = _cfg(distractor_count_min=2, distractor_count_max=4)
    for s in range(20):
        _, tracks = render_distractors(
            cfg, cfg.n_frames, cfg.canvas_shape, 1.0, np.random.default_rng(s)
        )
        assert 2 <= len(tracks) <= 4


def test_unknown_distractor_type_raises():
    cfg = _cfg(distractor_types=("not_a_real_type",))
    with pytest.raises(ValueError):
        render_distractors(cfg, cfg.n_frames, cfg.canvas_shape, 1.0, np.random.default_rng(0))


@pytest.mark.parametrize("dtype", ["flicker_blob", "moving_false_blob", "static_hot_pixel"])
def test_distractor_peak_matches_snr_convention(dtype):
    """Rendered peak should equal ``amplitude_for_snr(snr_db, noise_sigma)`` for the chosen distractor."""
    from synthetic.snr import amplitude_for_snr

    cfg = _cfg(
        distractor_types=(dtype,),
        distractor_peak_snr_db_min=3.0,
        distractor_peak_snr_db_max=3.0,
        distractor_sigma_min=1.0,
        distractor_sigma_max=1.0,
        distractor_dropout_prob=0.0,
        n_frames=10,
        canvas_shape=(96, 96),
    )
    rng = np.random.default_rng(0)
    sig, tracks = render_distractors(
        cfg, cfg.n_frames, cfg.canvas_shape, cfg.noise_sigma, rng, force_min_one=True
    )
    assert tracks[0].peak_snr_db == pytest.approx(3.0, abs=1e-9)
    expected = amplitude_for_snr(3.0, cfg.noise_sigma)
    measured_peak = float(sig.max())
    # Rendering peak-normalizes to amplitude exactly per render_gaussian, but
    # multiple visible-frame draws share the peak (per-frame max == amplitude).
    assert measured_peak == pytest.approx(expected, abs=1e-4)


def test_multi_noise_dots_spans_multiple_frames():
    cfg = _cfg(distractor_types=("multi_noise_dots",), n_frames=10)
    _, tracks = render_distractors(
        cfg, cfg.n_frames, cfg.canvas_shape, 1.0, np.random.default_rng(0), force_min_one=True
    )
    visible = tracks[0].visible
    # ~10-30 dots scattered over 10 frames -> at least a few frames visible.
    assert int(visible.sum()) >= 3
