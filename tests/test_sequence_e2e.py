import numpy as np
import pytest

from synthetic.sequence import SequenceConfig, generate_sequence


def test_basic_shapes():
    cfg = SequenceConfig(n_frames=10, canvas_shape=(256, 256), snr_db=0.0, seed=42)
    s = generate_sequence(cfg)
    assert s.frames.shape == (10, 256, 256)
    assert s.target_signal.shape == (10, 256, 256)
    assert s.positions.shape == (10, 2)
    assert s.mask_soft.shape == (10, 256, 256)
    assert s.mask_hard.shape == (10, 256, 256)
    assert s.observed.shape == (10,)
    assert s.observed.dtype == np.bool_
    assert s.mask_hard.dtype == np.bool_
    assert s.frames.dtype == np.float32


def test_seed_determinism():
    cfg = SequenceConfig(n_frames=10, snr_db=-5.0, seed=123)
    s1 = generate_sequence(cfg)
    s2 = generate_sequence(cfg)
    np.testing.assert_array_equal(s1.frames, s2.frames)
    np.testing.assert_array_equal(s1.positions, s2.positions)
    assert s1.snr_db_measured == s2.snr_db_measured


def test_different_seeds_differ():
    s1 = generate_sequence(SequenceConfig(n_frames=10, snr_db=0.0, seed=1))
    s2 = generate_sequence(SequenceConfig(n_frames=10, snr_db=0.0, seed=2))
    assert not np.allclose(s1.frames, s2.frames)


@pytest.mark.parametrize("snr_db", [-25.0, -10.0, 0.0, 5.0])
def test_measured_snr_matches_prescribed(snr_db):
    cfg = SequenceConfig(n_frames=5, snr_db=snr_db, seed=42)
    s = generate_sequence(cfg)
    assert abs(s.snr_db_measured - snr_db) < 0.3


def test_mask_hard_one_pixel_per_frame():
    s = generate_sequence(SequenceConfig(n_frames=10, snr_db=10.0, seed=42))
    for t in range(10):
        assert int(s.mask_hard[t].sum()) == 1


def test_mask_soft_peak_unity():
    s = generate_sequence(SequenceConfig(n_frames=5, snr_db=0.0, seed=42))
    for t in range(5):
        assert abs(float(s.mask_soft[t].max()) - 1.0) < 1e-5


def test_dropout_rate_statistical():
    n_frames = 20
    p = 0.3
    n_trials = 200
    total = 0
    for s_idx in range(n_trials):
        cfg = SequenceConfig(n_frames=n_frames, snr_db=0.0, dropout_prob=p, seed=s_idx)
        s = generate_sequence(cfg)
        assert bool(s.observed[0])
        total += int(s.observed.sum())
    # First frame guaranteed observed; others Bernoulli(1-p).
    expected_per = 1.0 + (n_frames - 1) * (1.0 - p)
    expected = expected_per * n_trials
    assert abs(total - expected) < 0.05 * expected


def test_clutter_changes_frames_keeps_target():
    cfg_no = SequenceConfig(n_frames=5, snr_db=0.0, clutter_rms=0.0, seed=42)
    cfg_yes = SequenceConfig(n_frames=5, snr_db=0.0, clutter_rms=0.5, seed=42)
    s_no = generate_sequence(cfg_no)
    s_yes = generate_sequence(cfg_yes)
    assert not np.allclose(s_no.frames, s_yes.frames)
    # Target signal (noise/clutter-free) should be identical -- only the additive noise differs.
    np.testing.assert_array_equal(s_no.target_signal, s_yes.target_signal)


def test_maneuvers_changes_trajectory():
    s_no = generate_sequence(SequenceConfig(n_frames=20, snr_db=0.0, maneuvers=False, seed=42))
    s_yes = generate_sequence(
        SequenceConfig(n_frames=20, snr_db=0.0, maneuvers=True, maneuver_prob=0.5, seed=42)
    )
    assert not np.allclose(s_no.positions, s_yes.positions)


def test_jitter_preserves_snr():
    cfg = SequenceConfig(n_frames=5, snr_db=-5.0, jitter_std_px=2.0, seed=42)
    s = generate_sequence(cfg)
    assert abs(s.snr_db_measured - (-5.0)) < 0.3


def test_positions_are_in_canvas():
    s = generate_sequence(SequenceConfig(n_frames=20, snr_db=0.0, seed=7))
    H, W = s.config.canvas_shape
    assert (s.positions[:, 0] >= 0).all() and (s.positions[:, 0] <= H - 1).all()
    assert (s.positions[:, 1] >= 0).all() and (s.positions[:, 1] <= W - 1).all()
