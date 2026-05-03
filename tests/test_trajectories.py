import numpy as np
import pytest

from synthetic.trajectories import sample_trajectory


def test_cv_is_straight_line_when_noiseless():
    rng = np.random.default_rng(42)
    traj = sample_trajectory(
        n_frames=20,
        canvas_shape=(256, 256),
        motion="cv",
        brownian_std_px=0.0,
        rng=rng,
    )
    diffs = np.diff(traj, axis=0)
    assert np.allclose(diffs - diffs[0], 0.0, atol=1e-9)


def test_cv_brownian_std_matches():
    rng = np.random.default_rng(42)
    traj = sample_trajectory(
        n_frames=200,
        canvas_shape=(2048, 2048),
        motion="cv",
        brownian_std_px=1.0,
        margin=8.0,
        rng=rng,
    )
    diffs = np.diff(traj, axis=0)
    # Per-axis std of diffs ~ brownian_std (the constant velocity is removed by differencing).
    assert 0.85 < diffs[:, 0].std() < 1.15
    assert 0.85 < diffs[:, 1].std() < 1.15


def test_ca_second_derivative_constant_when_noiseless():
    rng = np.random.default_rng(42)
    traj = sample_trajectory(
        n_frames=20,
        canvas_shape=(2048, 2048),
        motion="ca",
        accel_std=0.05,
        brownian_std_px=0.0,
        margin=4.0,
        rng=rng,
    )
    second = np.diff(traj, n=2, axis=0)
    assert np.allclose(second - second[0], 0.0, atol=1e-9)


def test_in_bounds_always():
    rng = np.random.default_rng(42)
    H, W = 256, 256
    margin = 10.0
    for _ in range(50):
        traj = sample_trajectory(
            n_frames=20,
            canvas_shape=(H, W),
            motion="cv",
            brownian_std_px=0.5,
            margin=margin,
            rng=rng,
        )
        assert traj[:, 0].min() >= margin - 1e-9
        assert traj[:, 0].max() <= H - 1 - margin + 1e-9
        assert traj[:, 1].min() >= margin - 1e-9
        assert traj[:, 1].max() <= W - 1 - margin + 1e-9


def test_maneuvers_reduce_path_alignment():
    """Without maneuvers, consecutive velocity-vector cosines ~ 1. With maneuvers, lower."""
    rng = np.random.default_rng(42)
    common = dict(
        n_frames=200,
        canvas_shape=(4096, 4096),
        motion="cv",
        brownian_std_px=0.0,
        speed_min=0.5,
        speed_max=0.5,
        margin=4.0,
        max_attempts=500,
    )
    straight = sample_trajectory(**common, maneuvers=False, rng=np.random.default_rng(1))
    swerving = sample_trajectory(
        **common,
        maneuvers=True,
        maneuver_prob=0.5,
        heading_std_rad=1.0,
        rng=np.random.default_rng(2),
    )

    def mean_cos(traj):
        d = np.diff(traj, axis=0)
        n = np.linalg.norm(d, axis=1)
        d = d / np.where(n > 0, n, 1.0)[:, None]
        return float((d[:-1] * d[1:]).sum(axis=1).mean())

    assert mean_cos(straight) > 0.999
    assert mean_cos(swerving) < 0.95


def test_invalid_motion():
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError):
        sample_trajectory(n_frames=10, canvas_shape=(256, 256), motion="invalid", rng=rng)


def test_unfeasible_speed_raises():
    rng = np.random.default_rng(42)
    with pytest.raises(RuntimeError):
        sample_trajectory(
            n_frames=20,
            canvas_shape=(50, 50),
            motion="cv",
            brownian_std_px=0.0,
            speed_min=10.0,
            speed_max=10.0,
            margin=4.0,
            rng=rng,
            max_attempts=5,
        )
