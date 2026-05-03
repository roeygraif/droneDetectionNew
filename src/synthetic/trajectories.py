"""Target motion models: CV, CA, optional maneuvers, with boundary rejection."""

from __future__ import annotations

from typing import Literal

import numpy as np


def _rotate(vec: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


def _propagate(
    n_frames: int,
    init_pos: np.ndarray,
    init_vel: np.ndarray,
    accel: np.ndarray,
    brownian_std: float,
    maneuver_prob: float,
    heading_std_rad: float,
    rng: np.random.Generator,
) -> np.ndarray:
    pos = np.zeros((n_frames, 2), dtype=np.float64)
    pos[0] = init_pos
    vel = init_vel.astype(np.float64).copy()
    a = accel.astype(np.float64).copy()
    for t in range(1, n_frames):
        if maneuver_prob > 0.0 and rng.random() < maneuver_prob:
            vel = _rotate(vel, rng.normal(0.0, heading_std_rad))
        vel = vel + a
        step_noise = rng.normal(0.0, brownian_std, size=2) if brownian_std > 0 else 0.0
        pos[t] = pos[t - 1] + vel + step_noise
    return pos


def _trajectory_in_bounds(
    positions: np.ndarray, canvas_shape: tuple[int, int], margin: float
) -> bool:
    H, W = canvas_shape
    return bool(
        (positions[:, 0] >= margin).all()
        and (positions[:, 0] <= H - 1 - margin).all()
        and (positions[:, 1] >= margin).all()
        and (positions[:, 1] <= W - 1 - margin).all()
    )


def sample_trajectory(
    n_frames: int,
    canvas_shape: tuple[int, int],
    motion: Literal["cv", "ca"] = "cv",
    maneuvers: bool = False,
    maneuver_prob: float = 0.0,
    heading_std_rad: float = 0.0,
    brownian_std_px: float = 0.0,
    speed_min: float = 0.3,
    speed_max: float = 1.5,
    accel_std: float = 0.0,
    margin: float = 4.0,
    rng: np.random.Generator | None = None,
    max_attempts: int = 200,
) -> np.ndarray:
    """Sample an in-bounds trajectory of shape ``(n_frames, 2)`` (y, x positions).

    Initial conditions are drawn from a sensible prior (uniform position within
    margins, uniform heading, uniform speed in ``[speed_min, speed_max]``).
    Trajectories that would exit the canvas (with safety ``margin`` from the
    edge) are rejected and resampled. Raises ``RuntimeError`` if more than
    ``max_attempts`` rejections occur — this signals the parameters are too
    aggressive (e.g., speed too high for the canvas size and N).
    """
    if rng is None:
        rng = np.random.default_rng()
    if motion not in ("cv", "ca"):
        raise ValueError(f"motion must be 'cv' or 'ca', got {motion!r}")

    H, W = canvas_shape
    eff_maneuver_prob = maneuver_prob if maneuvers else 0.0

    for _ in range(max_attempts):
        init_pos = rng.uniform([margin, margin], [H - 1 - margin, W - 1 - margin])
        speed = rng.uniform(speed_min, speed_max)
        heading = rng.uniform(0.0, 2.0 * np.pi)
        init_vel = speed * np.array([np.sin(heading), np.cos(heading)])

        if motion == "ca" and accel_std > 0.0:
            accel = rng.normal(0.0, accel_std, size=2)
        else:
            accel = np.zeros(2)

        traj = _propagate(
            n_frames=n_frames,
            init_pos=init_pos,
            init_vel=init_vel,
            accel=accel,
            brownian_std=brownian_std_px,
            maneuver_prob=eff_maneuver_prob,
            heading_std_rad=heading_std_rad,
            rng=rng,
        )
        if _trajectory_in_bounds(traj, canvas_shape, margin):
            return traj

    raise RuntimeError(
        f"Failed to sample an in-bounds trajectory after {max_attempts} attempts. "
        f"Reduce speed_max ({speed_max}) or canvas margin ({margin})."
    )
