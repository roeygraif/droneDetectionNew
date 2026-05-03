"""Sequence composition: trajectory + target render + AWGN + clutter + corruptions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from synthetic.backgrounds import _fourier_shift, awgn, clutter_sequence
from synthetic.corruptions import apply_frame_dropout
from synthetic.snr import amplitude_for_snr, measure_peak_snr_db
from synthetic.targets import render_gaussian
from synthetic.trajectories import sample_trajectory


@dataclass
class SequenceConfig:
    """All knobs for synthetic sequence generation. Defaults give a clean baseline."""

    n_frames: int = 10
    canvas_shape: tuple[int, int] = (256, 256)

    # Independent variable for the sweep
    snr_db: float = 0.0

    # Target PSF
    target_sigma: float = 1.0

    # Motion model
    motion: Literal["cv", "ca"] = "cv"
    maneuvers: bool = False
    maneuver_prob: float = 0.05
    heading_std_rad: float = 0.3
    brownian_std_px: float = 0.1
    speed_min_px_per_frame: float = 0.3
    speed_max_px_per_frame: float = 1.5
    accel_std_px_per_frame_sq: float = 0.0

    # Noise + clutter
    noise_sigma: float = 1.0
    clutter_rms: float = 0.0
    clutter_spectral_exponent: float = 2.0
    clutter_drift_px_per_frame: float = 0.0

    # Wildcard corruptions
    dropout_prob: float = 0.0
    dropout_mode: Literal["blank", "repeat_last"] = "blank"
    jitter_std_px: float = 0.0

    boundary_margin_px: float = 4.0
    seed: int = 0


@dataclass
class SequenceSample:
    """One generated sample. Numpy arrays; convert to torch in the dataset wrapper."""

    frames: np.ndarray              # (N, H, W) float32 -- the observed sequence
    target_signal: np.ndarray       # (N, H, W) float32 -- noise-free render (oracle / SNR check)
    positions: np.ndarray           # (N, 2)    float32 -- apparent (y, x) target centers
    mask_soft: np.ndarray           # (N, H, W) float32 -- Gaussian heatmap, peak=1
    mask_hard: np.ndarray           # (N, H, W) bool    -- nearest-pixel target indicator
    observed: np.ndarray            # (N,)      bool    -- false on dropped frames
    snr_db_prescribed: float
    snr_db_measured: float
    config: SequenceConfig = field(repr=False)


def generate_sequence(cfg: SequenceConfig) -> SequenceSample:
    """Generate one sequence end-to-end. Deterministic given ``cfg.seed``."""
    rng = np.random.default_rng(cfg.seed)
    H, W = cfg.canvas_shape

    # 1) Trajectory in the stable world frame.
    trajectory = sample_trajectory(
        n_frames=cfg.n_frames,
        canvas_shape=cfg.canvas_shape,
        motion=cfg.motion,
        maneuvers=cfg.maneuvers,
        maneuver_prob=cfg.maneuver_prob,
        heading_std_rad=cfg.heading_std_rad,
        brownian_std_px=cfg.brownian_std_px,
        speed_min=cfg.speed_min_px_per_frame,
        speed_max=cfg.speed_max_px_per_frame,
        accel_std=cfg.accel_std_px_per_frame_sq,
        margin=cfg.boundary_margin_px + 3.0 * cfg.target_sigma,
        rng=rng,
    )

    # 2) Camera jitter offsets per frame (zero if disabled). The camera shakes ->
    #    target and clutter both shift together; AWGN (per-pixel readout noise) does not.
    if cfg.jitter_std_px > 0.0:
        jitter_offsets = rng.normal(0.0, cfg.jitter_std_px, size=(cfg.n_frames, 2))
    else:
        jitter_offsets = np.zeros((cfg.n_frames, 2), dtype=np.float64)

    apparent_positions = trajectory + jitter_offsets

    # 3) Render target + masks at the apparent positions.
    peak = amplitude_for_snr(cfg.snr_db, cfg.noise_sigma)
    target_signal = np.zeros((cfg.n_frames, H, W), dtype=np.float32)
    mask_soft = np.zeros_like(target_signal)
    mask_hard = np.zeros((cfg.n_frames, H, W), dtype=bool)
    for t in range(cfg.n_frames):
        cy, cx = float(apparent_positions[t, 0]), float(apparent_positions[t, 1])
        render_gaussian(target_signal[t], cy, cx, cfg.target_sigma, peak)
        render_gaussian(mask_soft[t], cy, cx, cfg.target_sigma, 1.0)
        cy_i, cx_i = int(round(cy)), int(round(cx))
        if 0 <= cy_i < H and 0 <= cx_i < W:
            mask_hard[t, cy_i, cx_i] = True

    # 4) AWGN (independent per pixel; jitter does not affect it).
    awgn_field = awgn((cfg.n_frames, H, W), cfg.noise_sigma, rng)

    # 5) Clutter (optional, temporally coherent). Apply jitter on top so it shakes with target.
    if cfg.clutter_rms > 0.0:
        clutter = clutter_sequence(
            cfg.canvas_shape,
            cfg.n_frames,
            cfg.clutter_spectral_exponent,
            cfg.clutter_rms,
            cfg.clutter_drift_px_per_frame,
            rng,
        )
        if cfg.jitter_std_px > 0.0:
            for t in range(cfg.n_frames):
                clutter[t] = _fourier_shift(
                    clutter[t], float(jitter_offsets[t, 0]), float(jitter_offsets[t, 1])
                )
        frames = target_signal + awgn_field + clutter
    else:
        frames = target_signal + awgn_field

    # 6) Frame dropout (post-hoc corruption).
    if cfg.dropout_prob > 0.0:
        frames, observed = apply_frame_dropout(frames, cfg.dropout_prob, cfg.dropout_mode, rng)
    else:
        observed = np.ones(cfg.n_frames, dtype=bool)

    # 7) Verification: rendered target peak == prescribed amplitude (peak normalization);
    #    measured SNR therefore matches prescribed within numerical precision.
    snr_db_measured = measure_peak_snr_db(target_signal, awgn_field)

    return SequenceSample(
        frames=frames.astype(np.float32),
        target_signal=target_signal,
        positions=apparent_positions.astype(np.float32),
        mask_soft=mask_soft,
        mask_hard=mask_hard,
        observed=observed,
        snr_db_prescribed=float(cfg.snr_db),
        snr_db_measured=float(snr_db_measured),
        config=cfg,
    )
