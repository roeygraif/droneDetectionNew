"""Sequence composition: trajectory + target render + AWGN + clutter + distractors + corruptions.

Supports four sequence types:
  - ``positive_uav``                -- target present, no distractors by default
  - ``empty_background``            -- no target, only noise / clutter
  - ``hard_negative``               -- no target, one or more distractors
  - ``mixed_uav_and_distractors``   -- target present alongside distractors

The same deterministic-by-seed contract applies to all four: ``cfg.seed``
fully determines every random draw, including which sequence_type's branch
runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from synthetic.backgrounds import _fourier_shift, awgn, clutter_sequence
from synthetic.corruptions import apply_frame_dropout
from synthetic.distractors import DISTRACTOR_TYPES, DistractorTrack, render_distractors
from synthetic.snr import amplitude_for_snr, effective_scnr_db, measure_peak_snr_db
from synthetic.targets import render_gaussian
from synthetic.trajectories import sample_trajectory


SEQUENCE_TYPES = (
    "positive_uav",
    "empty_background",
    "hard_negative",
    "mixed_uav_and_distractors",
)


@dataclass
class SequenceConfig:
    """All knobs for synthetic sequence generation. Defaults give a clean positive baseline."""

    n_frames: int = 10
    canvas_shape: tuple[int, int] = (256, 256)

    # Sequence type — picks UAV / negative / mixed
    sequence_type: Literal[
        "positive_uav", "empty_background", "hard_negative", "mixed_uav_and_distractors"
    ] = "positive_uav"

    # Target peak SNR (independent variable for the sweep)
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

    # Frame-level / sensor corruption (zeros or repeats the *whole frame*)
    dropout_prob: float = 0.0
    dropout_mode: Literal["blank", "repeat_last"] = "blank"
    jitter_std_px: float = 0.0

    # Target-level visibility intermittency (zeros the target PSF for that frame,
    # but leaves clutter / AWGN / distractors intact). Separate from frame
    # dropout so the noise/clutter regression for the missing-target case is
    # still meaningful.
    target_dropout_enabled: bool = False
    target_dropout_prob: float = 0.0
    target_dropout_mode: Literal["bernoulli"] = "bernoulli"
    target_min_visible_frames: int = 1

    # Distractors (hard negatives) — see synthetic.distractors
    distractors_enabled: bool = False
    distractor_count_min: int = 0
    distractor_count_max: int = 5
    distractor_types: tuple[str, ...] = field(default_factory=lambda: DISTRACTOR_TYPES)
    distractor_peak_snr_db_min: float = -20.0
    distractor_peak_snr_db_max: float = 5.0
    distractor_sigma_min: float = 0.5
    distractor_sigma_max: float = 2.0
    distractor_lifetime_min: int = 1
    distractor_lifetime_max: int = 10
    distractor_dropout_prob: float = 0.0
    distractor_motion_max_px: float = 3.0

    boundary_margin_px: float = 4.0
    seed: int = 0


@dataclass
class SequenceSample:
    """One generated sample.

    Notes:
    - ``target_signal`` is *after* target-visibility dropout (zeros on dropped
      frames). ``target_signal_clean`` is the same render *before* target
      dropout — useful for SNR checks and as an oracle.
    - For target-absent sequences (``empty_background`` / ``hard_negative``),
      ``target_signal*`` are zeros, masks are zeros, ``positions`` are NaN, and
      SNR / SCNR fields are NaN.
    - ``distractor_tracks`` is a list of :class:`DistractorTrack`. Each track's
      ``positions`` are NaN on frames where the distractor is not visible. The
      tracks are *not* labels for the model — they exist purely for analysis.
    """

    frames: np.ndarray              # (N, H, W) float32 -- observed sequence
    target_signal: np.ndarray       # (N, H, W) float32 -- target after visibility dropout
    target_signal_clean: np.ndarray # (N, H, W) float32 -- target before visibility dropout
    distractor_signal: np.ndarray   # (N, H, W) float32
    clutter_signal: np.ndarray      # (N, H, W) float32 -- zeros when disabled
    awgn_field: np.ndarray          # (N, H, W) float32 -- per-pixel readout noise sample
    positions: np.ndarray           # (N, 2) float32 -- apparent (y, x); NaN if no target
    target_visible: np.ndarray      # (N,) bool
    mask_soft: np.ndarray           # (N, H, W) float32 -- Gaussian heatmap, peak=1
    mask_hard: np.ndarray           # (N, H, W) bool
    observed: np.ndarray            # (N,) bool -- false on frame-dropped frames
    has_target: bool
    sequence_type: str
    distractor_tracks: list[DistractorTrack]
    snr_db_prescribed: float
    snr_db_measured: float
    effective_scnr_db_per_frame: np.ndarray   # (N,) float32; NaN where not applicable
    effective_scnr_db_mean: float             # NaN if no visible target frames
    local_background_std_per_frame: np.ndarray  # (N,) float32; NaN where not applicable
    config: SequenceConfig = field(repr=False)


def _sample_target_visibility(cfg: SequenceConfig, n_frames: int, rng: np.random.Generator) -> np.ndarray:
    """Per-frame target visibility mask. Honors ``target_min_visible_frames``."""
    if not cfg.target_dropout_enabled or cfg.target_dropout_prob <= 0.0:
        return np.ones(n_frames, dtype=bool)
    if cfg.target_dropout_mode != "bernoulli":
        raise ValueError(f"Unsupported target_dropout_mode {cfg.target_dropout_mode!r}")

    min_visible = max(0, min(int(n_frames), int(cfg.target_min_visible_frames)))
    p = float(cfg.target_dropout_prob)
    for _ in range(20):
        visible = rng.random(n_frames) >= p
        if int(visible.sum()) >= min_visible:
            return visible
    # Fallback: pick exactly ``min_visible`` random frames to be visible.
    visible = np.zeros(n_frames, dtype=bool)
    if min_visible > 0:
        idx = rng.choice(n_frames, size=min_visible, replace=False)
        visible[idx] = True
    return visible


def _per_frame_scnr_and_bgstd(
    apparent_positions: np.ndarray,
    target_visible: np.ndarray,
    target_peak: float,
    awgn_field: np.ndarray,
    clutter_signal: np.ndarray,
    distractor_signal: np.ndarray,
    target_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-frame ``(effective_scnr_db, local_bg_std)``. NaN where target is not visible."""
    n = awgn_field.shape[0]
    eff = np.full(n, math.nan, dtype=np.float32)
    bg_std = np.full(n, math.nan, dtype=np.float32)
    bg = awgn_field + clutter_signal + distractor_signal  # background-only
    for t in range(n):
        if not bool(target_visible[t]):
            continue
        cy = float(apparent_positions[t, 0])
        cx = float(apparent_positions[t, 1])
        if not (math.isfinite(cy) and math.isfinite(cx)):
            continue
        scnr, s = effective_scnr_db(target_peak, bg[t], cy, cx, target_sigma)
        eff[t] = np.float32(scnr)
        bg_std[t] = np.float32(s)
    return eff, bg_std


def generate_sequence(cfg: SequenceConfig) -> SequenceSample:
    """Generate one sequence end-to-end. Deterministic given ``cfg.seed``."""
    if cfg.sequence_type not in SEQUENCE_TYPES:
        raise ValueError(f"Unknown sequence_type {cfg.sequence_type!r}; valid: {SEQUENCE_TYPES}")

    rng = np.random.default_rng(cfg.seed)
    H, W = cfg.canvas_shape
    N = cfg.n_frames

    has_target = cfg.sequence_type in ("positive_uav", "mixed_uav_and_distractors")
    seq_needs_distractors = cfg.sequence_type in ("hard_negative", "mixed_uav_and_distractors")
    render_distractors_flag = seq_needs_distractors or bool(cfg.distractors_enabled)

    # --- Camera jitter (shifts target + clutter together; not AWGN) ----------
    if cfg.jitter_std_px > 0.0:
        jitter_offsets = rng.normal(0.0, cfg.jitter_std_px, size=(N, 2))
    else:
        jitter_offsets = np.zeros((N, 2), dtype=np.float64)

    # --- Target rendering ----------------------------------------------------
    target_signal_clean = np.zeros((N, H, W), dtype=np.float32)
    target_signal = np.zeros((N, H, W), dtype=np.float32)
    mask_soft = np.zeros((N, H, W), dtype=np.float32)
    mask_hard = np.zeros((N, H, W), dtype=bool)
    apparent_positions = np.full((N, 2), np.nan, dtype=np.float32)
    target_visible = np.zeros(N, dtype=bool)
    target_peak = 0.0

    if has_target:
        trajectory = sample_trajectory(
            n_frames=N,
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
        target_visible = _sample_target_visibility(cfg, N, rng)
        ap = trajectory + jitter_offsets
        apparent_positions = ap.astype(np.float32)
        target_peak = amplitude_for_snr(cfg.snr_db, cfg.noise_sigma)

        for t in range(N):
            cy, cx = float(ap[t, 0]), float(ap[t, 1])
            # Render the clean (visibility-independent) signal every frame —
            # it serves as the SNR oracle and is useful for debugging.
            render_gaussian(target_signal_clean[t], cy, cx, cfg.target_sigma, target_peak)
            if target_visible[t]:
                # Only contribute to the *added* target signal, masks, and
                # frames when the target is visible — no leakage of PSF tails
                # on dropped frames.
                render_gaussian(target_signal[t], cy, cx, cfg.target_sigma, target_peak)
                render_gaussian(mask_soft[t], cy, cx, cfg.target_sigma, 1.0)
                cy_i, cx_i = int(round(cy)), int(round(cx))
                if 0 <= cy_i < H and 0 <= cx_i < W:
                    mask_hard[t, cy_i, cx_i] = True

    # --- Distractors ---------------------------------------------------------
    if render_distractors_flag:
        distractor_signal, distractor_tracks = render_distractors(
            cfg, N, cfg.canvas_shape, cfg.noise_sigma, rng,
            force_min_one=seq_needs_distractors,
        )
    else:
        distractor_signal = np.zeros((N, H, W), dtype=np.float32)
        distractor_tracks = []

    # --- AWGN (per-pixel; unaffected by jitter) ------------------------------
    awgn_field = awgn((N, H, W), cfg.noise_sigma, rng)

    # --- Clutter (optional, temporally coherent) -----------------------------
    clutter_signal = np.zeros((N, H, W), dtype=np.float32)
    if cfg.clutter_rms > 0.0:
        clutter = clutter_sequence(
            cfg.canvas_shape,
            N,
            cfg.clutter_spectral_exponent,
            cfg.clutter_rms,
            cfg.clutter_drift_px_per_frame,
            rng,
        )
        if cfg.jitter_std_px > 0.0:
            for t in range(N):
                clutter[t] = _fourier_shift(
                    clutter[t], float(jitter_offsets[t, 0]), float(jitter_offsets[t, 1])
                )
        clutter_signal = clutter.astype(np.float32, copy=False)

    # --- Compose observed frames --------------------------------------------
    frames = target_signal + distractor_signal + awgn_field + clutter_signal

    # --- Frame-level dropout (applied LAST, after all composition) -----------
    if cfg.dropout_prob > 0.0:
        frames, observed = apply_frame_dropout(frames, cfg.dropout_prob, cfg.dropout_mode, rng)
    else:
        observed = np.ones(N, dtype=bool)

    # --- SNR / SCNR ----------------------------------------------------------
    if has_target:
        # Peak SNR is computed against the clean rendered target — this matches
        # the prescribed value regardless of which frames were visibility-dropped.
        snr_db_measured = measure_peak_snr_db(target_signal_clean, awgn_field)
        eff_per_frame, bg_std_per_frame = _per_frame_scnr_and_bgstd(
            apparent_positions, target_visible, target_peak,
            awgn_field, clutter_signal, distractor_signal,
            cfg.target_sigma,
        )
        mask = target_visible & np.isfinite(eff_per_frame)
        eff_mean = float(np.mean(eff_per_frame[mask])) if mask.any() else math.nan
    else:
        snr_db_measured = math.nan
        eff_per_frame = np.full(N, math.nan, dtype=np.float32)
        bg_std_per_frame = np.full(N, math.nan, dtype=np.float32)
        eff_mean = math.nan

    return SequenceSample(
        frames=frames.astype(np.float32, copy=False),
        target_signal=target_signal,
        target_signal_clean=target_signal_clean,
        distractor_signal=distractor_signal.astype(np.float32, copy=False),
        clutter_signal=clutter_signal,
        awgn_field=awgn_field.astype(np.float32, copy=False),
        positions=apparent_positions,
        target_visible=target_visible,
        mask_soft=mask_soft,
        mask_hard=mask_hard,
        observed=observed,
        has_target=has_target,
        sequence_type=cfg.sequence_type,
        distractor_tracks=distractor_tracks,
        snr_db_prescribed=float(cfg.snr_db),
        snr_db_measured=float(snr_db_measured),
        effective_scnr_db_per_frame=eff_per_frame,
        effective_scnr_db_mean=float(eff_mean),
        local_background_std_per_frame=bg_std_per_frame,
        config=cfg,
    )
