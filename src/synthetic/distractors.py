"""Hard-negative distractors for false-alarm-aware training.

Distractors are negative examples: clutter that *looks* drone-like in a single
frame or over short temporal windows, but is not a real UAV. They exist so the
detector learns to discriminate true tracks from plausible-but-false ones.

Each renderer returns:
  - ``signal`` -- ``(N, H, W)`` float32 contribution to be added to the scene.
  - ``track``  -- a :class:`DistractorTrack` with positions (``NaN`` on invisible
                  frames), a per-frame visibility mask, peak SNR, sigma,
                  lifetime, and a small ``flags`` dict of type-specific metadata.

Distractors use the same peak-SNR convention as the real target:
    A_peak = noise_sigma * 10**(snr_db / 20)

All randomness flows through the supplied :class:`numpy.random.Generator`
(derived from ``cfg.seed``) so generation is deterministic per config + seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np

from synthetic.snr import amplitude_for_snr
from synthetic.targets import render_gaussian
from synthetic.trajectories import sample_trajectory


DISTRACTOR_TYPES: tuple[str, ...] = (
    "static_hot_pixel",
    "flicker_blob",
    "moving_false_blob",
    "false_tracklet",
    "multi_noise_dots",
)


@dataclass
class DistractorTrack:
    """Per-distractor metadata.

    ``positions`` is ``(N, 2)`` float32; entries are NaN on frames where the
    distractor is not visible (or where the type doesn't have a single
    well-defined center, e.g. ``multi_noise_dots`` — there ``positions[t]`` is
    the last dot rendered in frame ``t``, kept for inspection only).
    """

    type: str
    positions: np.ndarray            # (N, 2) float32, NaN on invisible frames
    visible: np.ndarray              # (N,) bool
    peak_snr_db: float
    sigma: float
    lifetime: int
    flags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Preserve arrays as numpy; asdict deep-copies them already.
        return d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nan_positions(n: int) -> np.ndarray:
    a = np.empty((n, 2), dtype=np.float32)
    a[:] = np.nan
    return a


def _pick_lifetime(cfg, n_frames: int, rng: np.random.Generator) -> int:
    lo = max(1, int(cfg.distractor_lifetime_min))
    hi = max(lo, min(int(n_frames), int(cfg.distractor_lifetime_max)))
    if lo >= hi:
        return min(hi, n_frames)
    return int(rng.integers(lo, hi + 1))


def _pick_window(lifetime: int, n_frames: int, rng: np.random.Generator) -> tuple[int, int]:
    lifetime = min(lifetime, n_frames)
    if lifetime >= n_frames:
        return 0, n_frames
    start = int(rng.integers(0, n_frames - lifetime + 1))
    return start, start + lifetime


def _draw_position(canvas_shape: tuple[int, int], margin: float, rng: np.random.Generator) -> tuple[float, float]:
    H, W = canvas_shape
    margin = float(np.clip(margin, 0.0, min(H, W) / 2 - 1.0))
    y = float(rng.uniform(margin, max(margin + 1.0, H - 1 - margin)))
    x = float(rng.uniform(margin, max(margin + 1.0, W - 1 - margin)))
    return y, x


def _draw_peak_snr(cfg, rng: np.random.Generator) -> float:
    lo, hi = float(cfg.distractor_peak_snr_db_min), float(cfg.distractor_peak_snr_db_max)
    if lo >= hi:
        return lo
    return float(rng.uniform(lo, hi))


def _draw_sigma(cfg, rng: np.random.Generator,
                lo_override: float | None = None,
                hi_override: float | None = None) -> float:
    lo = float(cfg.distractor_sigma_min if lo_override is None else lo_override)
    hi = float(cfg.distractor_sigma_max if hi_override is None else hi_override)
    if lo >= hi:
        return max(0.3, lo)
    return float(rng.uniform(lo, hi))


# ---------------------------------------------------------------------------
# Distractor renderers — one per type
# ---------------------------------------------------------------------------

def _render_static_hot_pixel(cfg, n_frames, canvas_shape, noise_sigma, rng):
    """A narrow, fixed-location 'pixel' that mostly persists across a window."""
    H, W = canvas_shape
    # Hot pixels are pixel-narrow regardless of the global sigma range.
    sigma = max(0.3, _draw_sigma(cfg, rng, hi_override=0.7))
    peak_snr_db = _draw_peak_snr(cfg, rng)
    peak = amplitude_for_snr(peak_snr_db, noise_sigma)
    margin = max(2.0, 3.0 * sigma)
    cy, cx = _draw_position(canvas_shape, margin, rng)

    lifetime = _pick_lifetime(cfg, n_frames, rng)
    t0, t1 = _pick_window(lifetime, n_frames, rng)
    visible = np.zeros(n_frames, dtype=bool)
    visible[t0:t1] = True
    if cfg.distractor_dropout_prob > 0.0:
        flicker = rng.random(n_frames) < float(cfg.distractor_dropout_prob)
        visible &= ~flicker

    signal = np.zeros((n_frames, H, W), dtype=np.float32)
    positions = _nan_positions(n_frames)
    for t in range(n_frames):
        if visible[t]:
            render_gaussian(signal[t], cy, cx, sigma, peak)
            positions[t, 0] = np.float32(cy)
            positions[t, 1] = np.float32(cx)

    return signal, DistractorTrack(
        type="static_hot_pixel",
        positions=positions,
        visible=visible,
        peak_snr_db=peak_snr_db,
        sigma=sigma,
        lifetime=int(visible.sum()),
        flags={"start_frame": int(t0), "end_frame": int(t1)},
    )


def _render_flicker_blob(cfg, n_frames, canvas_shape, noise_sigma, rng):
    """A blob at (roughly) a fixed location that appears/disappears randomly."""
    H, W = canvas_shape
    sigma = _draw_sigma(cfg, rng)
    peak_snr_db = _draw_peak_snr(cfg, rng)
    peak = amplitude_for_snr(peak_snr_db, noise_sigma)
    margin = max(2.0, 3.0 * sigma)
    cy0, cx0 = _draw_position(canvas_shape, margin + 1.0, rng)

    lifetime = _pick_lifetime(cfg, n_frames, rng)
    t0, t1 = _pick_window(lifetime, n_frames, rng)

    # Flicker rate is at least 0.4 so it really flickers, even if the global
    # distractor_dropout_prob is low.
    flicker_rate = float(max(0.4, cfg.distractor_dropout_prob))
    visible = np.zeros(n_frames, dtype=bool)
    for t in range(t0, t1):
        if rng.random() >= flicker_rate:
            visible[t] = True

    signal = np.zeros((n_frames, H, W), dtype=np.float32)
    positions = _nan_positions(n_frames)
    jitter_amp = 0.5
    for t in range(n_frames):
        if visible[t]:
            cy = float(np.clip(cy0 + rng.normal(0.0, jitter_amp), margin, H - 1 - margin))
            cx = float(np.clip(cx0 + rng.normal(0.0, jitter_amp), margin, W - 1 - margin))
            render_gaussian(signal[t], cy, cx, sigma, peak)
            positions[t, 0] = np.float32(cy)
            positions[t, 1] = np.float32(cx)

    return signal, DistractorTrack(
        type="flicker_blob",
        positions=positions,
        visible=visible,
        peak_snr_db=peak_snr_db,
        sigma=sigma,
        lifetime=int(visible.sum()),
        flags={"flicker_rate": flicker_rate, "center": (cy0, cx0)},
    )


def _render_moving_false_blob(cfg, n_frames, canvas_shape, noise_sigma, rng):
    """Drone-sized blob with motion + one 'wrong' behavior flag.

    Behaviors (chosen at random):
      - "jump": teleport in mid-track
      - "wrong_velocity": speed outside UAV envelope
      - "wrong_accel": large acceleration (uses CA motion)
      - "direction_changes": frequent maneuvers
      - "short_lived": lifetime clipped well below N
    """
    H, W = canvas_shape
    sigma = _draw_sigma(cfg, rng)
    peak_snr_db = _draw_peak_snr(cfg, rng)
    peak = amplitude_for_snr(peak_snr_db, noise_sigma)
    margin = max(2.0, 3.0 * sigma)

    behaviors = ("jump", "wrong_velocity", "wrong_accel", "direction_changes", "short_lived")
    behavior = str(rng.choice(behaviors))
    flags: dict = {"behavior": behavior}

    lifetime = _pick_lifetime(cfg, n_frames, rng)
    if behavior == "short_lived":
        lifetime = max(2, min(lifetime, max(2, n_frames // 4)))
    lifetime = min(lifetime, n_frames)
    t0, t1 = _pick_window(lifetime, n_frames, rng)

    speed_min = float(cfg.speed_min_px_per_frame)
    speed_max = float(cfg.distractor_motion_max_px)
    if behavior == "wrong_velocity":
        speed_max = max(cfg.distractor_motion_max_px, 2.0 * float(cfg.speed_max_px_per_frame))
        speed_min = 0.8 * speed_max

    use_ca = behavior == "wrong_accel"
    accel_std = 0.3 if use_ca else 0.0
    maneuvers = behavior == "direction_changes"
    heading_std = 1.2 if maneuvers else 0.3
    maneuver_prob = 0.5 if maneuvers else 0.0

    try:
        traj = sample_trajectory(
            n_frames=t1 - t0,
            canvas_shape=canvas_shape,
            motion="ca" if use_ca else "cv",
            maneuvers=maneuvers,
            maneuver_prob=maneuver_prob,
            heading_std_rad=heading_std,
            brownian_std_px=0.1,
            speed_min=speed_min,
            speed_max=speed_max,
            accel_std=accel_std,
            margin=margin,
            rng=rng,
            max_attempts=50,
        )
    except RuntimeError:
        # Intentionally aggressive params can fail bounds — fall back to static.
        cy, cx = _draw_position(canvas_shape, margin, rng)
        traj = np.tile(np.array([cy, cx], dtype=np.float64), (t1 - t0, 1))
        flags["fallback_static"] = True

    if behavior == "jump" and (t1 - t0) >= 3:
        local_j = int(rng.integers(1, t1 - t0))
        dy = float(rng.uniform(-1.0, 1.0)) * (H * 0.25)
        dx = float(rng.uniform(-1.0, 1.0)) * (W * 0.25)
        traj[local_j:, 0] += dy
        traj[local_j:, 1] += dx
        traj[:, 0] = np.clip(traj[:, 0], margin, H - 1 - margin)
        traj[:, 1] = np.clip(traj[:, 1], margin, W - 1 - margin)
        flags["jump_frame"] = int(t0 + local_j)

    visible = np.zeros(n_frames, dtype=bool)
    visible[t0:t1] = True
    if cfg.distractor_dropout_prob > 0.0:
        flicker = np.zeros(n_frames, dtype=bool)
        flicker[t0:t1] = rng.random(t1 - t0) < float(cfg.distractor_dropout_prob)
        visible &= ~flicker

    signal = np.zeros((n_frames, H, W), dtype=np.float32)
    positions = _nan_positions(n_frames)
    for local_t, t in enumerate(range(t0, t1)):
        if visible[t]:
            cy = float(np.clip(traj[local_t, 0], 0.0, H - 1.0))
            cx = float(np.clip(traj[local_t, 1], 0.0, W - 1.0))
            render_gaussian(signal[t], cy, cx, sigma, peak)
            positions[t, 0] = np.float32(cy)
            positions[t, 1] = np.float32(cx)

    return signal, DistractorTrack(
        type="moving_false_blob",
        positions=positions,
        visible=visible,
        peak_snr_db=peak_snr_db,
        sigma=sigma,
        lifetime=int(visible.sum()),
        flags=flags,
    )


def _render_false_tracklet(cfg, n_frames, canvas_shape, noise_sigma, rng):
    """A few weak blobs that nearly line up into a plausible short track.

    Designed to fool temporal accumulators that link nearby detections across
    frames. Blob positions follow a noisy line; not all frames have a blob.
    """
    H, W = canvas_shape
    sigma = _draw_sigma(cfg, rng)
    peak_snr_db = _draw_peak_snr(cfg, rng)
    peak = amplitude_for_snr(peak_snr_db, noise_sigma)
    margin = max(2.0, 3.0 * sigma)

    n_blobs = int(rng.integers(3, 7))  # 3..6
    n_blobs = min(n_blobs, n_frames)

    speed = float(rng.uniform(float(cfg.speed_min_px_per_frame),
                              max(float(cfg.speed_min_px_per_frame) + 1e-3,
                                  float(cfg.distractor_motion_max_px))))
    heading = float(rng.uniform(0.0, 2.0 * np.pi))
    sy, sx = np.sin(heading), np.cos(heading)
    # Recenter so the endpoint lies within the canvas.
    needed = margin + speed * n_blobs + 2.0
    cy0, cx0 = _draw_position(canvas_shape, needed, rng) if needed * 2 < min(H, W) else _draw_position(canvas_shape, margin, rng)

    signal = np.zeros((n_frames, H, W), dtype=np.float32)
    visible = np.zeros(n_frames, dtype=bool)
    positions = _nan_positions(n_frames)
    chosen_frames = sorted(int(t) for t in rng.choice(n_frames, size=n_blobs, replace=False))
    for k, t in enumerate(chosen_frames):
        cy = cy0 + speed * k * sy + float(rng.normal(0.0, 1.0))
        cx = cx0 + speed * k * sx + float(rng.normal(0.0, 1.0))
        cy = float(np.clip(cy, margin, H - 1 - margin))
        cx = float(np.clip(cx, margin, W - 1 - margin))
        if cfg.distractor_dropout_prob > 0.0 and rng.random() < float(cfg.distractor_dropout_prob):
            continue
        render_gaussian(signal[t], cy, cx, sigma, peak)
        visible[t] = True
        positions[t, 0] = np.float32(cy)
        positions[t, 1] = np.float32(cx)

    return signal, DistractorTrack(
        type="false_tracklet",
        positions=positions,
        visible=visible,
        peak_snr_db=peak_snr_db,
        sigma=sigma,
        lifetime=int(visible.sum()),
        flags={"n_blobs": int(n_blobs), "frames": list(chosen_frames),
               "heading_rad": float(heading), "speed_px_per_frame": float(speed)},
    )


def _render_multi_noise_dots(cfg, n_frames, canvas_shape, noise_sigma, rng):
    """Many low-SNR transient dots scattered across frames (no temporal coherence)."""
    H, W = canvas_shape
    # Dots are small.
    sigma = max(0.3, _draw_sigma(cfg, rng, hi_override=0.8))
    peak_snr_db = _draw_peak_snr(cfg, rng)
    peak = amplitude_for_snr(peak_snr_db, noise_sigma)
    margin = max(2.0, 3.0 * sigma)

    n_dots = int(rng.integers(10, 31))  # 10..30
    signal = np.zeros((n_frames, H, W), dtype=np.float32)
    visible = np.zeros(n_frames, dtype=bool)
    positions = _nan_positions(n_frames)
    dot_records: list[tuple[int, float, float]] = []
    for _ in range(n_dots):
        t = int(rng.integers(0, n_frames))
        cy, cx = _draw_position(canvas_shape, margin, rng)
        render_gaussian(signal[t], cy, cx, sigma, peak)
        visible[t] = True
        positions[t, 0] = np.float32(cy)
        positions[t, 1] = np.float32(cx)
        dot_records.append((t, cy, cx))

    return signal, DistractorTrack(
        type="multi_noise_dots",
        positions=positions,
        visible=visible,
        peak_snr_db=peak_snr_db,
        sigma=sigma,
        lifetime=int(visible.sum()),
        flags={"n_dots": int(n_dots)},
    )


_RENDERERS: dict[str, Callable] = {
    "static_hot_pixel": _render_static_hot_pixel,
    "flicker_blob": _render_flicker_blob,
    "moving_false_blob": _render_moving_false_blob,
    "false_tracklet": _render_false_tracklet,
    "multi_noise_dots": _render_multi_noise_dots,
}


def render_distractors(
    cfg,
    n_frames: int,
    canvas_shape: tuple[int, int],
    noise_sigma: float,
    rng: np.random.Generator,
    force_min_one: bool = False,
) -> tuple[np.ndarray, list[DistractorTrack]]:
    """Render all distractors for one sequence; return ``(signal, tracks)``.

    Count is drawn uniformly from ``[distractor_count_min, distractor_count_max]``.
    When ``force_min_one`` is set (hard_negative / mixed sequence types), the
    lower bound is bumped to 1 so the scene actually has a hard negative in it.
    Types are drawn uniformly from ``cfg.distractor_types`` (or the full set if
    none is configured).
    """
    types = tuple(cfg.distractor_types) if cfg.distractor_types else DISTRACTOR_TYPES
    for t in types:
        if t not in _RENDERERS:
            raise ValueError(f"Unknown distractor type {t!r}; valid: {DISTRACTOR_TYPES}")

    lo = max(0, int(cfg.distractor_count_min))
    hi = max(lo, int(cfg.distractor_count_max))
    if force_min_one:
        lo = max(1, lo)
        hi = max(lo, hi)
    if hi == lo:
        count = lo
    else:
        count = int(rng.integers(lo, hi + 1))

    H, W = canvas_shape
    if count == 0:
        return np.zeros((n_frames, H, W), dtype=np.float32), []

    total = np.zeros((n_frames, H, W), dtype=np.float32)
    tracks: list[DistractorTrack] = []
    for _ in range(count):
        dtype = str(rng.choice(types))
        sig, tr = _RENDERERS[dtype](cfg, n_frames, canvas_shape, noise_sigma, rng)
        total += sig
        tracks.append(tr)
    return total, tracks
