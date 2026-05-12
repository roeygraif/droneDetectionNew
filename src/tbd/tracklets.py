"""Beam-search tracklet builder.

Links weak per-frame candidates into short motion-consistent tracklets. The
algorithm is a simple multi-hypothesis tracker:

  - Start hypotheses from the top-K candidates of every frame.
  - At each subsequent frame, extend each active hypothesis to nearby
    candidates (within ``max_link_distance``), penalizing motion changes and
    missed detections.
  - Keep the top ``beam_size`` hypotheses per frame.
  - At the end, return the top ``final_top_m`` distinct tracklets.

Determinism: the algorithm is deterministic given a fixed candidate ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tbd.candidates import Candidate


@dataclass
class TrackletConfig:
    """Knobs for the beam-search tracklet builder."""

    tracklet_len: int = 10
    tracklet_beam_size: int = 200
    tracklet_max_link_distance: float = 4.0
    tracklet_max_misses: int = 3
    tracklet_start_top_k: int = 50
    tracklet_motion_penalty: float = 0.1
    tracklet_miss_penalty: float = 0.5
    tracklet_min_observed_points: int = 3
    tracklet_final_top_m: int = 100


@dataclass
class Tracklet:
    """A short multi-frame hypothesis chain."""

    positions: np.ndarray              # (T, 2) float32 -- (y, x), NaN where missing
    candidate_scores: np.ndarray       # (T,) float32 -- 0 where missing
    start_t: int
    end_t: int                         # exclusive
    score: float
    miss_count: int
    velocity: Optional[np.ndarray] = None  # (2,) float32, last estimated (vy, vx)
    metadata: dict = field(default_factory=dict)


@dataclass
class _Hypothesis:
    """Internal mutable hypothesis used during beam search."""

    positions: list[Optional[tuple[float, float]]]
    scores: list[float]
    accumulated_score: float
    miss_count: int
    consecutive_misses: int
    velocity: Optional[tuple[float, float]]
    last_observed_t: int
    start_t: int
    n_observed: int

    def predict(self, t: int) -> tuple[float, float] | None:
        last = self.last_observed_t
        if last < self.start_t:
            return None
        last_pos = None
        for i in range(len(self.positions) - 1, -1, -1):
            if self.positions[i] is not None:
                last_pos = self.positions[i]
                last_idx = i
                break
        else:
            return None
        if self.velocity is None:
            return last_pos
        dt = t - (self.start_t + last_idx)
        vy, vx = self.velocity
        return (last_pos[0] + vy * dt, last_pos[1] + vx * dt)


def _motion_cost(prev_velocity: tuple[float, float] | None,
                 new_velocity: tuple[float, float]) -> float:
    if prev_velocity is None:
        return 0.0
    dvy = new_velocity[0] - prev_velocity[0]
    dvx = new_velocity[1] - prev_velocity[1]
    return float(np.hypot(dvy, dvx))


def _estimate_velocity(positions: list[Optional[tuple[float, float]]],
                       start_t: int) -> tuple[float, float] | None:
    # Use the last two observed positions to estimate velocity.
    last_idx = None
    prev_idx = None
    for i in range(len(positions) - 1, -1, -1):
        if positions[i] is not None:
            if last_idx is None:
                last_idx = i
            else:
                prev_idx = i
                break
    if last_idx is None or prev_idx is None:
        return None
    dt = last_idx - prev_idx
    if dt <= 0:
        return None
    p_last = positions[last_idx]
    p_prev = positions[prev_idx]
    return ((p_last[0] - p_prev[0]) / dt, (p_last[1] - p_prev[1]) / dt)


def _hypothesis_to_tracklet(h: _Hypothesis, cfg: TrackletConfig) -> Tracklet:
    T = len(h.positions)
    pos_arr = np.full((T, 2), np.nan, dtype=np.float32)
    score_arr = np.zeros(T, dtype=np.float32)
    for i, p in enumerate(h.positions):
        if p is not None:
            pos_arr[i, 0] = np.float32(p[0])
            pos_arr[i, 1] = np.float32(p[1])
            score_arr[i] = np.float32(h.scores[i])
    vel = None
    if h.velocity is not None:
        vel = np.array(h.velocity, dtype=np.float32)
    return Tracklet(
        positions=pos_arr,
        candidate_scores=score_arr,
        start_t=h.start_t,
        end_t=h.start_t + T,
        score=float(h.accumulated_score),
        miss_count=int(h.miss_count),
        velocity=vel,
        metadata={"n_observed": int(h.n_observed)},
    )


def build_tracklets(
    candidates_by_frame: list[list[Candidate]],
    cfg: TrackletConfig | None = None,
) -> list[Tracklet]:
    """Beam-search MHT over per-frame candidates.

    All tracklets are anchored at ``t = 0`` (positions before ``start_t`` are
    NaN, after ``end_t`` are NaN). This makes downstream alignment with the
    full sequence length trivial.
    """
    if cfg is None:
        cfg = TrackletConfig()

    T = len(candidates_by_frame)
    if T == 0:
        return []

    L = max(1, int(cfg.tracklet_len))
    max_d = float(cfg.tracklet_max_link_distance)
    max_misses = int(cfg.tracklet_max_misses)

    # Active hypotheses, keyed by their absolute end frame to keep tidy.
    active: list[_Hypothesis] = []
    finished: list[_Hypothesis] = []

    def _seed_from_candidates(t: int) -> list[_Hypothesis]:
        seeds: list[_Hypothesis] = []
        top = candidates_by_frame[t][: max(0, int(cfg.tracklet_start_top_k))]
        for c in top:
            positions: list[Optional[tuple[float, float]]] = [None] * L
            scores = [0.0] * L
            positions[0] = (c.y, c.x)
            scores[0] = c.score
            seeds.append(_Hypothesis(
                positions=positions,
                scores=scores,
                accumulated_score=c.score,
                miss_count=0,
                consecutive_misses=0,
                velocity=None,
                last_observed_t=t,
                start_t=t,
                n_observed=1,
            ))
        return seeds

    def _prune(beam: list[_Hypothesis]) -> list[_Hypothesis]:
        if len(beam) <= cfg.tracklet_beam_size:
            return beam
        # keep top by accumulated_score
        beam.sort(key=lambda h: -h.accumulated_score)
        return beam[: int(cfg.tracklet_beam_size)]

    # Seed from t=0 then walk forward.
    active.extend(_seed_from_candidates(0))
    active = _prune(active)

    for t in range(1, T):
        next_active: list[_Hypothesis] = []
        cands_t = candidates_by_frame[t]

        for h in active:
            local_idx = t - h.start_t
            if local_idx >= L:
                # tracklet length reached; finalize.
                finished.append(h)
                continue

            pred = h.predict(t)
            best_extensions: list[_Hypothesis] = []

            # Option A: extend with a candidate within max_d of prediction.
            if cands_t and pred is not None:
                py, px = pred
                for c in cands_t:
                    dy = c.y - py
                    dx = c.x - px
                    d = float(np.hypot(dy, dx))
                    if d > max_d:
                        continue
                    # Build a new hypothesis = h + this candidate.
                    new_positions = list(h.positions)
                    new_scores = list(h.scores)
                    new_positions[local_idx] = (c.y, c.x)
                    new_scores[local_idx] = c.score

                    new_velocity = _estimate_velocity(new_positions, h.start_t)
                    motion_cost = _motion_cost(h.velocity, new_velocity) if new_velocity else 0.0

                    new_score = (
                        h.accumulated_score
                        + c.score
                        - cfg.tracklet_motion_penalty * motion_cost
                    )
                    new_h = _Hypothesis(
                        positions=new_positions,
                        scores=new_scores,
                        accumulated_score=new_score,
                        miss_count=h.miss_count,
                        consecutive_misses=0,
                        velocity=new_velocity if new_velocity else h.velocity,
                        last_observed_t=t,
                        start_t=h.start_t,
                        n_observed=h.n_observed + 1,
                    )
                    best_extensions.append(new_h)

            # Option B: a "miss" — extend with no detection.
            if h.consecutive_misses + 1 <= max_misses:
                new_positions = list(h.positions)
                new_scores = list(h.scores)
                miss_h = _Hypothesis(
                    positions=new_positions,
                    scores=new_scores,
                    accumulated_score=h.accumulated_score - cfg.tracklet_miss_penalty,
                    miss_count=h.miss_count + 1,
                    consecutive_misses=h.consecutive_misses + 1,
                    velocity=h.velocity,
                    last_observed_t=h.last_observed_t,
                    start_t=h.start_t,
                    n_observed=h.n_observed,
                )
                best_extensions.append(miss_h)

            next_active.extend(best_extensions)

        # Also seed new tracklets at this frame so we don't only track from t=0.
        # (Each seed is independent and will compete on the same beam.)
        next_active.extend(_seed_from_candidates(t))
        active = _prune(next_active)

    # Anything still active at the end has run its course — keep what passes
    # the minimum-observed-points threshold.
    for h in active:
        finished.append(h)

    finished = [
        h for h in finished
        if h.n_observed >= cfg.tracklet_min_observed_points
    ]

    finished.sort(key=lambda h: -h.accumulated_score)
    finished = finished[: int(cfg.tracklet_final_top_m)]

    return [_hypothesis_to_tracklet(h, cfg) for h in finished]


__all__ = ["Tracklet", "TrackletConfig", "build_tracklets"]
