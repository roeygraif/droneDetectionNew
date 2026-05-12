"""Tracklet crop-tube dataset.

Pipeline per item:

    SequenceConfig -> generate_sequence
                   -> compute_evidence_maps
                   -> extract_candidates
                   -> build_tracklets
                   -> extract_crop_tube  (one tube per tracklet)
                   -> label vs. ground truth UAV trajectory

Each yielded item corresponds to *one tracklet*, not one sequence. A single
synthetic sequence therefore produces multiple training items — at most one
positive tracklet (the one that overlaps the true UAV) and many negatives
(tracklets seeded on noise / clutter / distractors).

The dataset also supports a "balanced" sampling mode where it ensures that
each yielded item alternates positive / negative as long as the underlying
sequence stream provides candidates of each class. This sidesteps the
extreme negative skew that would otherwise dominate the training loss.
"""

from __future__ import annotations

import math
import os
import pickle
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

from synthetic.sequence import SequenceConfig, SequenceSample, generate_sequence

from tbd.accumulator import AccumulatorConfig, extract_seed_tracklets
from tbd.candidates import CandidateConfig, extract_candidates
from tbd.crop_tubes import CropTubeConfig, extract_crop_tube
from tbd.evidence import EvidenceConfig, compute_evidence_maps
from tbd.tracklets import Tracklet, TrackletConfig, build_tracklets


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_TYPE_MIX = {
    "positive_uav": 0.4,
    "empty_background": 0.2,
    "hard_negative": 0.2,
    "mixed_uav_and_distractors": 0.2,
}


@dataclass
class TrackletDatasetConfig:
    """Knobs for the tracklet dataset.

    Holds (a) the upstream synthetic generator schedule, (b) the TBD pipeline
    sub-configs, and (c) the label-assignment thresholds.
    """

    # --- synthetic sequence schedule -----------------------------------------
    base_config: SequenceConfig = field(default_factory=SequenceConfig)
    snr_range_db: tuple[float, float] = (-25.0, 5.0)
    n_choices: tuple[int, ...] = (10,)
    sequence_type_probs: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_TYPE_MIX))
    seed: int = 0
    n_samples: Optional[int] = None

    # --- TBD pipeline --------------------------------------------------------
    # ``tracklet_source`` picks how tracklets are produced from the evidence
    # maps. "beam_search" uses the per-frame top-K candidate extractor + the
    # motion-consistent beam search — fine at moderate SNR. "accumulator"
    # uses classical TBD: sum evidence along constant-velocity hypotheses on
    # a small grid and peak-pick the accumulated score map. The accumulator
    # is much stronger at very low SNR because it integrates the target
    # signal coherently across all T frames before any thresholding.
    tracklet_source: str = "beam_search"      # "beam_search" | "accumulator"
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    tracklets: TrackletConfig = field(default_factory=TrackletConfig)
    accumulator: AccumulatorConfig = field(default_factory=AccumulatorConfig)
    crop_tubes: CropTubeConfig = field(default_factory=CropTubeConfig)

    # --- labeling ------------------------------------------------------------
    positive_radius_px: float = 3.0
    positive_min_visible_overlap: int = 3
    heatmap_gaussian_sigma: float = 1.5

    # --- yielding strategy ---------------------------------------------------
    max_tracklets_per_sequence: int = 16
    yield_balanced: bool = True
    negatives_per_positive: int = 3

    # --- hard-negative mining cache ------------------------------------------
    hard_negative_cache_path: Optional[str] = None
    mined_negative_sampling_prob: float = 0.0


# ---------------------------------------------------------------------------
# Labeling helpers
# ---------------------------------------------------------------------------

def _gaussian_2d(size: int, cy: float, cx: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
    return g.astype(np.float32, copy=False)


def _label_tracklet(
    tracklet: Tracklet,
    sample: SequenceSample,
    crop_centers: np.ndarray,
    cfg: TrackletDatasetConfig,
) -> dict:
    """Compute per-tracklet labels from ground truth.

    Returns a dict with track_label, visibility_label (T,), heatmap_label
    (T,S,S), offset_label (T,2) and a few useful metadata fields.
    """
    T = crop_centers.shape[0]
    S = int(cfg.crop_tubes.crop_size)
    half = S // 2
    heatmap_label = np.zeros((T, S, S), dtype=np.float32)
    offset_label = np.zeros((T, 2), dtype=np.float32)
    visibility_label = np.zeros(T, dtype=np.float32)

    if not bool(sample.has_target):
        return {
            "track_label": 0.0,
            "visibility_label": visibility_label,
            "heatmap_label": heatmap_label,
            "offset_label": offset_label,
            "n_overlap": 0,
            "mean_overlap_dist": float("nan"),
        }

    gt_positions = sample.positions
    target_visible = sample.target_visible

    n_overlap = 0
    sum_dist = 0.0

    for t in range(T):
        if not bool(target_visible[t]):
            continue
        py = float(gt_positions[t, 0])
        px = float(gt_positions[t, 1])
        cy = float(crop_centers[t, 0])
        cx = float(crop_centers[t, 1])
        if not (math.isfinite(py) and math.isfinite(px) and math.isfinite(cy) and math.isfinite(cx)):
            continue
        d = math.hypot(py - cy, px - cx)
        if d <= cfg.positive_radius_px:
            n_overlap += 1
            sum_dist += d
            visibility_label[t] = 1.0

            # Heatmap label centered at the true target in crop coordinates.
            local_y = (py - cy) + half
            local_x = (px - cx) + half
            heatmap_label[t] = _gaussian_2d(S, local_y, local_x, cfg.heatmap_gaussian_sigma)
            offset_label[t, 0] = np.float32(py - cy)
            offset_label[t, 1] = np.float32(px - cx)

    is_positive = n_overlap >= int(cfg.positive_min_visible_overlap)
    if not is_positive:
        # If overlap was insufficient, reset visibility / heatmap — this is a
        # negative tracklet that happened to pass near the target sometimes.
        visibility_label[:] = 0.0
        heatmap_label[:] = 0.0
        offset_label[:] = 0.0

    return {
        "track_label": 1.0 if is_positive else 0.0,
        "visibility_label": visibility_label,
        "heatmap_label": heatmap_label,
        "offset_label": offset_label,
        "n_overlap": int(n_overlap),
        "mean_overlap_dist": float(sum_dist / max(1, n_overlap)),
    }


# ---------------------------------------------------------------------------
# Per-sequence pipeline
# ---------------------------------------------------------------------------

def build_tube_sample(
    sample: SequenceSample,
    cfg: TrackletDatasetConfig,
) -> list[dict]:
    """Run the full TBD pipeline on one ``SequenceSample`` and return one
    labeled tube-item per tracklet.

    Caller decides what to do with the list (e.g. shuffle, balance, batch).
    """
    frames_np = np.asarray(sample.frames, dtype=np.float32)
    evidence = compute_evidence_maps(frames_np, cfg.evidence)

    if cfg.tracklet_source == "accumulator":
        acc_cfg = replace(cfg.accumulator, crop_size=cfg.crop_tubes.crop_size)
        tracklets = extract_seed_tracklets(evidence["evidence"], acc_cfg)
    elif cfg.tracklet_source == "beam_search":
        cand_cfg = replace(cfg.candidates, crop_size=cfg.crop_tubes.crop_size)
        candidates = extract_candidates(evidence["evidence"], cand_cfg)
        tracklets = build_tracklets(candidates, cfg.tracklets)
    else:
        raise ValueError(f"Unknown tracklet_source {cfg.tracklet_source!r}")
    items: list[dict] = []
    for tr in tracklets[: int(cfg.max_tracklets_per_sequence)]:
        tube = extract_crop_tube(frames_np, evidence, tr, cfg.crop_tubes)
        labels = _label_tracklet(tr, sample, tube["crop_centers"].numpy(), cfg)

        item = {
            "crop_tube": tube["crop_tube"],
            "valid_mask": tube["valid_mask"],
            "crop_centers": tube["crop_centers"],
            "track_label": torch.tensor(labels["track_label"], dtype=torch.float32),
            "visibility_label": torch.from_numpy(labels["visibility_label"]),
            "heatmap_label": torch.from_numpy(labels["heatmap_label"]),
            "offset_label": torch.from_numpy(labels["offset_label"]),
            "sequence_type": sample.sequence_type,
            "snr_db_prescribed": float(sample.snr_db_prescribed),
            "snr_db_measured": float(sample.snr_db_measured),
            "effective_scnr_db_mean": float(sample.effective_scnr_db_mean),
            "tracklet_score": float(tr.score),
            "tracklet_start_t": int(tr.start_t),
            "tracklet_end_t": int(tr.end_t),
            "tracklet_n_observed": int(tr.metadata.get("n_observed", 0)),
            "has_target": bool(sample.has_target),
            "n_overlap": int(labels["n_overlap"]),
        }
        items.append(item)
    return items


# ---------------------------------------------------------------------------
# IterableDataset
# ---------------------------------------------------------------------------

def _normalize_type_probs(probs: dict[str, float]) -> tuple[tuple[str, ...], np.ndarray]:
    total = float(sum(probs.values()))
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"sequence_type_probs must sum to > 0; got {probs}")
    types = tuple(probs.keys())
    weights = np.array([probs[t] / total for t in types], dtype=np.float64)
    return types, weights


class TrackletCropDataset(IterableDataset):
    """Stream of labeled crop-tube items.

    Each iteration step generates one synthetic sequence, runs the TBD
    pipeline, and yields its tracklet items one at a time (subject to the
    balanced-sampling policy if enabled).
    """

    def __init__(self, cfg: TrackletDatasetConfig):
        super().__init__()
        self.cfg = cfg
        self._types, self._weights = _normalize_type_probs(cfg.sequence_type_probs)
        self._mined_cache: Optional[list[dict]] = None
        if cfg.hard_negative_cache_path and os.path.exists(cfg.hard_negative_cache_path):
            try:
                with open(cfg.hard_negative_cache_path, "rb") as f:
                    self._mined_cache = pickle.load(f)
            except Exception:
                self._mined_cache = None

    def _sample_one_sequence(self, rng: np.random.Generator) -> SequenceSample:
        snr_lo, snr_hi = self.cfg.snr_range_db
        snr_db = float(rng.uniform(snr_lo, snr_hi))
        n_frames = int(rng.choice(self.cfg.n_choices))
        seq_type = str(rng.choice(self._types, p=self._weights))
        seq_seed = int(rng.integers(0, 2**31 - 1))
        cfg = replace(
            self.cfg.base_config,
            snr_db=snr_db,
            n_frames=n_frames,
            seed=seq_seed,
            sequence_type=seq_type,
        )
        return generate_sequence(cfg)

    def _maybe_inject_mined(self, rng: random.Random) -> Optional[dict]:
        if not self._mined_cache:
            return None
        if rng.random() >= self.cfg.mined_negative_sampling_prob:
            return None
        return dict(rng.choice(self._mined_cache))

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rng = np.random.default_rng(np.array([self.cfg.seed, worker_id], dtype=np.uint64))
        py_rng = random.Random(int(rng.integers(0, 2**31 - 1)))

        emitted = 0
        n_target = self.cfg.n_samples
        while n_target is None or emitted < n_target:
            sample = self._sample_one_sequence(rng)
            items = build_tube_sample(sample, self.cfg)
            if not items:
                continue

            if self.cfg.yield_balanced:
                positives = [it for it in items if it["track_label"].item() >= 0.5]
                negatives = [it for it in items if it["track_label"].item() < 0.5]
                py_rng.shuffle(positives)
                py_rng.shuffle(negatives)
                queue: list[dict] = []
                for p in positives:
                    queue.append(p)
                    queue.extend(negatives[: self.cfg.negatives_per_positive])
                    del negatives[: self.cfg.negatives_per_positive]
                # If no positives, just emit a few negatives.
                if not positives:
                    queue.extend(negatives[: max(1, self.cfg.negatives_per_positive)])
            else:
                queue = list(items)
                py_rng.shuffle(queue)

            for it in queue:
                # Optionally inject a mined hard negative in place of a random
                # native negative.
                if it["track_label"].item() < 0.5:
                    inj = self._maybe_inject_mined(py_rng)
                    if inj is not None:
                        yield inj
                        emitted += 1
                        if n_target is not None and emitted >= n_target:
                            return
                        continue
                yield it
                emitted += 1
                if n_target is not None and emitted >= n_target:
                    return


__all__ = [
    "TrackletCropDataset",
    "TrackletDatasetConfig",
    "build_tube_sample",
]
