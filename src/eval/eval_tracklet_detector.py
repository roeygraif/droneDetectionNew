"""Sequence-level evaluation for the tracklet detector.

For each synthetic sequence we run the full pipeline (evidence → candidates →
tracklets → model) and reduce to a single per-sequence detection score —
typically the max over all tracklet scores. Aggregating across sequences gives:

  - per-sequence ROC-AUC / PR-AUC
  - Pd at fixed false alarm rates (per sequence and per frame)
  - false alarm rate (per sequence and per frame)
  - localization error on detected positives
  - SNR-binned waterfall (the headline plot for the thesis)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
import yaml

from data.tracklet_dataset import (
    TrackletDatasetConfig,
    build_tube_sample,
)
from models.cnn_gru_baseline import CropCNNGRU, CropCNNGRUConfig
from models.recurrent_unet import (
    TrackletRecurrentUNet,
    TrackletRecurrentUNetConfig,
)
from synthetic.sequence import SEQUENCE_TYPES, SequenceConfig, generate_sequence
from training.metrics import pd_at_far, pr_auc, roc_auc


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_SNR_GRID = (-25.0, -23.0, -21.0, -19.0, -17.0, -15.0, -13.0, -11.0,
                     -9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0)
_DEFAULT_TARGET_FAR = (1e-2, 1e-3)


@dataclass
class EvalConfig:
    """What to evaluate and over which sweep."""

    checkpoint_path: str = "checkpoints/tracklet/epoch_0.pt"
    output_dir: str = "eval_outputs"
    device: str = "auto"

    sequence_types: tuple[str, ...] = SEQUENCE_TYPES
    snr_grid_db: tuple[float, ...] = _DEFAULT_SNR_GRID
    n_choices: tuple[int, ...] = (10,)
    runs_per_cell: int = 8

    target_far_per_sequence: tuple[float, ...] = _DEFAULT_TARGET_FAR
    target_far_per_frame: tuple[float, ...] = _DEFAULT_TARGET_FAR

    seed: int = 42

    # Pipeline config (re-use the training-time defaults; can be overridden).
    dataset: TrackletDatasetConfig = field(default_factory=TrackletDatasetConfig)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class SequenceResult:
    """One row in the per-sequence results table."""

    seed: int
    snr_db: float
    n_frames: int
    sequence_type: str
    has_target: bool
    n_tracklets: int
    best_score: float
    best_track_label: int
    best_localization_error: float


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _build_model_from_checkpoint(ckpt: dict) -> torch.nn.Module:
    name = ckpt.get("model_name", "tracklet_recurrent_unet")
    in_channels = int(ckpt.get("in_channels", 5))
    crop_size = int(ckpt.get("crop_size", 31))
    if name == "tracklet_recurrent_unet":
        return TrackletRecurrentUNet(TrackletRecurrentUNetConfig(
            in_channels=in_channels, crop_size=crop_size, use_convlstm=True,
        ))
    if name == "recurrent_unet_no_lstm":
        return TrackletRecurrentUNet(TrackletRecurrentUNetConfig(
            in_channels=in_channels, crop_size=crop_size, use_convlstm=False,
        ))
    if name == "cnn_gru":
        return CropCNNGRU(CropCNNGRUConfig(in_channels=in_channels, crop_size=crop_size))
    raise ValueError(f"Unknown model_name {name!r}")


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = _build_model_from_checkpoint(ckpt).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Sequence scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_sequence(
    sample,
    model: torch.nn.Module,
    cfg: EvalConfig,
    device: torch.device,
) -> dict:
    """Run the pipeline on one sample, return per-sequence and per-tracklet info."""
    items = build_tube_sample(sample, cfg.dataset)
    if not items:
        return {
            "n_tracklets": 0,
            "scores": np.zeros(0, dtype=np.float32),
            "labels": np.zeros(0, dtype=np.int64),
            "loc_errors": np.full(0, np.nan, dtype=np.float32),
            "best_score": -math.inf,
            "best_idx": -1,
        }

    tubes = torch.stack([it["crop_tube"] for it in items], dim=0).to(device)
    out = model(tubes)
    scores = torch.sigmoid(out["track_logit"]).cpu().numpy()
    labels = np.array([int(it["track_label"].item()) for it in items], dtype=np.int64)

    # Per-tracklet localization error: mean distance to GT on visible frames.
    loc_errors = np.full(len(items), np.nan, dtype=np.float32)
    if bool(sample.has_target):
        gt_pos = sample.positions
        target_vis = sample.target_visible
        for i, it in enumerate(items):
            centers = it["crop_centers"].numpy()
            T = centers.shape[0]
            ds = []
            for t in range(T):
                if not bool(target_vis[t]):
                    continue
                py, px = float(gt_pos[t, 0]), float(gt_pos[t, 1])
                cy, cx = float(centers[t, 0]), float(centers[t, 1])
                if any(not math.isfinite(v) for v in (py, px, cy, cx)):
                    continue
                ds.append(math.hypot(py - cy, px - cx))
            if ds:
                loc_errors[i] = float(np.mean(ds))

    best_idx = int(np.argmax(scores))
    return {
        "n_tracklets": len(items),
        "scores": scores.astype(np.float32),
        "labels": labels,
        "loc_errors": loc_errors,
        "best_score": float(scores[best_idx]),
        "best_idx": best_idx,
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _iter_cells(cfg: EvalConfig) -> Iterator[tuple[int, float, int, str, int]]:
    """Yield (cell_idx, snr_db, n_frames, seq_type, seed) tuples for the sweep."""
    rng = np.random.default_rng(cfg.seed)
    cell_idx = 0
    for snr_db in cfg.snr_grid_db:
        for n_frames in cfg.n_choices:
            for seq_type in cfg.sequence_types:
                for _ in range(cfg.runs_per_cell):
                    seed = int(rng.integers(0, 2**31 - 1))
                    yield cell_idx, float(snr_db), int(n_frames), str(seq_type), seed
                    cell_idx += 1


def evaluate(cfg: EvalConfig) -> dict:
    """Run the sweep and write per-sequence results + summary tables to disk."""
    device = torch.device(
        "cuda" if cfg.device == "cuda"
        else ("cpu" if cfg.device == "cpu"
              else ("cuda" if torch.cuda.is_available() else "cpu"))
    )
    model = load_model(cfg.checkpoint_path, device)
    os.makedirs(cfg.output_dir, exist_ok=True)

    rows: list[dict] = []
    for cell_idx, snr_db, n_frames, seq_type, seed in _iter_cells(cfg):
        seq_cfg = replace(
            cfg.dataset.base_config,
            snr_db=snr_db,
            n_frames=n_frames,
            seed=seed,
            sequence_type=seq_type,
        )
        sample = generate_sequence(seq_cfg)
        result = score_sequence(sample, model, cfg, device)

        # Sequence-level positive label = has_target (the model is asked
        # "is there a UAV in this sequence at all?").
        seq_label = int(bool(sample.has_target))
        seq_score = float(result["best_score"]) if result["n_tracklets"] > 0 else -math.inf
        best_idx = result["best_idx"]
        loc_err = (float(result["loc_errors"][best_idx])
                   if best_idx >= 0 else float("nan"))

        rows.append({
            "cell_idx": cell_idx,
            "seed": seed,
            "snr_db": snr_db,
            "n_frames": n_frames,
            "sequence_type": seq_type,
            "has_target": bool(sample.has_target),
            "n_tracklets": int(result["n_tracklets"]),
            "best_score": seq_score,
            "best_track_label_correct": int(result["labels"][best_idx]) if best_idx >= 0 else 0,
            "best_loc_err": loc_err,
            "seq_label": seq_label,
        })

    # Persist per-sequence rows.
    with open(Path(cfg.output_dir) / "per_sequence.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)

    summary = _summarize(rows, cfg)
    with open(Path(cfg.output_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return {"rows": rows, "summary": summary, "output_dir": cfg.output_dir}


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def _summarize(rows: list[dict], cfg: EvalConfig) -> dict:
    if not rows:
        return {}
    scores = np.array([r["best_score"] for r in rows], dtype=np.float64)
    labels = np.array([r["seq_label"] for r in rows], dtype=np.int64)
    # Replace -inf scores (no tracklets at all) with the min observed - 1.
    finite_mask = np.isfinite(scores)
    if finite_mask.any():
        lo = float(scores[finite_mask].min()) - 1.0
        scores = np.where(finite_mask, scores, lo)
    else:
        scores = np.zeros_like(scores)

    overall = {
        "n_sequences": int(scores.size),
        "auc_roc": roc_auc(scores, labels),
        "auc_pr": pr_auc(scores, labels),
    }

    # Pd @ FAR — pool positives and negatives.
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    far_table = []
    for far in cfg.target_far_per_sequence:
        # FAR per sequence: normalize by neg_scores.size
        res = pd_at_far(pos_scores, neg_scores, far, max(1, neg_scores.size))
        far_table.append({"target_far_per_sequence": far, **res})
    overall["pd_at_far_per_sequence"] = far_table

    # Per-frame FAR: each negative sequence contributes n_frames "trials".
    neg_idx = labels == 0
    neg_n_frames = np.array([r["n_frames"] for r in rows], dtype=np.int64)[neg_idx]
    total_neg_frames = int(neg_n_frames.sum()) if neg_n_frames.size else 0
    per_frame_table = []
    for far in cfg.target_far_per_frame:
        if total_neg_frames == 0:
            per_frame_table.append({"target_far_per_frame": far, "pd": float("nan")})
            continue
        # Treat each negative *sequence's* max score as one trial (conservative).
        # Convert per-frame FAR to per-sequence FAR by multiplying by average n_frames.
        avg_n = float(neg_n_frames.mean())
        seq_far = far * avg_n
        res = pd_at_far(pos_scores, neg_scores, seq_far, max(1, neg_scores.size))
        per_frame_table.append({"target_far_per_frame": far, **res, "avg_n_frames": avg_n})
    overall["pd_at_far_per_frame"] = per_frame_table

    # Per-SNR / sequence_type breakdown.
    by_snr = {}
    snrs = sorted({r["snr_db"] for r in rows})
    for snr in snrs:
        idx = np.array([r["snr_db"] == snr for r in rows])
        s_scores = scores[idx]
        s_labels = labels[idx]
        if (s_labels.sum() == 0) or ((s_labels == 0).sum() == 0):
            by_snr[snr] = {"auc_roc": float("nan"), "auc_pr": float("nan"), "n": int(idx.sum())}
            continue
        by_snr[snr] = {
            "auc_roc": roc_auc(s_scores, s_labels),
            "auc_pr": pr_auc(s_scores, s_labels),
            "n": int(idx.sum()),
        }
    overall["by_snr"] = by_snr

    by_type = {}
    for seq_type in sorted({r["sequence_type"] for r in rows}):
        idx = np.array([r["sequence_type"] == seq_type for r in rows])
        by_type[seq_type] = {
            "mean_score": float(scores[idx].mean()) if idx.any() else float("nan"),
            "n": int(idx.sum()),
        }
    overall["by_sequence_type"] = by_type

    # Localization error among true detections (label=1 and best tracklet matched).
    loc_pos = np.array([
        r["best_loc_err"] for r in rows
        if r["seq_label"] == 1 and math.isfinite(r["best_loc_err"])
    ], dtype=np.float64)
    overall["mean_localization_error"] = float(loc_pos.mean()) if loc_pos.size else float("nan")
    overall["median_localization_error"] = float(np.median(loc_pos)) if loc_pos.size else float("nan")

    # Minimum SNR for target Pd (use the first target FAR).
    if far_table:
        threshold = far_table[0].get("threshold", float("nan"))
        target_pd = 0.9
        snr_for_pd = float("nan")
        for snr in snrs:
            idx = np.array([(r["snr_db"] == snr and r["seq_label"] == 1) for r in rows])
            if not idx.any():
                continue
            pd = float((scores[idx] >= threshold).mean())
            if pd >= target_pd:
                snr_for_pd = snr
                break
        overall["min_snr_for_pd_0.9"] = snr_for_pd

    return overall


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    cfg = EvalConfig()
    if args.config:
        with open(args.config) as f:
            overrides = yaml.safe_load(f) or {}
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    if args.checkpoint:
        cfg.checkpoint_path = args.checkpoint
    if args.output_dir:
        cfg.output_dir = args.output_dir

    evaluate(cfg)


if __name__ == "__main__":
    main()


__all__ = ["EvalConfig", "SequenceResult", "evaluate", "load_model", "score_sequence"]
