"""Train SpikingTrackletUNet using the SAME curriculum as the baseline.

Reuses everything from the main project except the model architecture:
  - synthetic.sequence + data.tracklet_dataset for data generation
  - training.losses.TrackletLoss for the optimization objective
  - training.metrics.{roc_auc, pr_auc} for validation
  - scripts.train_demo.{make_video_specs, build_dataset_streaming,
    build_dataset, collate, _build_stratified_batches, snr_sweep} for the
    training and sweep glue

The only thing different is the model — SpikingTrackletUNet instead of
TrackletRecurrentUNet. The trained checkpoint and SNR sweep are saved into
``spikeNN/results/`` for the compare.py script.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

# Resolve repo root and import the main project's `src/` modules + scripts.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Main-project imports.
from synthetic.sequence import SequenceConfig  # noqa: E402
from data.tracklet_dataset import TrackletDatasetConfig  # noqa: E402
from training.losses import TrackletLoss, TrackletLossConfig  # noqa: E402
from training.metrics import roc_auc, pr_auc  # noqa: E402
from tbd.accumulator import AccumulatorConfig  # noqa: E402
from tbd.candidates import CandidateConfig  # noqa: E402
from tbd.crop_tubes import CropTubeConfig  # noqa: E402
from tbd.evidence import EvidenceConfig  # noqa: E402
from tbd.tracklets import TrackletConfig  # noqa: E402

# Reuse training-glue helpers from the existing train_demo.
import train_demo as td  # noqa: E402

# SNN model.
from spikeNN.models.spiking_recurrent_unet import (  # noqa: E402
    SpikingTrackletUNet,
    SpikingTrackletUNetConfig,
)


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RESULTS_DIR = ROOT / "spikeNN" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SNN_CHECKPOINT = RESULTS_DIR / "snn_checkpoint.pt"
SNN_HISTORY = RESULTS_DIR / "snn_history.json"
SNN_SNR_SWEEP = RESULTS_DIR / "snn_snr_sweep.json"
SNN_STAGE_CKPT = lambda stage: RESULTS_DIR / f"snn_stage{stage}_checkpoint.pt"

# Inherit the SAME hyperparameters as the baseline so any Pd difference
# reflects architecture, not curriculum.
SEED = td.SEED
N_FRAMES = td.N_FRAMES                       # 32
CROP_SIZE = td.CROP_SIZE                     # 15
CANVAS = td.CANVAS                           # (64, 64)
TARGET_SIGMA = td.TARGET_SIGMA
NOISE_SIGMA = td.NOISE_SIGMA
BATCH_SIZE = td.BATCH_SIZE                   # 8
POS_PER_BATCH = td.POS_PER_BATCH             # 2
WEIGHT_DECAY = td.WEIGHT_DECAY               # 1e-4

STAGE1 = dict(snr=td.STAGE1_SNR_RANGE_DB, n_train=td.STAGE1_N_TRAIN_VIDEOS,
              n_val=td.STAGE1_N_VAL_VIDEOS, epochs=td.STAGE1_EPOCHS,
              lr=td.STAGE1_LR, batches=td.STAGE1_BATCHES_PER_EPOCH)
STAGE2 = dict(snr=td.STAGE2_SNR_RANGE_DB, n_train=td.STAGE2_N_TRAIN_VIDEOS,
              n_val=td.STAGE2_N_VAL_VIDEOS, epochs=td.STAGE2_EPOCHS,
              lr=td.STAGE2_LR, batches=td.STAGE2_BATCHES_PER_EPOCH)
STAGE3 = dict(snr=td.STAGE3_SNR_RANGE_DB, n_train=td.STAGE3_N_TRAIN_VIDEOS,
              n_val=td.STAGE3_N_VAL_VIDEOS, epochs=td.STAGE3_EPOCHS,
              lr=td.STAGE3_LR, batches=td.STAGE3_BATCHES_PER_EPOCH,
              max_neg=td.STAGE3_MAX_NEGATIVES)

SNR_GRID = (-20.0, -15.0, -12.0, -9.0, -6.0, -3.0, 0.0, 3.0)
SNR_RUNS_PER_CELL = 20


def build_seq_cfg() -> SequenceConfig:
    return SequenceConfig(
        canvas_shape=CANVAS,
        n_frames=N_FRAMES,
        target_sigma=TARGET_SIGMA,
        noise_sigma=NOISE_SIGMA,
        motion="cv",
        speed_min_px_per_frame=0.3,
        speed_max_px_per_frame=1.5,
        boundary_margin_px=4.0,
    )


def make_ds_cfg(snr_range_db: tuple[float, float]) -> TrackletDatasetConfig:
    """Identical to train_demo.make_ds_cfg — accumulator front-end, 5 channels,
    crop 15x15. Anything the baseline saw, the SNN will see too."""
    base = build_seq_cfg()
    return TrackletDatasetConfig(
        base_config=base,
        snr_range_db=snr_range_db,
        n_choices=(N_FRAMES,),
        sequence_type_probs={
            "positive_uav": 0.5,
            "empty_background": 0.25,
            "hard_negative": 0.25,
        },
        tracklet_source="accumulator",
        evidence=EvidenceConfig(),
        candidates=CandidateConfig(candidate_top_k=100, crop_size=CROP_SIZE),
        tracklets=TrackletConfig(
            tracklet_len=N_FRAMES,
            tracklet_beam_size=128,
            tracklet_max_link_distance=4.0,
            tracklet_max_misses=3,
            tracklet_start_top_k=50,
            tracklet_min_observed_points=3,
            tracklet_final_top_m=10,
        ),
        accumulator=AccumulatorConfig(
            speed_max_px_per_frame=2.0,
            n_speeds=7,
            n_directions=12,
            accumulator_top_k=20,
            accumulator_nms_radius=3,
            accumulator_min_observed_points=3,
            crop_size=CROP_SIZE,
            bilinear=True,
        ),
        crop_tubes=CropTubeConfig(crop_size=CROP_SIZE),
        positive_radius_px=3.5,
        positive_min_visible_overlap=2,
        max_tracklets_per_sequence=20,
    )


def run_stage(model, loss_fn, optimizer, train_items, val_items, epochs,
              device, stage_name, epoch_offset, rng, batches_per_epoch):
    """Mirror of train_demo.run_stage — copied to avoid module-level state
    contamination from train_demo when imported twice in the SNN context.

    Returns (history, best_val_auc).
    """
    history: list[dict] = []
    best_state = None
    best_val_auc = -1.0

    for local_ep in range(epochs):
        epoch = epoch_offset + local_ep
        model.train()
        train_losses = []
        batches = td._build_stratified_batches(
            train_items, BATCH_SIZE, POS_PER_BATCH, batches_per_epoch, rng,
        )
        for batch_items in batches:
            if not batch_items:
                continue
            batch = td.collate(batch_items)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["crop_tube"])
            losses = loss_fn(out, batch)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(losses["loss"].item()))
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # Validate
        model.eval()
        val_losses = []
        val_scores: list[float] = []
        val_labels: list[int] = []
        with torch.no_grad():
            for i in range(0, len(val_items), BATCH_SIZE):
                batch_items = val_items[i:i + BATCH_SIZE]
                if not batch_items:
                    continue
                batch = td.collate(batch_items)
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch["crop_tube"])
                losses = loss_fn(out, batch)
                val_losses.append(float(losses["loss"].item()))
                val_scores.extend(torch.sigmoid(out["track_logit"]).cpu().numpy().tolist())
                val_labels.extend(batch["track_label"].cpu().numpy().tolist())
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        val_auc = roc_auc(val_scores, val_labels)
        val_pr = pr_auc(val_scores, val_labels)
        history.append({
            "epoch": epoch, "stage": stage_name,
            "train_loss": train_loss, "val_loss": val_loss,
            "val_roc_auc": val_auc, "val_pr_auc": val_pr,
        })
        print(f"[{stage_name:8s}] Epoch {epoch:2d} | train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_roc_auc={val_auc:.3f}  val_pr_auc={val_pr:.3f}",
              flush=True)
        if (not math.isnan(val_auc)) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"             ^ new best val_roc_auc={val_auc:.3f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[{stage_name:8s}] Restored best-val checkpoint (val_roc_auc={best_val_auc:.3f})",
              flush=True)
    return history, best_val_auc


def main():
    print(f"=== SpikingTrackletUNet training ===", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    rng = random.Random(SEED)

    device = torch.device("cpu")  # CPU benchmark was actually faster than MPS for this model
    print(f"device={device}", flush=True)

    s1_ds_cfg = make_ds_cfg(STAGE1["snr"])
    s2_ds_cfg = make_ds_cfg(STAGE2["snr"])
    s3_ds_cfg = make_ds_cfg(STAGE3["snr"])

    # --- Build curriculum datasets ----------------------------------------
    print("Generating curriculum datasets...", flush=True)
    t0 = time.time()
    s1_train_specs = td.make_video_specs(STAGE1["n_train"], seed=1001, snr_range_db=STAGE1["snr"])
    s1_val_specs = td.make_video_specs(STAGE1["n_val"], seed=2002, snr_range_db=STAGE1["snr"])
    s2_train_specs = td.make_video_specs(STAGE2["n_train"], seed=3003, snr_range_db=STAGE2["snr"])
    s2_val_specs = td.make_video_specs(STAGE2["n_val"], seed=4004, snr_range_db=STAGE2["snr"])
    s3_train_specs = td.make_video_specs(STAGE3["n_train"], seed=5005, snr_range_db=STAGE3["snr"])
    s3_val_specs = td.make_video_specs(STAGE3["n_val"], seed=6006, snr_range_db=STAGE3["snr"])

    s1_train, _ = td.build_dataset(s1_train_specs, s1_ds_cfg)
    s1_val, _ = td.build_dataset(s1_val_specs, s1_ds_cfg)
    s2_train, _ = td.build_dataset(s2_train_specs, s2_ds_cfg)
    s2_val, _ = td.build_dataset(s2_val_specs, s2_ds_cfg)
    s3_stream_rng = np.random.default_rng(SEED + 700)
    s3_train, _ = td.build_dataset_streaming(s3_train_specs, s3_ds_cfg,
                                              STAGE3["max_neg"], s3_stream_rng)
    s3_val, _ = td.build_dataset(s3_val_specs, s3_ds_cfg)

    def _stats(items):
        return f"{len(items)} tubes ({sum(int(it['track_label'].item()) for it in items)} pos)"
    print(f"  stage1 ({STAGE1['snr'][0]:+.0f}..{STAGE1['snr'][1]:+.0f} dB): train={_stats(s1_train)} val={_stats(s1_val)}", flush=True)
    print(f"  stage2 ({STAGE2['snr'][0]:+.0f}..{STAGE2['snr'][1]:+.0f} dB): train={_stats(s2_train)} val={_stats(s2_val)}", flush=True)
    print(f"  stage3 ({STAGE3['snr'][0]:+.0f}..{STAGE3['snr'][1]:+.0f} dB): train={_stats(s3_train)} val={_stats(s3_val)}", flush=True)
    print(f"  built in {time.time() - t0:.1f}s", flush=True)

    # --- Model + loss -----------------------------------------------------
    n_channels = len(s3_ds_cfg.crop_tubes.channels)
    snn_cfg = SpikingTrackletUNetConfig(
        in_channels=n_channels,
        base_channels=16,
        bottleneck_channels=32,
        crop_size=CROP_SIZE,
    )
    model = SpikingTrackletUNet(snn_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SNN model: {n_params:,} trainable parameters", flush=True)

    loss_fn = TrackletLoss(TrackletLossConfig(use_focal_loss=True))

    all_history: list[dict] = []
    stage_results: dict[str, float] = {}

    # --- Stage 1 ----------------------------------------------------------
    print(f"\n=== Stage 1 (warmup, {STAGE1['snr']}) ===", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=STAGE1["lr"], weight_decay=WEIGHT_DECAY)
    h1, best1 = run_stage(model, loss_fn, opt, s1_train, s1_val,
                          STAGE1["epochs"], device, "warmup", 0, rng, STAGE1["batches"])
    all_history.extend(h1)
    stage_results["stage1_best_val_auc"] = best1
    torch.save(model.state_dict(), SNN_STAGE_CKPT(1))

    # --- Stage 2 ----------------------------------------------------------
    print(f"\n=== Stage 2 (transition, {STAGE2['snr']}) ===", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=STAGE2["lr"], weight_decay=WEIGHT_DECAY)
    h2, best2 = run_stage(model, loss_fn, opt, s2_train, s2_val,
                          STAGE2["epochs"], device, "transit", STAGE1["epochs"], rng, STAGE2["batches"])
    all_history.extend(h2)
    stage_results["stage2_best_val_auc"] = best2
    torch.save(model.state_dict(), SNN_STAGE_CKPT(2))

    # --- Stage 3 (with rehearsal from earlier stages) --------------------
    print(f"\n=== Stage 3 (finetune, {STAGE3['snr']}) ===", flush=True)
    # Rehearsal: mix a fraction of stage1/stage2 tubes back in to prevent
    # catastrophic forgetting (same logic as the baseline).
    s3_train_with_rehearsal = list(s3_train)
    if s1_train:
        k1 = int(td.STAGE3_REHEARSAL_S1 * len(s1_train))
        s3_train_with_rehearsal.extend(random.sample(s1_train, min(k1, len(s1_train))))
    if s2_train:
        k2 = int(td.STAGE3_REHEARSAL_S2 * len(s2_train))
        s3_train_with_rehearsal.extend(random.sample(s2_train, min(k2, len(s2_train))))
    print(f"  stage3 train pool with rehearsal: {_stats(s3_train_with_rehearsal)}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=STAGE3["lr"], weight_decay=WEIGHT_DECAY)
    h3, best3 = run_stage(model, loss_fn, opt, s3_train_with_rehearsal, s3_val,
                          STAGE3["epochs"], device, "finetune",
                          STAGE1["epochs"] + STAGE2["epochs"], rng, STAGE3["batches"])
    all_history.extend(h3)
    stage_results["stage3_best_val_auc"] = best3

    # --- Save trained model + history -------------------------------------
    torch.save(model.state_dict(), SNN_CHECKPOINT)
    with open(SNN_HISTORY, "w") as f:
        json.dump({"history": all_history, "stage_results": stage_results,
                   "param_count": n_params}, f, indent=2)
    print(f"\nCheckpoint saved to {SNN_CHECKPOINT}", flush=True)
    print(f"History saved to {SNN_HISTORY}", flush=True)
    print(f"Per-stage best val ROC-AUC: stage1={best1:.3f}  stage2={best2:.3f}  stage3={best3:.3f}",
          flush=True)

    # --- SNR sweep on the SNN -------------------------------------------
    print(f"\n=== SNR sweep on SNN ({len(SNR_GRID)} cells, {SNR_RUNS_PER_CELL} runs/cell) ===",
          flush=True)
    snr_rows = td.snr_sweep(model, s3_ds_cfg, SNR_GRID,
                            runs_per_cell=SNR_RUNS_PER_CELL, device=device)
    with open(SNN_SNR_SWEEP, "w") as f:
        json.dump(snr_rows, f, indent=2)
    print(f"SNR sweep saved to {SNN_SNR_SWEEP}", flush=True)
    for r in snr_rows:
        print(f"  SNR={r['snr_db']:+5.1f} dB | Pd@FAR=1/100 = {r['pd_at_far_1_per_100']:.2f}  "
              f"(pos_mean={r['mean_pos_score']:.3f}  neg_mean={r['mean_neg_score']:.3f})",
              flush=True)


if __name__ == "__main__":
    main()
