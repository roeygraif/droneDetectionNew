"""Train LIAFTrackletUNet (V3) — hybrid SNN with analog firing.

V3 vs V2:
  - Hidden layers output CONTINUOUS analog activations (sigmoid of membrane)
    instead of binary spikes. Amplitude information propagates between layers.
  - Otherwise identical: same topology, same depth, same channel widths
    (143K params), same 3-stage curriculum.

This tests the diagnosis of the V2 residual gap at −6 dB: was it intrinsic
to spiking thresholding in the hidden layers (V3 fixes this) or to something
else (V3 won't help)?
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse curriculum/data glue from V1 training script.
from spikeNN.training import train_snn as v1  # noqa: E402

from training.losses import TrackletLoss, TrackletLossConfig  # noqa: E402

import train_demo as td  # noqa: E402

from spikeNN.models.liaf_unet import (  # noqa: E402
    LIAFTrackletUNet,
    LIAFTrackletUNetConfig,
)


RESULTS_DIR = ROOT / "spikeNN" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SNN_V3_CHECKPOINT = RESULTS_DIR / "snn_v3_checkpoint.pt"
SNN_V3_HISTORY = RESULTS_DIR / "snn_v3_history.json"
SNN_V3_SNR_SWEEP = RESULTS_DIR / "snn_v3_snr_sweep.json"
SNN_V3_STAGE_CKPT = lambda stage: RESULTS_DIR / f"snn_v3_stage{stage}_checkpoint.pt"


def main():
    print("=== LIAFTrackletUNet V3 training (LIAF / analog fire) ===", flush=True)

    torch.manual_seed(v1.SEED)
    np.random.seed(v1.SEED)
    random.seed(v1.SEED)
    rng = random.Random(v1.SEED)

    device = torch.device("cpu")
    print(f"device={device}", flush=True)

    s1_ds_cfg = v1.make_ds_cfg(v1.STAGE1["snr"])
    s2_ds_cfg = v1.make_ds_cfg(v1.STAGE2["snr"])
    s3_ds_cfg = v1.make_ds_cfg(v1.STAGE3["snr"])

    print("Generating curriculum datasets...", flush=True)
    t0 = time.time()
    s1_train_specs = td.make_video_specs(v1.STAGE1["n_train"], seed=1001, snr_range_db=v1.STAGE1["snr"])
    s1_val_specs = td.make_video_specs(v1.STAGE1["n_val"], seed=2002, snr_range_db=v1.STAGE1["snr"])
    s2_train_specs = td.make_video_specs(v1.STAGE2["n_train"], seed=3003, snr_range_db=v1.STAGE2["snr"])
    s2_val_specs = td.make_video_specs(v1.STAGE2["n_val"], seed=4004, snr_range_db=v1.STAGE2["snr"])
    s3_train_specs = td.make_video_specs(v1.STAGE3["n_train"], seed=5005, snr_range_db=v1.STAGE3["snr"])
    s3_val_specs = td.make_video_specs(v1.STAGE3["n_val"], seed=6006, snr_range_db=v1.STAGE3["snr"])

    s1_train, _ = td.build_dataset(s1_train_specs, s1_ds_cfg)
    s1_val, _ = td.build_dataset(s1_val_specs, s1_ds_cfg)
    s2_train, _ = td.build_dataset(s2_train_specs, s2_ds_cfg)
    s2_val, _ = td.build_dataset(s2_val_specs, s2_ds_cfg)
    s3_stream_rng = np.random.default_rng(v1.SEED + 700)
    s3_train, _ = td.build_dataset_streaming(s3_train_specs, s3_ds_cfg,
                                              v1.STAGE3["max_neg"], s3_stream_rng)
    s3_val, _ = td.build_dataset(s3_val_specs, s3_ds_cfg)

    def _stats(items):
        return f"{len(items)} tubes ({sum(int(it['track_label'].item()) for it in items)} pos)"
    print(f"  stage1: train={_stats(s1_train)} val={_stats(s1_val)}", flush=True)
    print(f"  stage2: train={_stats(s2_train)} val={_stats(s2_val)}", flush=True)
    print(f"  stage3: train={_stats(s3_train)} val={_stats(s3_val)}", flush=True)
    print(f"  built in {time.time() - t0:.1f}s", flush=True)

    n_channels = len(s3_ds_cfg.crop_tubes.channels)
    cfg = LIAFTrackletUNetConfig(
        in_channels=n_channels,
        base_channels=24,
        bottleneck_channels=48,
        crop_size=v1.CROP_SIZE,
    )
    model = LIAFTrackletUNet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LIAF V3 model: {n_params:,} trainable parameters", flush=True)

    loss_fn = TrackletLoss(TrackletLossConfig(use_focal_loss=True))

    all_history: list[dict] = []
    stage_results: dict[str, float] = {}

    print(f"\n=== Stage 1 (warmup, {v1.STAGE1['snr']}) ===", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=v1.STAGE1["lr"], weight_decay=v1.WEIGHT_DECAY)
    h1, best1 = v1.run_stage(model, loss_fn, opt, s1_train, s1_val, v1.STAGE1["epochs"],
                              device, "warmup", 0, rng, v1.STAGE1["batches"])
    all_history.extend(h1); stage_results["stage1_best_val_auc"] = best1
    torch.save(model.state_dict(), SNN_V3_STAGE_CKPT(1))

    print(f"\n=== Stage 2 (transition, {v1.STAGE2['snr']}) ===", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=v1.STAGE2["lr"], weight_decay=v1.WEIGHT_DECAY)
    h2, best2 = v1.run_stage(model, loss_fn, opt, s2_train, s2_val, v1.STAGE2["epochs"],
                              device, "transit", v1.STAGE1["epochs"], rng, v1.STAGE2["batches"])
    all_history.extend(h2); stage_results["stage2_best_val_auc"] = best2
    torch.save(model.state_dict(), SNN_V3_STAGE_CKPT(2))

    print(f"\n=== Stage 3 (finetune, {v1.STAGE3['snr']}) ===", flush=True)
    s3_train_with_rehearsal = list(s3_train)
    if s1_train:
        k1 = int(td.STAGE3_REHEARSAL_S1 * len(s1_train))
        s3_train_with_rehearsal.extend(random.sample(s1_train, min(k1, len(s1_train))))
    if s2_train:
        k2 = int(td.STAGE3_REHEARSAL_S2 * len(s2_train))
        s3_train_with_rehearsal.extend(random.sample(s2_train, min(k2, len(s2_train))))
    print(f"  stage3 train pool with rehearsal: {_stats(s3_train_with_rehearsal)}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=v1.STAGE3["lr"], weight_decay=v1.WEIGHT_DECAY)
    h3, best3 = v1.run_stage(model, loss_fn, opt, s3_train_with_rehearsal, s3_val,
                              v1.STAGE3["epochs"], device, "finetune",
                              v1.STAGE1["epochs"] + v1.STAGE2["epochs"], rng, v1.STAGE3["batches"])
    all_history.extend(h3); stage_results["stage3_best_val_auc"] = best3

    torch.save(model.state_dict(), SNN_V3_CHECKPOINT)
    with open(SNN_V3_HISTORY, "w") as f:
        json.dump({"history": all_history, "stage_results": stage_results,
                   "param_count": n_params,
                   "config": {"base_channels": 24, "bottleneck_channels": 48,
                              "model_type": "LIAF"}}, f, indent=2)
    print(f"\nCheckpoint: {SNN_V3_CHECKPOINT}", flush=True)
    print(f"Per-stage best val ROC-AUC: stage1={best1:.3f}  stage2={best2:.3f}  stage3={best3:.3f}",
          flush=True)

    print(f"\n=== SNR sweep on V3 ({len(v1.SNR_GRID)} cells, {v1.SNR_RUNS_PER_CELL} runs/cell) ===",
          flush=True)
    snr_rows = td.snr_sweep(model, s3_ds_cfg, v1.SNR_GRID,
                            runs_per_cell=v1.SNR_RUNS_PER_CELL, device=device)
    with open(SNN_V3_SNR_SWEEP, "w") as f:
        json.dump(snr_rows, f, indent=2)
    print(f"SNR sweep saved to {SNN_V3_SNR_SWEEP}", flush=True)
    for r in snr_rows:
        print(f"  SNR={r['snr_db']:+5.1f} dB | Pd@FAR=1/100 = {r['pd_at_far_1_per_100']:.2f}  "
              f"(pos_mean={r['mean_pos_score']:.3f}  neg_mean={r['mean_neg_score']:.3f})",
              flush=True)


if __name__ == "__main__":
    main()
