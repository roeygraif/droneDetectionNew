"""Train the tracklet detector on exactly 20 synthetic videos + QA report.

Run:
    python scripts/train_demo.py

Outputs land in ``demo_outputs/``:
    learning_curve.png, score_distribution.png, example_tubes.png,
    snr_sweep.png, history.json, qa_report.json
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

import matplotlib

matplotlib.use("Agg")  # headless
# Point matplotlib at the ffmpeg binary bundled in imageio_ffmpeg so MP4 writing
# works without a system-wide ffmpeg install.
try:
    import imageio_ffmpeg  # noqa: F401
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    _HAVE_FFMPEG = True
except Exception:
    _HAVE_FFMPEG = False
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np
import torch

# Allow `python scripts/train_demo.py` from repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.tracklet_dataset import TrackletDatasetConfig, build_tube_sample  # noqa: E402
from models.recurrent_unet import (  # noqa: E402
    TrackletRecurrentUNet,
    TrackletRecurrentUNetConfig,
)
from synthetic.sequence import SequenceConfig, generate_sequence  # noqa: E402
from tbd.accumulator import AccumulatorConfig  # noqa: E402
from tbd.candidates import CandidateConfig  # noqa: E402
from tbd.crop_tubes import CropTubeConfig  # noqa: E402
from tbd.evidence import EvidenceConfig  # noqa: E402
from tbd.tracklets import TrackletConfig  # noqa: E402
from training.losses import TrackletLoss, TrackletLossConfig  # noqa: E402
from training.metrics import pd_at_far, pr_auc, roc_auc  # noqa: E402


OUTPUT_DIR = ROOT / "demo_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = OUTPUT_DIR / "model_checkpoint.pt"

# When set, skip all training and load the saved checkpoint, then go straight
# to the comparison-video rendering step. Used when iterating on video
# explanations / cases without retraining (which takes ~30 min at T=32).
RENDER_ONLY = os.environ.get("DEMO_RENDER_ONLY") == "1"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATCH_SIZE = 8
POS_PER_BATCH = 2          # stratified — guarantees positives in every batch
WEIGHT_DECAY = 1e-4
SEED = 0
# Per-stage batches/epoch. Stages 1 and 2 have small positive pools and
# converge fast; pushing them past ~100 batches/epoch starts to overfit
# (we saw stage 2's val ROC-AUC drop from 0.81 -> 0.70 when this was 200).
# Stage 3 has 1500+ positives and needs more steps to actually consume them.
STAGE1_BATCHES_PER_EPOCH = 64
STAGE2_BATCHES_PER_EPOCH = 100
STAGE3_BATCHES_PER_EPOCH = 300

# Stage 4: hard-negative mining + remediation. After stage 3 finishes we run
# the model against target-absent sequences, harvest the highest-scoring
# tubes (the model's worst false positives), oversample them into the stage-3
# training pool, and continue training at a low LR for a few short epochs.
# The whole goal is to teach the classifier to push down the spurious-coherent
# tracklets the bilinear+finer-grid front-end now produces.
MINE_HARD_NEGATIVES = True
# Gentle mining (Run H baseline):
#   - score threshold: only keep tubes the model genuinely thinks are UAVs,
#     not borderline ones (which can look classifier-confusable with marginal
#     true positives).
#   - oversample 1x: each mined tube appears once in the pool, not three times.
#   - scan many target-absent sequences to compensate for the score floor.
MINING_N_SEQUENCES = 1200
MINING_TOP_K = 300
MINING_SCORE_THRESHOLD = 0.5
MINING_OVERSAMPLE = 1
STAGE4_EPOCHS = 5
STAGE4_LR = 1e-4
STAGE4_BATCHES_PER_EPOCH = 200

# Iterative mining (Run I): a second round of gentle mining against the post-
# stage-4 model. The model is now resistant to the stage-4 mined tubes, so
# this round should find a *different*, smaller set of false positives — the
# new weakest links. Scan more sequences since the per-sequence FP rate is
# expected to be lower.
ITERATIVE_MINING = True
STAGE5_N_SEQUENCES = 2000
STAGE5_TOP_K = 300
STAGE5_SCORE_THRESHOLD = 0.5
STAGE5_OVERSAMPLE = 1
STAGE5_EPOCHS = 5
STAGE5_LR = 1e-4
STAGE5_BATCHES_PER_EPOCH = 200

# Third mining round (Run J): same recipe again. Scan even more sequences
# because each previous round has shrunk the per-sequence FP rate at >=0.5.
#
# Empirically (Run J): three rounds is too many. The third round found only
# 0.67 % per-sequence FP rate (vs round 2's 4.95 %) — mining had saturated.
# Continuing to train regressed Pd at 0 dB by 10 points without compensating
# gains elsewhere. Keep this off by default; turn on if needed for diagnostics.
STAGE6_ENABLED = False
STAGE6_N_SEQUENCES = 3000
STAGE6_TOP_K = 300
STAGE6_SCORE_THRESHOLD = 0.5
STAGE6_OVERSAMPLE = 1
STAGE6_EPOCHS = 5
STAGE6_LR = 1e-4
STAGE6_BATCHES_PER_EPOCH = 200

CANVAS = (64, 64)
# T=32 doubles the temporal window again on top of T=16. The √T integration
# gain scales sublinearly, so this adds another ~1.5 dB of effective SNR over
# T=16 (vs the ~3 dB gain T=16 had over T=8). Wall time doubles to ~30 min.
N_FRAMES = 32
CROP_SIZE = 15
TARGET_SIGMA = 1.0
NOISE_SIGMA = 1.0

# 3-stage curriculum: each stage's SNR distribution overlaps the next by a few
# dB so the model can smoothly carry features from clean to noisy regimes.
# All three stages use the same model + accumulator front-end; only the
# training-data SNR + learning rate differ.
STAGE1_SNR_RANGE_DB = (2.0, 10.0)     # clean — "what does a UAV tube look like"
STAGE1_N_TRAIN_VIDEOS = 80
STAGE1_N_VAL_VIDEOS = 25
STAGE1_EPOCHS = 6
STAGE1_LR = 2e-3

STAGE2_SNR_RANGE_DB = (-5.0, 2.0)     # transitional — clean enough to learn from
STAGE2_N_TRAIN_VIDEOS = 200
STAGE2_N_VAL_VIDEOS = 50
STAGE2_EPOCHS = 5
STAGE2_LR = 7e-4

STAGE3_SNR_RANGE_DB = (-15.0, -3.0)   # deployment target
STAGE3_N_TRAIN_VIDEOS = 4000          # 10x previous
STAGE3_N_VAL_VIDEOS = 150
STAGE3_EPOCHS = 10
STAGE3_LR = 2e-4
# With 4000 videos we'd hold ~80,000 tubes at 44 KB each = ~3.5 GB. Keep ALL
# positives, reservoir-sample stage-3 negatives down to this cap to keep peak
# RAM bounded (we only need enough negatives for the model to learn "what
# noise looks like", and 5000 is plenty).
STAGE3_MAX_NEGATIVES = 5000

# Rehearsal — fraction of earlier-stage pools mixed into stage 3.
STAGE3_REHEARSAL_S1 = 0.25
STAGE3_REHEARSAL_S2 = 0.30

# The deployed model is used for the SNR sweep + example plots — stage 3's
# best snapshot, since stage 3 is the deployment target.
SNR_RANGE_DB = STAGE3_SNR_RANGE_DB


# ---------------------------------------------------------------------------
# Build a fixed set of videos -> labeled crop tubes
# ---------------------------------------------------------------------------

def make_video_specs(
    n_videos: int,
    seed: int,
    snr_range_db: tuple[float, float],
) -> list[dict]:
    """Pick (seed, snr_db, sequence_type) for each video so the set is balanced."""
    rng = np.random.default_rng(seed)
    types = ["positive_uav", "empty_background", "hard_negative"]
    specs = []
    for i in range(n_videos):
        snr_db = float(rng.uniform(*snr_range_db))
        # Stratified by index so we always have a roughly even mix.
        seq_type = types[i % len(types)]
        specs.append({
            "video_seed": int(rng.integers(1, 2**31 - 1)),
            "snr_db": snr_db,
            "sequence_type": seq_type,
        })
    return specs


def build_dataset_streaming(
    video_specs: list[dict],
    ds_cfg: TrackletDatasetConfig,
    max_negatives: int,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    """Like build_dataset, but processes one video at a time and reservoir-
    samples negatives down to ``max_negatives``. All positives are kept.

    Peak RAM is bounded to roughly ``len(positives) + max_negatives`` tubes —
    necessary when scaling stage 3 from 400 to 4000 videos.
    """
    positives: list[dict] = []
    neg_reservoir: list[dict] = []
    diag: list[dict] = []
    n_neg_seen = 0

    for v in video_specs:
        cfg = replace(
            ds_cfg.base_config,
            snr_db=v["snr_db"],
            n_frames=N_FRAMES,
            seed=v["video_seed"],
            sequence_type=v["sequence_type"],
        )
        sample = generate_sequence(cfg)
        items = build_tube_sample(sample, ds_cfg)
        n_pos_v = 0
        for it in items:
            it["_video_seed"] = v["video_seed"]
            it["_video_sequence_type"] = v["sequence_type"]
            it["_video_snr_db"] = v["snr_db"]
            if int(it["track_label"].item()) == 1:
                positives.append(it)
                n_pos_v += 1
            else:
                n_neg_seen += 1
                if len(neg_reservoir) < max_negatives:
                    neg_reservoir.append(it)
                else:
                    j = int(rng.integers(n_neg_seen))
                    if j < max_negatives:
                        neg_reservoir[j] = it
        diag.append({
            "video_seed": v["video_seed"],
            "snr_db": v["snr_db"],
            "sequence_type": v["sequence_type"],
            "has_target": bool(sample.has_target),
            "n_tubes": len(items),
            "n_positive_tubes": n_pos_v,
        })
    return positives + neg_reservoir, diag


def build_dataset(video_specs: list[dict], ds_cfg: TrackletDatasetConfig) -> tuple[list[dict], list[dict]]:
    """Generate each video and run the TBD pipeline to get tube items.

    Returns (tube_items, per_video_diagnostics).
    """
    all_items: list[dict] = []
    diag: list[dict] = []
    for v in video_specs:
        cfg = replace(
            ds_cfg.base_config,
            snr_db=v["snr_db"],
            n_frames=N_FRAMES,
            seed=v["video_seed"],
            sequence_type=v["sequence_type"],
        )
        sample = generate_sequence(cfg)
        items = build_tube_sample(sample, ds_cfg)
        for it in items:
            it["_video_seed"] = v["video_seed"]
            it["_video_sequence_type"] = v["sequence_type"]
            it["_video_snr_db"] = v["snr_db"]
        all_items.extend(items)
        diag.append({
            "video_seed": v["video_seed"],
            "snr_db": v["snr_db"],
            "sequence_type": v["sequence_type"],
            "has_target": bool(sample.has_target),
            "n_tubes": len(items),
            "n_positive_tubes": int(sum(int(it["track_label"].item()) for it in items)),
        })
    return all_items, diag


def collate(items: list[dict]) -> dict[str, torch.Tensor]:
    keys = ["crop_tube", "valid_mask", "track_label", "visibility_label", "heatmap_label"]
    return {k: torch.stack([it[k] for it in items], dim=0) for k in keys}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _build_stratified_batches(
    items: list[dict],
    batch_size: int,
    pos_per_batch: int,
    n_batches: int,
    rng: random.Random,
) -> list[list[dict]]:
    """Make ``n_batches`` batches each with exactly ``pos_per_batch`` positives.

    Positives are sampled with replacement (since at low SNR there may be only
    a handful unique). Negatives are sampled without replacement from a freshly
    shuffled pool — this gives broad negative coverage per epoch without
    over-rehearsing any individual negative.
    """
    pos = [it for it in items if int(it["track_label"].item()) == 1]
    neg = [it for it in items if int(it["track_label"].item()) == 0]
    if not pos or not neg:
        # Fall back to plain shuffle.
        plain = list(items)
        rng.shuffle(plain)
        return [plain[i : i + batch_size] for i in range(0, len(plain), batch_size)]

    n_neg_per_batch = max(1, batch_size - pos_per_batch)
    batches: list[list[dict]] = []
    neg_pool: list[dict] = []
    for _ in range(n_batches):
        if len(neg_pool) < n_neg_per_batch:
            neg_pool = list(neg); rng.shuffle(neg_pool)
        b = [rng.choice(pos) for _ in range(pos_per_batch)]
        b.extend(neg_pool[:n_neg_per_batch]); neg_pool = neg_pool[n_neg_per_batch:]
        rng.shuffle(b)
        batches.append(b)
    return batches


def run_stage(
    model,
    loss_fn,
    optimizer,
    train_items,
    val_items,
    epochs,
    batch_size,
    device,
    stage_name,
    epoch_offset,
    rng,
    batches_per_epoch,
):
    """Run one curriculum stage. Restores model to that stage's best-val
    snapshot at the end and returns the per-epoch history (with absolute
    epoch indices = epoch_offset + local epoch)."""
    history: list[dict] = []
    best_state = None
    best_val_auc = -1.0

    for local_ep in range(epochs):
        epoch = epoch_offset + local_ep
        model.train()
        train_losses = []
        batches = _build_stratified_batches(
            train_items, batch_size, POS_PER_BATCH, batches_per_epoch, rng,
        )
        for batch_items in batches:
            if not batch_items:
                continue
            batch = collate(batch_items)
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
            for i in range(0, len(val_items), batch_size):
                batch_items = val_items[i : i + batch_size]
                if not batch_items:
                    continue
                batch = collate(batch_items)
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
            "epoch": epoch,
            "stage": stage_name,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_roc_auc": val_auc,
            "val_pr_auc": val_pr,
        })
        print(f"[{stage_name:8s}] Epoch {epoch:2d} | train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_roc_auc={val_auc:.3f}  val_pr_auc={val_pr:.3f}")
        if (not math.isnan(val_auc)) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"             ^ new best val_roc_auc={val_auc:.3f}; snapshot saved")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[{stage_name:8s}] Restored best-val checkpoint (val_roc_auc={best_val_auc:.3f})")
    return history, best_val_auc


# ---------------------------------------------------------------------------
# Post-training scoring helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_items(model, items, device, batch_size=8):
    model.eval()
    scores: list[float] = []
    for i in range(0, len(items), batch_size):
        batch_items = items[i : i + batch_size]
        if not batch_items:
            continue
        batch = collate(batch_items)
        x = batch["crop_tube"].to(device)
        out = model(x)
        scores.extend(torch.sigmoid(out["track_logit"]).cpu().numpy().tolist())
    return scores


# ---------------------------------------------------------------------------
# Hard-negative mining (inline against the in-memory model)
# ---------------------------------------------------------------------------

@torch.no_grad()
def mine_hard_negatives_inline(
    model: torch.nn.Module,
    ds_cfg: TrackletDatasetConfig,
    snr_range_db: tuple[float, float],
    n_sequences: int,
    top_k: int,
    device: torch.device,
    rng: np.random.Generator,
    score_threshold: float = 0.0,
) -> list[dict]:
    """Run ``model`` on target-absent sequences, return the highest-scoring
    tubes the model is mistakenly calling UAVs. They are by construction
    label=0 (no UAV present) and represent the worst false positives the
    current model is producing.

    Two filters in series:
      1. Keep only tubes with predicted score >= ``score_threshold``. This
         excludes borderline tubes that may be classifier-confusable with
         marginal true positives.
      2. Of the survivors, return the ``top_k`` highest-scoring.

    The mined tubes' track_label is forced to 0.0 just in case, and
    visibility_label / heatmap_label are zeroed (they wouldn't be supervised
    anyway for negatives, but keep things clean).
    """
    model.eval()
    seq_types = ("empty_background", "hard_negative")
    candidates: list[tuple[float, dict]] = []

    for i in range(int(n_sequences)):
        snr_db = float(rng.uniform(*snr_range_db))
        seq_type = str(rng.choice(seq_types))
        seed = int(rng.integers(1, 2**31 - 1))
        cfg = replace(
            ds_cfg.base_config,
            snr_db=snr_db,
            n_frames=N_FRAMES,
            seed=seed,
            sequence_type=seq_type,
        )
        sample = generate_sequence(cfg)
        items = build_tube_sample(sample, ds_cfg)
        if not items:
            continue
        # Sanity: every item from a target-absent sequence must have label 0.
        tubes = torch.stack([it["crop_tube"] for it in items], dim=0).to(device)
        out = model(tubes)
        scores = torch.sigmoid(out["track_logit"]).cpu().numpy()
        for s, it in zip(scores, items):
            s_f = float(s)
            if s_f < float(score_threshold):
                continue
            # Be defensive — force the labels to 0 regardless of what build_tube_sample produced.
            it = dict(it)
            it["track_label"] = torch.tensor(0.0, dtype=torch.float32)
            it["visibility_label"] = torch.zeros_like(it["visibility_label"])
            it["heatmap_label"] = torch.zeros_like(it["heatmap_label"])
            it["_mined"] = True
            it["_mined_score"] = s_f
            it["_mined_snr_db"] = snr_db
            it["_mined_sequence_type"] = seq_type
            candidates.append((s_f, it))

    candidates.sort(key=lambda x: -x[0])
    return [c[1] for c in candidates[: int(top_k)]]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_learning_curve(history: list[dict], path: Path) -> None:
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train", marker="o")
    axes[0].plot(ep, [h["val_loss"] for h in history], label="val", marker="s")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_title("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, [h["val_roc_auc"] for h in history], label="ROC-AUC", marker="o", color="C2")
    axes[1].plot(ep, [h["val_pr_auc"] for h in history], label="PR-AUC", marker="s", color="C3")
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="random")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("AUC"); axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Validation AUC"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Mark the curriculum stage transition.
    stages = [h.get("stage") for h in history]
    transitions = [i for i in range(1, len(stages)) if stages[i] != stages[i - 1]]
    for i in transitions:
        x = (ep[i - 1] + ep[i]) / 2.0
        for ax in axes:
            ax.axvline(x, color="black", linestyle=":", alpha=0.6)
        axes[1].text(x, 0.05, f"  -> {stages[i]}", fontsize=8, alpha=0.7)

    fig.suptitle(
        f"Tracklet detector (TBD accumulator) — curriculum: "
        f"warmup [{STAGE1_SNR_RANGE_DB[0]:.0f}..{STAGE1_SNR_RANGE_DB[1]:.0f} dB] "
        f"-> finetune [{STAGE2_SNR_RANGE_DB[0]:.0f}..{STAGE2_SNR_RANGE_DB[1]:.0f} dB]"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution(val_items: list[dict], val_scores: list[float], path: Path) -> None:
    pos = [s for it, s in zip(val_items, val_scores) if int(it["track_label"].item()) == 1]
    neg = [s for it, s in zip(val_items, val_scores) if int(it["track_label"].item()) == 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)
    if neg:
        ax.hist(neg, bins=bins, alpha=0.55, label=f"negative (n={len(neg)})", color="C3")
    if pos:
        ax.hist(pos, bins=bins, alpha=0.55, label=f"positive (n={len(pos)})", color="C2")
    ax.set_xlabel("predicted track score"); ax.set_ylabel("count")
    ax.set_title("Validation score distribution by label"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_example_tubes(val_items: list[dict], val_scores: list[float], path: Path) -> None:
    pos = sorted(
        [(s, it) for it, s in zip(val_items, val_scores) if int(it["track_label"].item()) == 1],
        key=lambda x: -x[0],
    )
    neg = sorted(
        [(s, it) for it, s in zip(val_items, val_scores) if int(it["track_label"].item()) == 0],
        key=lambda x: -x[0],
    )
    picks = pos[:3] + neg[:3]
    if not picks:
        return
    T = picks[0][1]["crop_tube"].shape[0]
    rows = len(picks)
    fig, axes = plt.subplots(rows, T, figsize=(1.4 * T, 1.7 * rows))
    if rows == 1:
        axes = axes[None, :]
    for r, (score, it) in enumerate(picks):
        tube = it["crop_tube"].numpy()  # (T, C, S, S)
        # Use the "raw" channel (channel 0 from CropTubeConfig defaults).
        label = int(it["track_label"].item())
        seq_type = it.get("sequence_type", "?")
        for t in range(T):
            ax = axes[r, t]
            ax.imshow(tube[t, 0], cmap="gray", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if t == 0:
                color = "green" if label == 1 else "red"
                ax.set_ylabel(f"y={label} s={score:.2f}\n{seq_type}", color=color, fontsize=8)
    fig.suptitle("Example crop tubes — top 3 predicted positives, top 3 worst-FP negatives", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Post-training SNR sweep
# ---------------------------------------------------------------------------

@torch.no_grad()
def snr_sweep(
    model: torch.nn.Module,
    ds_cfg: TrackletDatasetConfig,
    snr_db_grid: tuple[float, ...],
    runs_per_cell: int,
    device: torch.device,
) -> list[dict]:
    """At each SNR we run ``runs_per_cell`` positive_uav and the same number of empty_background
    sequences (disjoint seeds), score each tracklet, take the max-per-sequence score,
    and compute Pd@FAR=1/100."""
    rng = np.random.default_rng(31337)
    rows = []
    for snr_db in snr_db_grid:
        pos_scores: list[float] = []
        neg_scores: list[float] = []
        for _ in range(runs_per_cell):
            # positive
            cfg_pos = replace(
                ds_cfg.base_config, snr_db=snr_db, n_frames=N_FRAMES,
                seed=int(rng.integers(1, 2**31 - 1)), sequence_type="positive_uav",
            )
            sample_pos = generate_sequence(cfg_pos)
            items = build_tube_sample(sample_pos, ds_cfg)
            if items:
                scores = score_items(model, items, device)
                pos_scores.append(max(scores))
            else:
                pos_scores.append(0.0)

            # negative (empty)
            cfg_neg = replace(
                ds_cfg.base_config, snr_db=snr_db, n_frames=N_FRAMES,
                seed=int(rng.integers(1, 2**31 - 1)), sequence_type="empty_background",
            )
            sample_neg = generate_sequence(cfg_neg)
            items = build_tube_sample(sample_neg, ds_cfg)
            if items:
                scores = score_items(model, items, device)
                neg_scores.append(max(scores))
            else:
                neg_scores.append(0.0)

        far_at_1_per_100 = pd_at_far(pos_scores, neg_scores, target_far=1e-2, n_normalization=max(1, len(neg_scores)))
        rows.append({
            "snr_db": float(snr_db),
            "pd_at_far_1_per_100": float(far_at_1_per_100["pd"]),
            "threshold": float(far_at_1_per_100["threshold"]),
            "mean_pos_score": float(np.mean(pos_scores)) if pos_scores else float("nan"),
            "mean_neg_score": float(np.mean(neg_scores)) if neg_scores else float("nan"),
        })
    return rows


def plot_snr_sweep(rows: list[dict], path: Path) -> None:
    snrs = [r["snr_db"] for r in rows]
    pd = [r["pd_at_far_1_per_100"] for r in rows]
    pos_mean = [r["mean_pos_score"] for r in rows]
    neg_mean = [r["mean_neg_score"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(snrs, pd, marker="o", color="C2")
    axes[0].set_xlabel("SNR (dB)"); axes[0].set_ylabel("Pd @ FAR=1/100")
    axes[0].set_title("Detection probability vs SNR"); axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(snrs, pos_mean, marker="o", label="mean(score|UAV present)", color="C2")
    axes[1].plot(snrs, neg_mean, marker="s", label="mean(score|empty)", color="C3")
    axes[1].set_xlabel("SNR (dB)"); axes[1].set_ylabel("mean predicted score")
    axes[1].set_title("Score separation vs SNR"); axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# QA report
# ---------------------------------------------------------------------------

def localization_error_top_positives(
    val_items: list[dict],
    val_scores: list[float],
    top_n: int = 5,
) -> dict:
    """Mean localization error among the ``top_n`` highest-scoring positive tubes."""
    indexed = [(s, it) for it, s in zip(val_items, val_scores) if int(it["track_label"].item()) == 1]
    indexed.sort(key=lambda x: -x[0])
    indexed = indexed[:top_n]
    if not indexed:
        return {"n": 0, "mean_loc_err": float("nan"), "median_loc_err": float("nan")}

    # Per-tube loc error: distance between crop_center and the GT trajectory on visible frames.
    errs: list[float] = []
    for _, it in indexed:
        centers = it["crop_centers"].numpy()
        # We don't have GT positions stored on the item; reconstruct from sequence_type/has_target.
        # The "n_overlap" stored at label time is a proxy; better: regenerate the sample.
        seq_type = it.get("_video_sequence_type", it.get("sequence_type"))
        if seq_type not in ("positive_uav", "mixed_uav_and_distractors"):
            continue
        # Use the heatmap label peak as proxy for the GT location in crop coords.
        hm = it["heatmap_label"].numpy()
        T, S, _ = hm.shape
        d_list = []
        half = S // 2
        for t in range(T):
            if hm[t].max() <= 0:
                continue
            iy, ix = np.unravel_index(int(np.argmax(hm[t])), hm[t].shape)
            # Distance from crop center (which is at (half, half) in crop coords) to the GT peak.
            d_list.append(float(math.hypot(iy - half, ix - half)))
        if d_list:
            errs.append(float(np.mean(d_list)))
    if not errs:
        return {"n": 0, "mean_loc_err": float("nan"), "median_loc_err": float("nan")}
    return {
        "n": len(errs),
        "mean_loc_err": float(np.mean(errs)),
        "median_loc_err": float(np.median(errs)),
    }


def qa_block(
    train_items, val_items, history, val_scores, snr_rows, loc_stats, dataset_diag,
) -> dict:
    checks: list[tuple[str, bool, str]] = []

    pos_train = sum(int(it["track_label"].item()) for it in train_items)
    pos_val = sum(int(it["track_label"].item()) for it in val_items)

    checks.append(("train_has_tubes", len(train_items) > 0, f"{len(train_items)} tubes"))
    checks.append(("train_has_positives", pos_train > 0, f"{pos_train} positive tubes"))
    checks.append(("val_has_tubes", len(val_items) > 0, f"{len(val_items)} tubes"))
    checks.append(("val_has_positives", pos_val > 0, f"{pos_val} positive tubes"))

    initial_loss = history[0]["train_loss"]
    final_loss = history[-1]["train_loss"]
    checks.append(("train_loss_decreased", final_loss < initial_loss,
                   f"{initial_loss:.4f} -> {final_loss:.4f}"))

    # Deployed-model AUC = AUC of val_scores against val labels. This is the
    # AUC of whatever checkpoint we kept (best-val-AUC snapshot), which is
    # what the SNR sweep / score distribution / example tubes are computed
    # against. The per-epoch history is the training trajectory, not the
    # deployed model.
    val_labels = [int(it["track_label"].item()) for it in val_items]
    deployed_auc = roc_auc(val_scores, val_labels)
    initial_auc = history[0]["val_roc_auc"]
    best_train_auc = max((h["val_roc_auc"] for h in history
                          if not math.isnan(h["val_roc_auc"])), default=float("nan"))
    checks.append(("deployed_val_auc_above_0_7",
                   (not math.isnan(deployed_auc)) and deployed_auc > 0.7,
                   f"deployed val_roc_auc={deployed_auc:.3f}  "
                   f"(best-seen during training={best_train_auc:.3f})"))
    checks.append(("val_auc_improved_by_0_10",
                   (not math.isnan(initial_auc)) and (not math.isnan(deployed_auc))
                   and deployed_auc - initial_auc >= 0.10,
                   f"{initial_auc:.3f} -> {deployed_auc:.3f}"))

    pos_scores = [s for it, s in zip(val_items, val_scores) if int(it["track_label"].item()) == 1]
    neg_scores = [s for it, s in zip(val_items, val_scores) if int(it["track_label"].item()) == 0]
    mean_pos = float(np.mean(pos_scores)) if pos_scores else float("nan")
    mean_neg = float(np.mean(neg_scores)) if neg_scores else float("nan")
    checks.append(("mean_pos_score_above_mean_neg",
                   (not math.isnan(mean_pos)) and (not math.isnan(mean_neg))
                   and mean_pos > mean_neg,
                   f"mean(pos)={mean_pos:.3f}, mean(neg)={mean_neg:.3f}"))

    loc_ok = (loc_stats["n"] > 0
              and (not math.isnan(loc_stats["mean_loc_err"]))
              and loc_stats["mean_loc_err"] < 3.0)
    checks.append(("loc_err_below_3px_on_top_positives", loc_ok,
                   f"n={loc_stats['n']}, mean_loc_err={loc_stats['mean_loc_err']:.2f} px"))

    if len(snr_rows) >= 2:
        pds = [r["pd_at_far_1_per_100"] for r in snr_rows]
        snr_sorted = sorted(snr_rows, key=lambda r: r["snr_db"])
        pds_sorted = [r["pd_at_far_1_per_100"] for r in snr_sorted]
        increasing = pds_sorted[-1] >= pds_sorted[0]
        any_finite = any(not math.isnan(p) for p in pds)
        checks.append(("snr_sweep_pd_increases", increasing and any_finite,
                       f"low={pds_sorted[0]:.2f}, high={pds_sorted[-1]:.2f}"))
    else:
        checks.append(("snr_sweep_pd_increases", False, "not enough cells"))

    n_pass = sum(1 for _, ok, _ in checks if ok)

    print("")
    print("[QA] checks")
    print("-" * 60)
    for name, ok, detail in checks:
        marker = "OK  " if ok else "FAIL"
        print(f"  [{marker}] {name}: {detail}")
    print(f"[QA] {n_pass}/{len(checks)} checks passed")

    return {
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
        "n_pass": n_pass,
        "n_total": len(checks),
        "summary": {
            "train_items": len(train_items),
            "val_items": len(val_items),
            "positive_train_tubes": pos_train,
            "positive_val_tubes": pos_val,
            "initial_train_loss": initial_loss,
            "final_train_loss": final_loss,
            "initial_val_roc_auc": initial_auc,
            "deployed_val_roc_auc": deployed_auc,
            "best_seen_val_roc_auc": best_train_auc,
            "mean_pos_score": mean_pos,
            "mean_neg_score": mean_neg,
            "localization_error_top_positives": loc_stats,
        },
        "dataset_diag": dataset_diag,
    }


# ---------------------------------------------------------------------------
# Side-by-side comparison videos: input | model | ground truth
# ---------------------------------------------------------------------------

@torch.no_grad()
def _verdict_for_case(snr_db: float, has_target: bool, best_score: float,
                      tracklet_on_target: bool | None) -> tuple[str, str]:
    """Pick a human-readable verdict + color given the case + model result.

    Colors: ``darkgreen`` = behaved as we wanted, ``darkorange`` = hit the
    physical detection limit (expected miss / lucky hit at deep SNR),
    ``red`` = wrong in a way to worry about (FA on negative, miss in saturated
    regime, hit-but-picked-distractor in a mixed scene).
    """
    pred = best_score >= 0.5
    if has_target:
        if snr_db >= 0.0:                            # saturated, Pd ~1.0
            if pred and tracklet_on_target is not False:
                return ("CORRECT DETECTION — model finds the UAV cleanly", "darkgreen")
            if pred and tracklet_on_target is False:
                return ("WRONG TARGET — model fired on a distractor, not the UAV", "red")
            return ("UNEXPECTED MISS — should easily detect in this SNR regime", "red")
        if snr_db >= -5.0:                           # threshold, Pd ~0.6-0.7
            if pred and tracklet_on_target is not False:
                return ("CORRECT DETECTION at threshold SNR (Pd~65%)", "darkgreen")
            if pred and tracklet_on_target is False:
                return ("WRONG TARGET — high score but tracklet snapped to a distractor", "red")
            return ("EXPECTED MISS — threshold SNR, the model misses ~35% of these", "darkorange")
        if snr_db >= -8.0:                           # marginal, Pd ~0.25
            if pred:
                return ("LUCKY DETECTION — marginal SNR, only ~25% should be caught", "darkgreen")
            return ("EXPECTED MISS — marginal SNR, the model gets only ~25% here", "darkorange")
        # Beyond the integration ceiling.
        if pred:
            return ("UNEXPECTED HIT below the detection floor — likely false alarm", "darkorange")
        return ("EXPECTED MISS — below detection floor (defines our SNR limit)", "darkgreen")
    # Target-absent (empty / hard_negative)
    if pred:
        return ("FALSE ALARM — model wrongly fires on a noise / clutter tracklet", "red")
    return ("CORRECT REJECTION — no false alarm raised", "darkgreen")


def _tracklet_hits_target(best_centers: np.ndarray | None,
                          gt_positions: np.ndarray,
                          target_visible: np.ndarray,
                          radius_px: float = 3.5) -> bool | None:
    """Did the best tracklet land on the real UAV trajectory?

    Returns True/False if a comparison is possible, None when the sequence has
    no target or the model produced no tracklet.
    """
    if best_centers is None or gt_positions is None:
        return None
    T = best_centers.shape[0]
    n_close = 0
    n_compared = 0
    for t in range(T):
        if t >= len(target_visible) or not bool(target_visible[t]):
            continue
        if not (np.isfinite(best_centers[t, 0]) and np.isfinite(gt_positions[t, 0])):
            continue
        dy = best_centers[t, 0] - gt_positions[t, 0]
        dx = best_centers[t, 1] - gt_positions[t, 1]
        n_compared += 1
        if (dy * dy + dx * dx) <= (radius_px * radius_px):
            n_close += 1
    if n_compared == 0:
        return None
    # Call it "on target" if at least half the visible-frame samples are within radius.
    return (n_close / n_compared) >= 0.5


def render_one_video(
    sample,
    model: torch.nn.Module,
    ds_cfg: TrackletDatasetConfig,
    out_path: Path,
    device: torch.device,
    title_suffix: str = "",
    explanation: str = "",
) -> dict:
    """Render a 3-panel animated comparison for one synthetic sequence.

    Panels (left to right):
      1. Input frames as the model sees them.
      2. Same frames + the model's BEST tracklet (highest predicted score) and
         all other tracklet hypotheses faintly.
      3. Same frames + the ground-truth UAV trajectory (only visible frames).

    If ``explanation`` is non-empty, it's rendered as a multi-line caption
    block below the panels — the last paragraph (separated by a blank line)
    is taken as the verdict and rendered in a colored box. Verdict format:
    ``"<COLOR>::<TEXT>"`` where COLOR is a matplotlib color name.
    """
    items = build_tube_sample(sample, ds_cfg)

    # Score all tracklets, pick the best.
    best_centers = None
    best_score = float("nan")
    all_centers: list[np.ndarray] = []
    other_scores: list[float] = []
    if items:
        tubes = torch.stack([it["crop_tube"] for it in items], dim=0).to(device)
        out = model(tubes)
        scores = torch.sigmoid(out["track_logit"]).cpu().numpy()
        best_idx = int(np.argmax(scores))
        best_centers = items[best_idx]["crop_centers"].numpy()
        best_score = float(scores[best_idx])
        all_centers = [it["crop_centers"].numpy() for it in items]
        other_scores = [float(s) for s in scores]

    frames = sample.frames
    T, H, W = frames.shape
    gt_pos = sample.positions
    target_vis = sample.target_visible
    snr_db = float(sample.snr_db_prescribed)

    vmin = float(np.percentile(frames, 1))
    vmax = float(np.percentile(frames, 99))

    on_target = (_tracklet_hits_target(best_centers, gt_pos, target_vis)
                 if sample.has_target else None)
    verdict_text, verdict_color = _verdict_for_case(
        snr_db, bool(sample.has_target), best_score, on_target,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 6.0))
    fig.subplots_adjust(top=0.88, bottom=0.34, left=0.04, right=0.98, wspace=0.08)

    def update(t: int):
        for ax in axes:
            ax.clear()
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)

        # ---- Panel 0: input
        axes[0].imshow(frames[t], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[0].set_title(f"Input — SNR {snr_db:+.1f} dB\nframe {t + 1}/{T}",
                          fontsize=10)

        # ---- Panel 1: model prediction
        axes[1].imshow(frames[t], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        # Other tracklets faint.
        for centers, s in zip(all_centers, other_scores):
            if centers is best_centers:
                continue
            for tau in range(t + 1):
                if np.isfinite(centers[tau, 0]):
                    axes[1].plot(centers[tau, 1], centers[tau, 0],
                                 marker="o", color="cyan", markersize=2.5, alpha=0.25)
        # Best tracklet bold.
        if best_centers is not None:
            past_y, past_x = [], []
            for tau in range(t + 1):
                if np.isfinite(best_centers[tau, 0]):
                    past_y.append(best_centers[tau, 0]); past_x.append(best_centers[tau, 1])
            if past_y:
                axes[1].plot(past_x, past_y, "-", color="lime", linewidth=1.2, alpha=0.85)
                axes[1].plot(past_x[-1], past_y[-1], marker="o", markersize=14,
                             markerfacecolor="none", markeredgewidth=2, markeredgecolor="lime")
        score_text = "n/a" if math.isnan(best_score) else f"{best_score:.3f}"
        verdict_word = "UAV" if best_score >= 0.5 else "no UAV"
        axes[1].set_title(f"Model — best score={score_text} ({verdict_word})\n"
                          f"{len(all_centers)} tracklets considered",
                          fontsize=10)

        # ---- Panel 2: ground truth
        axes[2].imshow(frames[t], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        if sample.has_target:
            past_y, past_x = [], []
            for tau in range(t + 1):
                if bool(target_vis[tau]) and np.isfinite(gt_pos[tau, 0]):
                    past_y.append(gt_pos[tau, 0]); past_x.append(gt_pos[tau, 1])
            if past_y:
                axes[2].plot(past_x, past_y, "-", color="red", linewidth=1.0, alpha=0.65)
                axes[2].plot(past_x[-1], past_y[-1], marker="x", markersize=14,
                             markeredgewidth=2, color="red")
            vis_label = "visible" if (target_vis[t] if t < len(target_vis) else False) else "occluded"
        else:
            vis_label = "(no target)"
        axes[2].set_title(f"Ground truth — {sample.sequence_type}\n{vis_label}",
                          fontsize=10)

    fig.suptitle(title_suffix or f"Comparison @ SNR {snr_db:+.1f} dB",
                 fontsize=12, fontweight="bold", y=0.965)

    # Static caption block under the panels. We add it once; the FuncAnimation
    # update only redraws inside the axes, so this text persists across frames.
    if explanation:
        fig.text(0.5, 0.28, explanation, ha="center", va="top",
                 fontsize=10, color="black", wrap=True)
    # Colored verdict ribbon at the very bottom (constant across frames).
    score_text = "n/a" if math.isnan(best_score) else f"{best_score:.3f}"
    on_target_label = ""
    if on_target is True:
        on_target_label = "tracklet lands on the real UAV"
    elif on_target is False:
        on_target_label = "tracklet does NOT lie on the real UAV"
    ribbon = f"RESULT: {verdict_text}   |   best score = {score_text}"
    if on_target_label:
        ribbon += f"   |   {on_target_label}"
    fig.text(0.5, 0.06, ribbon, ha="center", va="center",
             fontsize=10.5, color="white", fontweight="bold",
             bbox=dict(facecolor=verdict_color, edgecolor="none",
                       boxstyle="round,pad=0.5"))

    anim = FuncAnimation(fig, update, frames=T, interval=600)
    if _HAVE_FFMPEG:
        anim.save(out_path, writer=FFMpegWriter(fps=2, codec="libx264",
                                                bitrate=1800,
                                                extra_args=["-pix_fmt", "yuv420p"]))
    else:
        # Fallback: GIF via matplotlib's pure-Python writer.
        anim.save(out_path.with_suffix(".gif"), writer=PillowWriter(fps=2))
    plt.close(fig)

    return {
        "path": str(out_path),
        "snr_db": snr_db,
        "sequence_type": sample.sequence_type,
        "has_target": bool(sample.has_target),
        "best_score": best_score,
        "n_tracklets": len(items),
        "verdict_text": verdict_text,
        "verdict_color": verdict_color,
        "tracklet_on_target": on_target,
    }


def render_comparison_videos(
    model: torch.nn.Module,
    ds_cfg: TrackletDatasetConfig,
    out_dir: Path,
    device: torch.device,
) -> list[dict]:
    """Render comparison videos across SNR + sequence types with captions.

    Each case has a multi-line explanation rendered below the panels and a
    color-coded verdict ribbon (green = good, orange = expected limit, red =
    wrong in a worrying way). The set is designed for showing a non-technical
    audience: easy positives first, then threshold/marginal, then deep-SNR
    failures, then negatives, then mixed scenes.
    """
    cases = [
        # (snr_db, sequence_type, seed, filename, short_title, what_we_test, expected)
        (+10.0, "positive_uav", 100, "01_easy_uav_snr+10.mp4",
         "Easy positive (+10 dB)",
         "A bright UAV against quiet noise. The drone amplitude is ~3x the noise "
         "level — single-frame detectable.",
         "Sanity check. We expect the model to detect this trivially."),
        (+5.0, "positive_uav", 111, "02_easy_uav_snr+5.mp4",
         "Easy positive (+5 dB)",
         "UAV ~1.8x the noise amplitude. Still clearly visible in any single frame.",
         "Should detect with a high confidence score (near 1.0)."),
        (+3.0, "positive_uav", 200, "03_strong_uav_snr+3.mp4",
         "Strong positive (+3 dB)",
         "UAV ~1.4x noise. We measured Pd ~100% at this SNR — saturated regime.",
         "Should detect cleanly. The tracklet should hug the red ground-truth line."),
        (0.0, "positive_uav", 250, "04_equal_uav_snr_0.mp4",
         "Equal-power positive (0 dB)",
         "UAV amplitude EQUALS noise. In any single frame the drone is "
         "indistinguishable from the noise floor. Only temporal integration "
         "over 32 frames recovers it.",
         "Should still detect — Pd was ~100% here. Demonstrates the integration gain."),
        (-3.0, "positive_uav", 222, "05_threshold_uav_snr-3.mp4",
         "Threshold positive (-3 dB)",
         "UAV is HALF the noise amplitude. This is the edge of where T=32 "
         "integration can help. Measured Pd ~65% — about a third of these will miss.",
         "Should usually detect, but variance is real. Tracklet may wobble around the GT."),
        (-3.0, "positive_uav", 777, "06_threshold_uav_snr-3_alt.mp4",
         "Threshold variance demo (-3 dB)",
         "Same SNR and drone profile as case 05 but a different random seed. "
         "Shown to demonstrate that at threshold, the SAME SNR sometimes hits "
         "and sometimes misses.",
         "Random outcome — illustrates the probabilistic nature of low-SNR detection."),
        (-6.0, "positive_uav", 350, "07_marginal_uav_snr-6.mp4",
         "Marginal positive (-6 dB)",
         "UAV is ~1/4 the noise amplitude. We measured Pd ~25% — only one in "
         "four runs should catch the drone.",
         "Probably MISSES. A hit here is impressive; a miss is the expected outcome."),
        (-10.0, "positive_uav", 333, "08_hard_uav_snr-10.mp4",
         "Hard positive (-10 dB)",
         "UAV is ~1/10 the noise amplitude. Past the T=32 integration ceiling.",
         "Expected to MISS. This case maps out where our system breaks."),
        (-15.0, "positive_uav", 400, "09_impossible_uav_snr-15.mp4",
         "Impossible regime (-15 dB)",
         "UAV is ~1/30 the noise amplitude. No algorithm finds this without "
         "vastly more frames or a different sensor.",
         "Expected MISS. This case defines the lower-SNR limit of our system."),
        (-3.0, "empty_background", 444, "10_empty_snr-3.mp4",
         "Empty sky (no UAV, no clutter)",
         "Pure Gaussian noise. No target. No distractors.",
         "Model must REJECT — best score should fall well below 0.5."),
        (-3.0, "hard_negative", 555, "11_hard_negative_snr-3.mp4",
         "Hard negative (distractor only)",
         "No real UAV. Just one or more moving distractors — bird-like clutter "
         "that the accumulator readily picks up as 'something is here.'",
         "Model must REJECT. After hard-negative mining (Stages 4-5), this is "
         "the regime where we improved the most."),
        (+3.0, "mixed_uav_and_distractors", 600, "12_mixed_uav_snr+3.mp4",
         "Mixed scene (UAV + distractor, +3 dB)",
         "Real UAV present alongside distractor clutter. Model must pick the "
         "RIGHT moving object out of several.",
         "Should detect the UAV. If score is high but the green tracklet is "
         "NOT on the red GT line, the model got fooled by a distractor."),
        (-3.0, "mixed_uav_and_distractors", 700, "13_mixed_uav_snr-3.mp4",
         "Mixed scene at threshold (UAV + distractor, -3 dB)",
         "Hardest realistic case: low-SNR UAV alongside louder distractors. "
         "The classifier must overcome the temptation to lock onto whichever "
         "tracklet has the loudest evidence sum.",
         "Even a high score is suspicious here — check whether the tracklet "
         "lands on the UAV (good) or on the distractor (wrong target)."),
    ]
    records: list[dict] = []
    for idx, (snr_db, seq_type, seed, fname, title, what, expected) in enumerate(cases, 1):
        cfg = replace(
            ds_cfg.base_config, snr_db=snr_db, n_frames=N_FRAMES,
            seed=seed, sequence_type=seq_type,
        )
        sample = generate_sequence(cfg)
        explanation = (
            f"Case {idx:02d}: {title}\n"
            f"What this tests: {what}\n"
            f"What we expect: {expected}"
        )
        suptitle = f"Case {idx:02d}/13 — {title}"
        rec = render_one_video(sample, model, ds_cfg, out_dir / fname, device,
                               title_suffix=suptitle, explanation=explanation)
        rec["case_idx"] = idx
        rec["short_title"] = title
        records.append(rec)
        outcome = "HIT" if (rec["has_target"] and rec["best_score"] >= 0.5) else \
                  "FA"  if (not rec["has_target"] and rec["best_score"] >= 0.5) else \
                  "MISS" if rec["has_target"] else "OK"
        print(f"  -> {fname}: score={rec['best_score']:.3f}  "
              f"{rec['n_tracklets']} tracklets  [{outcome}]  "
              f"verdict_color={rec['verdict_color']}")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device("cpu")  # explicit; we want a clean CPU sanity run
    print(f"device={device}")

    base_seq_cfg = SequenceConfig(
        canvas_shape=CANVAS,
        n_frames=N_FRAMES,
        target_sigma=TARGET_SIGMA,
        noise_sigma=NOISE_SIGMA,
        motion="cv",
        speed_min_px_per_frame=0.3,
        speed_max_px_per_frame=1.5,
        boundary_margin_px=4.0,
        # SNR / seq type overridden per video.
    )

    def make_ds_cfg(snr_range_db: tuple[float, float]) -> TrackletDatasetConfig:
        return TrackletDatasetConfig(
            base_config=base_seq_cfg,
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
                n_speeds=7,             # speeds 0, 0.33, 0.67, 1.0, 1.33, 1.67, 2.0
                n_directions=12,        # 12 uniform headings -> 1+6*12 = 73 hypotheses
                accumulator_top_k=20,
                accumulator_nms_radius=3,
                accumulator_min_observed_points=3,
                crop_size=CROP_SIZE,
                bilinear=True,          # sub-pixel sampling per frame
            ),
            crop_tubes=CropTubeConfig(crop_size=CROP_SIZE),
            positive_radius_px=3.5,
            positive_min_visible_overlap=2,
            max_tracklets_per_sequence=20,
        )

    s1_ds_cfg = make_ds_cfg(STAGE1_SNR_RANGE_DB)
    s2_ds_cfg = make_ds_cfg(STAGE2_SNR_RANGE_DB)
    s3_ds_cfg = make_ds_cfg(STAGE3_SNR_RANGE_DB)

    # -- Render-only short-circuit ----------------------------------------
    # When set, load the saved model and skip directly to video rendering.
    # Used when iterating on the comparison-video case list / explanations
    # without paying the full ~30 min training cost.
    if RENDER_ONLY:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"DEMO_RENDER_ONLY=1 set but no checkpoint at {CHECKPOINT_PATH}. "
                "Run training once (without DEMO_RENDER_ONLY) to create one."
            )
        print(f"RENDER_ONLY: loading model from {CHECKPOINT_PATH}")
        model = TrackletRecurrentUNet(TrackletRecurrentUNetConfig(
            in_channels=len(s3_ds_cfg.crop_tubes.channels),
            base_channels=16,
            bottleneck_channels=32,
            use_convlstm=True,
            crop_size=CROP_SIZE,
        )).to(device)
        state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        video_dir = OUTPUT_DIR / "videos"
        video_dir.mkdir(exist_ok=True)
        with torch.no_grad():
            render_comparison_videos(model, s3_ds_cfg, video_dir, device)
        print(f"Render-only done. Videos in {video_dir}")
        return

    # -- Build train/val tubes for all three stages ------------------------
    print("Generating curriculum datasets...")
    t0 = time.time()
    s1_train_specs = make_video_specs(STAGE1_N_TRAIN_VIDEOS, seed=1001, snr_range_db=STAGE1_SNR_RANGE_DB)
    s1_val_specs = make_video_specs(STAGE1_N_VAL_VIDEOS, seed=2002, snr_range_db=STAGE1_SNR_RANGE_DB)
    s2_train_specs = make_video_specs(STAGE2_N_TRAIN_VIDEOS, seed=3003, snr_range_db=STAGE2_SNR_RANGE_DB)
    s2_val_specs = make_video_specs(STAGE2_N_VAL_VIDEOS, seed=4004, snr_range_db=STAGE2_SNR_RANGE_DB)
    s3_train_specs = make_video_specs(STAGE3_N_TRAIN_VIDEOS, seed=5005, snr_range_db=STAGE3_SNR_RANGE_DB)
    s3_val_specs = make_video_specs(STAGE3_N_VAL_VIDEOS, seed=6006, snr_range_db=STAGE3_SNR_RANGE_DB)

    s1_train, s1_train_diag = build_dataset(s1_train_specs, s1_ds_cfg)
    s1_val, s1_val_diag = build_dataset(s1_val_specs, s1_ds_cfg)
    s2_train, s2_train_diag = build_dataset(s2_train_specs, s2_ds_cfg)
    s2_val, s2_val_diag = build_dataset(s2_val_specs, s2_ds_cfg)

    # Stage 3 is large — stream-build it with reservoir-sampled negatives.
    s3_stream_rng = np.random.default_rng(SEED + 700)
    s3_train, s3_train_diag = build_dataset_streaming(
        s3_train_specs, s3_ds_cfg, STAGE3_MAX_NEGATIVES, s3_stream_rng,
    )
    s3_val, s3_val_diag = build_dataset(s3_val_specs, s3_ds_cfg)

    def _stats(items):
        return f"{len(items)} tubes ({sum(int(it['track_label'].item()) for it in items)} pos)"

    print(f"  warmup     ({STAGE1_SNR_RANGE_DB[0]:+.0f}..{STAGE1_SNR_RANGE_DB[1]:+.0f} dB): "
          f"train={_stats(s1_train)}  val={_stats(s1_val)}")
    print(f"  transition ({STAGE2_SNR_RANGE_DB[0]:+.0f}..{STAGE2_SNR_RANGE_DB[1]:+.0f} dB): "
          f"train={_stats(s2_train)}  val={_stats(s2_val)}")
    print(f"  finetune   ({STAGE3_SNR_RANGE_DB[0]:+.0f}..{STAGE3_SNR_RANGE_DB[1]:+.0f} dB): "
          f"train={_stats(s3_train)}  val={_stats(s3_val)}")
    print(f"  built in {time.time() - t0:.1f}s")

    # -- Model + loss ------------------------------------------------------
    model = TrackletRecurrentUNet(TrackletRecurrentUNetConfig(
        in_channels=len(s3_ds_cfg.crop_tubes.channels),
        base_channels=16,
        bottleneck_channels=32,
        use_convlstm=True,
        crop_size=CROP_SIZE,
    )).to(device)
    loss_fn = TrackletLoss(TrackletLossConfig()).to(device)
    rng = random.Random(SEED)

    # -- Stage 1: warmup -------------------------------------------------
    print("")
    print(f"=== Stage 1 (warmup) — SNR {STAGE1_SNR_RANGE_DB[0]:+.0f}..{STAGE1_SNR_RANGE_DB[1]:+.0f} dB ===")
    t0 = time.time()
    optim1 = torch.optim.AdamW(model.parameters(), lr=STAGE1_LR, weight_decay=WEIGHT_DECAY)
    h1, best1 = run_stage(
        model, loss_fn, optim1, s1_train, s1_val,
        epochs=STAGE1_EPOCHS, batch_size=BATCH_SIZE, device=device,
        stage_name="warmup", epoch_offset=0, rng=rng,
        batches_per_epoch=STAGE1_BATCHES_PER_EPOCH,
    )
    print(f"Stage 1 done in {time.time() - t0:.1f}s. Best val_roc_auc={best1:.3f}")

    # -- Stage 2: transitional ------------------------------------------
    print("")
    print(f"=== Stage 2 (transition) — SNR {STAGE2_SNR_RANGE_DB[0]:+.0f}..{STAGE2_SNR_RANGE_DB[1]:+.0f} dB ===")
    t0 = time.time()
    optim2 = torch.optim.AdamW(model.parameters(), lr=STAGE2_LR, weight_decay=WEIGHT_DECAY)
    h2, best2 = run_stage(
        model, loss_fn, optim2, s2_train, s2_val,
        epochs=STAGE2_EPOCHS, batch_size=BATCH_SIZE, device=device,
        stage_name="transitn", epoch_offset=STAGE1_EPOCHS, rng=rng,
        batches_per_epoch=STAGE2_BATCHES_PER_EPOCH,
    )
    print(f"Stage 2 done in {time.time() - t0:.1f}s. Best val_roc_auc={best2:.3f}")

    # -- Stage 3: deployment-target fine-tune with rehearsal ---------------
    print("")
    print(f"=== Stage 3 (finetune) — SNR {STAGE3_SNR_RANGE_DB[0]:+.0f}..{STAGE3_SNR_RANGE_DB[1]:+.0f} dB ===")
    t0 = time.time()
    s1_rehearsal = list(s1_train); rng.shuffle(s1_rehearsal)
    s1_rehearsal = s1_rehearsal[: int(STAGE3_REHEARSAL_S1 * len(s1_train))]
    s2_rehearsal = list(s2_train); rng.shuffle(s2_rehearsal)
    s2_rehearsal = s2_rehearsal[: int(STAGE3_REHEARSAL_S2 * len(s2_train))]
    s3_train_with_rehearsal = s3_train + s2_rehearsal + s1_rehearsal
    print(f"  rehearsal: +{len(s1_rehearsal)} stage-1 tubes, +{len(s2_rehearsal)} stage-2 tubes "
          f"on top of {len(s3_train)} stage-3 tubes")

    optim3 = torch.optim.AdamW(model.parameters(), lr=STAGE3_LR, weight_decay=WEIGHT_DECAY)
    h3, best3 = run_stage(
        model, loss_fn, optim3, s3_train_with_rehearsal, s3_val,
        epochs=STAGE3_EPOCHS, batch_size=BATCH_SIZE, device=device,
        stage_name="finetune", epoch_offset=STAGE1_EPOCHS + STAGE2_EPOCHS, rng=rng,
        batches_per_epoch=STAGE3_BATCHES_PER_EPOCH,
    )
    print(f"Stage 3 done in {time.time() - t0:.1f}s. Best val_roc_auc={best3:.3f}")

    history = h1 + h2 + h3
    best4: float | None = None
    mining_record: dict | None = None

    # -- Stage 4: hard-negative mining + remediation ----------------------
    if MINE_HARD_NEGATIVES:
        print("")
        print(f"=== Stage 4 (mining + remediation) — SNR "
              f"{STAGE3_SNR_RANGE_DB[0]:+.0f}..{STAGE3_SNR_RANGE_DB[1]:+.0f} dB ===")
        t0 = time.time()
        mining_rng = np.random.default_rng(SEED + 8001)
        mined = mine_hard_negatives_inline(
            model, s3_ds_cfg, STAGE3_SNR_RANGE_DB,
            n_sequences=MINING_N_SEQUENCES, top_k=MINING_TOP_K,
            device=device, rng=mining_rng,
            score_threshold=MINING_SCORE_THRESHOLD,
        )
        scores_arr = [it["_mined_score"] for it in mined]
        print(f"  mined {len(mined)} hard-negative tubes from "
              f"{MINING_N_SEQUENCES} target-absent sequences")
        if scores_arr:
            print(f"  mined score range: [{min(scores_arr):.3f}, {max(scores_arr):.3f}]  "
                  f"median {np.median(scores_arr):.3f}")
        oversampled = mined * MINING_OVERSAMPLE
        s4_train = s3_train_with_rehearsal + oversampled
        print(f"  stage-4 train pool: {len(s4_train)} tubes "
              f"(stage-3 pool: {len(s3_train_with_rehearsal)}, "
              f"mined x{MINING_OVERSAMPLE}: {len(oversampled)})")

        optim4 = torch.optim.AdamW(model.parameters(), lr=STAGE4_LR, weight_decay=WEIGHT_DECAY)
        h4, best4 = run_stage(
            model, loss_fn, optim4, s4_train, s3_val,
            epochs=STAGE4_EPOCHS, batch_size=BATCH_SIZE, device=device,
            stage_name="mining", epoch_offset=STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS,
            rng=rng, batches_per_epoch=STAGE4_BATCHES_PER_EPOCH,
        )
        print(f"Stage 4 done in {time.time() - t0:.1f}s. Best val_roc_auc={best4:.3f}")
        history = history + h4
        mining_record = {
            "n_mined": len(mined),
            "n_sequences_scanned": MINING_N_SEQUENCES,
            "top_k": MINING_TOP_K,
            "score_threshold": MINING_SCORE_THRESHOLD,
            "oversample_factor": MINING_OVERSAMPLE,
            "mined_score_min": float(min(scores_arr)) if scores_arr else float("nan"),
            "mined_score_max": float(max(scores_arr)) if scores_arr else float("nan"),
            "mined_score_median": float(np.median(scores_arr)) if scores_arr else float("nan"),
        }

    best5: float | None = None
    mining_record_2: dict | None = None
    # -- Stage 5: iterative second mining round --------------------------
    if MINE_HARD_NEGATIVES and ITERATIVE_MINING:
        print("")
        print(f"=== Stage 5 (iterative mining round 2) — SNR "
              f"{STAGE3_SNR_RANGE_DB[0]:+.0f}..{STAGE3_SNR_RANGE_DB[1]:+.0f} dB ===")
        t0 = time.time()
        mining_rng_2 = np.random.default_rng(SEED + 8002)  # different seed than round 1
        mined_2 = mine_hard_negatives_inline(
            model, s3_ds_cfg, STAGE3_SNR_RANGE_DB,
            n_sequences=STAGE5_N_SEQUENCES, top_k=STAGE5_TOP_K,
            device=device, rng=mining_rng_2,
            score_threshold=STAGE5_SCORE_THRESHOLD,
        )
        scores_arr_2 = [it["_mined_score"] for it in mined_2]
        print(f"  mined {len(mined_2)} new hard-negative tubes from "
              f"{STAGE5_N_SEQUENCES} target-absent sequences (round 2)")
        if scores_arr_2:
            print(f"  round-2 mined score range: [{min(scores_arr_2):.3f}, {max(scores_arr_2):.3f}]  "
                  f"median {np.median(scores_arr_2):.3f}")
        # Pool = stage-3 pool + round-1 mined + round-2 mined (cumulative knowledge)
        s5_train = s3_train_with_rehearsal + mined + mined_2
        print(f"  stage-5 train pool: {len(s5_train)} tubes "
              f"({len(s3_train_with_rehearsal)} stage-3 + {len(mined)} round-1 + {len(mined_2)} round-2)")

        optim5 = torch.optim.AdamW(model.parameters(), lr=STAGE5_LR, weight_decay=WEIGHT_DECAY)
        h5, best5 = run_stage(
            model, loss_fn, optim5, s5_train, s3_val,
            epochs=STAGE5_EPOCHS, batch_size=BATCH_SIZE, device=device,
            stage_name="iter-mine",
            epoch_offset=STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS + STAGE4_EPOCHS,
            rng=rng, batches_per_epoch=STAGE5_BATCHES_PER_EPOCH,
        )
        print(f"Stage 5 done in {time.time() - t0:.1f}s. Best val_roc_auc={best5:.3f}")
        history = history + h5
        mining_record_2 = {
            "n_mined": len(mined_2),
            "n_sequences_scanned": STAGE5_N_SEQUENCES,
            "top_k": STAGE5_TOP_K,
            "score_threshold": STAGE5_SCORE_THRESHOLD,
            "oversample_factor": STAGE5_OVERSAMPLE,
            "mined_score_min": float(min(scores_arr_2)) if scores_arr_2 else float("nan"),
            "mined_score_max": float(max(scores_arr_2)) if scores_arr_2 else float("nan"),
            "mined_score_median": float(np.median(scores_arr_2)) if scores_arr_2 else float("nan"),
        }

    best6: float | None = None
    mining_record_3: dict | None = None
    # -- Stage 6: third iterative mining round ----------------------------
    if MINE_HARD_NEGATIVES and ITERATIVE_MINING and STAGE6_ENABLED:
        print("")
        print(f"=== Stage 6 (iterative mining round 3) — SNR "
              f"{STAGE3_SNR_RANGE_DB[0]:+.0f}..{STAGE3_SNR_RANGE_DB[1]:+.0f} dB ===")
        t0 = time.time()
        mining_rng_3 = np.random.default_rng(SEED + 8003)
        mined_3 = mine_hard_negatives_inline(
            model, s3_ds_cfg, STAGE3_SNR_RANGE_DB,
            n_sequences=STAGE6_N_SEQUENCES, top_k=STAGE6_TOP_K,
            device=device, rng=mining_rng_3,
            score_threshold=STAGE6_SCORE_THRESHOLD,
        )
        scores_arr_3 = [it["_mined_score"] for it in mined_3]
        print(f"  mined {len(mined_3)} new hard-negative tubes from "
              f"{STAGE6_N_SEQUENCES} target-absent sequences (round 3)")
        if scores_arr_3:
            print(f"  round-3 mined score range: [{min(scores_arr_3):.3f}, {max(scores_arr_3):.3f}]  "
                  f"median {np.median(scores_arr_3):.3f}")
        # Pool = stage-3 pool + all three rounds of mined tubes
        s6_train = s3_train_with_rehearsal + mined + mined_2 + mined_3
        print(f"  stage-6 train pool: {len(s6_train)} tubes "
              f"({len(s3_train_with_rehearsal)} stage-3 "
              f"+ {len(mined)} round-1 + {len(mined_2)} round-2 + {len(mined_3)} round-3)")

        optim6 = torch.optim.AdamW(model.parameters(), lr=STAGE6_LR, weight_decay=WEIGHT_DECAY)
        h6, best6 = run_stage(
            model, loss_fn, optim6, s6_train, s3_val,
            epochs=STAGE6_EPOCHS, batch_size=BATCH_SIZE, device=device,
            stage_name="iter-3",
            epoch_offset=STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS
                          + STAGE4_EPOCHS + STAGE5_EPOCHS,
            rng=rng, batches_per_epoch=STAGE6_BATCHES_PER_EPOCH,
        )
        print(f"Stage 6 done in {time.time() - t0:.1f}s. Best val_roc_auc={best6:.3f}")
        history = history + h6
        mining_record_3 = {
            "n_mined": len(mined_3),
            "n_sequences_scanned": STAGE6_N_SEQUENCES,
            "top_k": STAGE6_TOP_K,
            "score_threshold": STAGE6_SCORE_THRESHOLD,
            "oversample_factor": STAGE6_OVERSAMPLE,
            "mined_score_min": float(min(scores_arr_3)) if scores_arr_3 else float("nan"),
            "mined_score_max": float(max(scores_arr_3)) if scores_arr_3 else float("nan"),
            "mined_score_median": float(np.median(scores_arr_3)) if scores_arr_3 else float("nan"),
        }
    # The deployed model is now Stage 3's best snapshot; downstream evaluation
    # is on the low-SNR (Stage 3) val set since that's the deployment target.
    val_items = s3_val
    val_diag = {"warmup_train": s1_train_diag, "warmup_val": s1_val_diag,
                "transition_train": s2_train_diag, "transition_val": s2_val_diag,
                "finetune_train": s3_train_diag, "finetune_val": s3_val_diag}
    train_items = s3_train

    # -- Plots --------------------------------------------------------------
    plot_learning_curve(history, OUTPUT_DIR / "learning_curve.png")

    val_scores = score_items(model, val_items, device)
    plot_score_distribution(val_items, val_scores, OUTPUT_DIR / "score_distribution.png")
    plot_example_tubes(val_items, val_scores, OUTPUT_DIR / "example_tubes.png")

    # -- Post-training SNR sweep -------------------------------------------
    snr_grid = (-20.0, -15.0, -12.0, -9.0, -6.0, -3.0, 0.0, 3.0)
    snr_rows = snr_sweep(model, s3_ds_cfg, snr_grid, runs_per_cell=20, device=device)
    plot_snr_sweep(snr_rows, OUTPUT_DIR / "snr_sweep.png")

    # -- Save model checkpoint --------------------------------------------
    # Future video-rendering iterations can reuse this via DEMO_RENDER_ONLY=1.
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved model checkpoint to {CHECKPOINT_PATH}")

    # -- Comparison videos -------------------------------------------------
    video_dir = OUTPUT_DIR / "videos"
    video_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        video_records = render_comparison_videos(model, s3_ds_cfg, video_dir, device)

    # -- QA ---------------------------------------------------------------
    loc_stats = localization_error_top_positives(val_items, val_scores, top_n=5)
    report = qa_block(train_items, val_items, history, val_scores, snr_rows, loc_stats,
                      dataset_diag=val_diag)
    report["stage1_best_val_roc_auc"] = best1
    report["stage2_best_val_roc_auc"] = best2
    report["stage3_best_val_roc_auc"] = best3
    if best4 is not None:
        report["stage4_best_val_roc_auc"] = best4
    if mining_record is not None:
        report["mining"] = mining_record
    if best5 is not None:
        report["stage5_best_val_roc_auc"] = best5
    if mining_record_2 is not None:
        report["mining_round_2"] = mining_record_2
    if best6 is not None:
        report["stage6_best_val_roc_auc"] = best6
    if mining_record_3 is not None:
        report["mining_round_3"] = mining_record_3
    report["videos"] = video_records

    with open(OUTPUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(OUTPUT_DIR / "qa_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(OUTPUT_DIR / "snr_sweep.json", "w") as f:
        json.dump(snr_rows, f, indent=2)

    print("")
    print(f"Outputs in: {OUTPUT_DIR}")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
