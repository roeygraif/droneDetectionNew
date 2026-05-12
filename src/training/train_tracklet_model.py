"""Training entry-point for the tracklet-guided UAV detector.

Usage:
    python -m training.train_tracklet_model --config configs/tracklet_default.yaml

The script is intentionally compact: one config file, one model (selected by
name), one balanced iterable dataset, and a few metrics on a held-out val
stream.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.tracklet_dataset import TrackletCropDataset, TrackletDatasetConfig
from models.cnn_gru_baseline import CropCNNGRU, CropCNNGRUConfig
from models.recurrent_unet import TrackletRecurrentUNet, TrackletRecurrentUNetConfig
from synthetic.sequence import SequenceConfig
from tbd.candidates import CandidateConfig
from tbd.crop_tubes import CropTubeConfig
from tbd.evidence import EvidenceConfig
from tbd.tracklets import TrackletConfig
from training.losses import TrackletLoss, TrackletLossConfig
from training.metrics import pr_auc, precision_recall_fpr, roc_auc


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Top-level training config."""

    model_name: str = "tracklet_recurrent_unet"  # or "cnn_gru" or "recurrent_unet_no_lstm"
    batch_size: int = 8
    train_samples_per_epoch: int = 256
    val_samples: int = 128
    epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0
    num_workers: int = 0
    device: str = "auto"  # "cpu" | "cuda" | "auto"
    checkpoint_dir: str = "checkpoints/tracklet"
    log_every: int = 10
    seed: int = 0

    dataset: TrackletDatasetConfig = field(default_factory=TrackletDatasetConfig)
    loss: TrackletLossConfig = field(default_factory=TrackletLossConfig)


# ---------------------------------------------------------------------------
# YAML -> dataclass helpers
# ---------------------------------------------------------------------------

def _apply_overrides(obj, overrides: dict) -> object:
    """Apply a nested dict of overrides onto a dataclass (in place semantics)."""
    if not isinstance(overrides, dict):
        return overrides
    for k, v in overrides.items():
        if not hasattr(obj, k):
            raise KeyError(f"Unknown field {k!r} on {type(obj).__name__}")
        current = getattr(obj, k)
        if dataclasses.is_dataclass(current) and isinstance(v, dict):
            _apply_overrides(current, v)
        else:
            if k in ("canvas_shape", "distractor_types", "n_choices", "snr_range_db", "channels"):
                if v is not None:
                    v = tuple(v) if k != "n_choices" else tuple(int(x) for x in v)
            setattr(obj, k, v)
    return obj


def load_train_config(path: str) -> TrainConfig:
    cfg = TrainConfig()
    if path:
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        _apply_overrides(cfg, overrides)
    return cfg


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_items(items: list[dict]) -> dict:
    """Stack tensors; pass through scalars as 1-D tensors."""
    if not items:
        raise ValueError("empty batch")
    keys = items[0].keys()
    out = {}
    for k in keys:
        v = items[0][k]
        if isinstance(v, torch.Tensor):
            out[k] = torch.stack([it[k] for it in items], dim=0)
        elif isinstance(v, (int, float, bool)):
            dtype = torch.float32 if isinstance(v, float) else torch.long
            if isinstance(v, bool):
                dtype = torch.bool
            out[k] = torch.tensor([it[k] for it in items], dtype=dtype)
        else:
            out[k] = [it[k] for it in items]
    return out


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(name: str, in_channels: int, crop_size: int) -> torch.nn.Module:
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


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _iter_batches(dataset: TrackletCropDataset, batch_size: int, max_batches: int) -> Iterable[dict]:
    it = iter(dataset)
    batch: list[dict] = []
    n = 0
    for item in it:
        batch.append(item)
        if len(batch) == batch_size:
            yield collate_items(batch)
            batch = []
            n += 1
            if n >= max_batches:
                return
    if batch:
        yield collate_items(batch)


def _epoch(
    model: torch.nn.Module,
    loss_fn: TrackletLoss,
    dataset: TrackletCropDataset,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    batch_size: int,
    max_batches: int,
    grad_clip: float | None,
    log_every: int,
    tag: str,
) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    all_scores: list[float] = []
    all_labels: list[int] = []
    loss_acc = 0.0
    n_batches = 0
    pos_count = 0
    total_count = 0

    for batch_idx, batch in enumerate(_iter_batches(dataset, batch_size, max_batches)):
        batch_t = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                   for k, v in batch.items()}
        x = batch_t["crop_tube"]
        with torch.set_grad_enabled(is_train):
            outputs = model(x)
            losses = loss_fn(outputs, batch_t)
            loss = losses["loss"]
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        loss_acc += float(loss.item())
        n_batches += 1

        scores = torch.sigmoid(outputs["track_logit"]).detach().cpu().numpy()
        labels = batch_t["track_label"].detach().cpu().numpy().astype(np.int64)
        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())
        pos_count += int(labels.sum())
        total_count += int(labels.size)

        if log_every > 0 and (batch_idx + 1) % log_every == 0:
            print(f"  [{tag}] batch {batch_idx + 1}: loss={loss.item():.4f} "
                  f"(track={float(losses['loss_track']):.4f} "
                  f"hm={float(losses['loss_heatmap']):.4f} "
                  f"vis={float(losses['loss_visibility']):.4f})")

    metrics = {
        "loss": loss_acc / max(1, n_batches),
        "auc_roc": roc_auc(all_scores, all_labels),
        "auc_pr": pr_auc(all_scores, all_labels),
        "pos_ratio": pos_count / max(1, total_count),
        "n_items": total_count,
    }
    pr = precision_recall_fpr(all_scores, all_labels, threshold=0.5)
    metrics.update({f"{k}@0.5": v for k, v in pr.items()})
    return metrics


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = _resolve_device(cfg.device)

    # Train / val dataset use different seeds so they don't share sequences.
    train_ds_cfg = dataclasses.replace(cfg.dataset, seed=cfg.seed)
    val_ds_cfg = dataclasses.replace(cfg.dataset, seed=cfg.seed + 9999)
    train_ds = TrackletCropDataset(train_ds_cfg)
    val_ds = TrackletCropDataset(val_ds_cfg)

    in_channels = len(cfg.dataset.crop_tubes.channels)
    model = build_model(cfg.model_name, in_channels, cfg.dataset.crop_tubes.crop_size).to(device)
    loss_fn = TrackletLoss(cfg.loss).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    history: list[dict] = []

    max_train_batches = max(1, cfg.train_samples_per_epoch // cfg.batch_size)
    max_val_batches = max(1, cfg.val_samples // cfg.batch_size)

    for epoch in range(cfg.epochs):
        t0 = time.time()
        tr_metrics = _epoch(model, loss_fn, train_ds, optim, device,
                            cfg.batch_size, max_train_batches,
                            cfg.grad_clip, cfg.log_every, f"train e{epoch}")
        va_metrics = _epoch(model, loss_fn, val_ds, None, device,
                            cfg.batch_size, max_val_batches,
                            None, 0, f"val e{epoch}")
        dt = time.time() - t0

        print(f"Epoch {epoch}: train_loss={tr_metrics['loss']:.4f} "
              f"val_loss={va_metrics['loss']:.4f} "
              f"val_auc={va_metrics['auc_roc']:.3f} "
              f"val_pr_auc={va_metrics['auc_pr']:.3f} "
              f"({dt:.1f}s)")

        history.append({"epoch": epoch, "train": tr_metrics, "val": va_metrics, "time_sec": dt})

        ckpt_path = Path(cfg.checkpoint_dir) / f"epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model_name": cfg.model_name,
            "in_channels": in_channels,
            "crop_size": cfg.dataset.crop_tubes.crop_size,
            "config": dataclasses.asdict(cfg),
            "val_metrics": va_metrics,
        }, ckpt_path)

    with open(Path(cfg.checkpoint_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    return {"history": history, "checkpoint_dir": cfg.checkpoint_dir}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--model", type=str, default=None,
                   help="Override model_name from config")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    cfg = load_train_config(args.config) if args.config else TrainConfig()
    if args.model:
        cfg.model_name = args.model
    if args.epochs:
        cfg.epochs = args.epochs
    if args.device:
        cfg.device = args.device

    train(cfg)


if __name__ == "__main__":
    main()


__all__ = ["TrainConfig", "build_model", "collate_items", "load_train_config", "train"]
