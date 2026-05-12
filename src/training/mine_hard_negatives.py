"""Mine hard-negative crop tubes from target-absent sequences.

Procedure:
  1. Load a trained model.
  2. Generate ``empty_background`` and ``hard_negative`` sequences.
  3. Run the TBD pipeline on each, score each tracklet.
  4. Keep the top ``mined_negatives_per_epoch`` tubes by score (these are the
     false positives the model is most fooled by).
  5. Pickle them to ``hard_negative_cache_path``.

The training dataset reads this cache and injects mined tubes at probability
``mined_negative_sampling_prob`` in place of random negatives.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pickle
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import yaml

from data.tracklet_dataset import (
    TrackletDatasetConfig,
    build_tube_sample,
)
from eval.eval_tracklet_detector import load_model
from synthetic.sequence import SequenceConfig, generate_sequence


@dataclass
class MiningConfig:
    """Knobs for the hard-negative miner."""

    checkpoint_path: str = "checkpoints/tracklet/epoch_0.pt"
    output_cache_path: str = "hard_negatives.pkl"
    device: str = "auto"

    n_sequences: int = 200
    mined_negatives_per_epoch: int = 256
    sequence_types: tuple[str, ...] = ("empty_background", "hard_negative")
    n_choices: tuple[int, ...] = (10,)
    snr_range_db: tuple[float, float] = (-25.0, 5.0)
    seed: int = 1234

    dataset: TrackletDatasetConfig = field(default_factory=TrackletDatasetConfig)


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _strip_for_cache(item: dict) -> dict:
    """Convert torch tensors to a serializable form for pickling. Drops the
    sequence-level metadata that isn't needed at train time."""
    out = {}
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = v
    out["mined"] = True
    return out


@torch.no_grad()
def mine(cfg: MiningConfig) -> dict:
    device = _resolve_device(cfg.device)
    model = load_model(cfg.checkpoint_path, device)
    rng = np.random.default_rng(cfg.seed)

    scored: list[tuple[float, dict]] = []

    for i in range(cfg.n_sequences):
        snr_db = float(rng.uniform(*cfg.snr_range_db))
        n_frames = int(rng.choice(cfg.n_choices))
        seq_type = str(rng.choice(cfg.sequence_types))
        seq_seed = int(rng.integers(0, 2**31 - 1))
        seq_cfg = replace(
            cfg.dataset.base_config,
            snr_db=snr_db, n_frames=n_frames, seed=seq_seed,
            sequence_type=seq_type,
        )
        sample = generate_sequence(seq_cfg)
        if sample.has_target:
            # By contract we should only mine from target-absent sequences;
            # skip if config accidentally enabled a target type.
            continue
        items = build_tube_sample(sample, cfg.dataset)
        if not items:
            continue
        tubes = torch.stack([it["crop_tube"] for it in items], dim=0).to(device)
        out = model(tubes)
        scores = torch.sigmoid(out["track_logit"]).cpu().numpy()
        for s, it in zip(scores, items):
            # By construction every item here is label=0; we want the highest-
            # scoring ones (the ones the model thinks ARE UAVs).
            scored.append((float(s), _strip_for_cache(it)))

    scored.sort(key=lambda t: -t[0])
    top = [it for _, it in scored[: int(cfg.mined_negatives_per_epoch)]]

    os.makedirs(os.path.dirname(cfg.output_cache_path) or ".", exist_ok=True)
    with open(cfg.output_cache_path, "wb") as f:
        pickle.dump(top, f)

    return {"n_mined": len(top), "cache_path": cfg.output_cache_path,
            "score_range": (
                float(scored[0][0]) if scored else float("nan"),
                float(scored[-1][0]) if scored else float("nan"),
            )}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    cfg = MiningConfig()
    if args.config:
        with open(args.config) as f:
            overrides = yaml.safe_load(f) or {}
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    if args.checkpoint:
        cfg.checkpoint_path = args.checkpoint
    if args.output:
        cfg.output_cache_path = args.output
    result = mine(cfg)
    print(f"Mined {result['n_mined']} hard negatives -> {result['cache_path']}")


if __name__ == "__main__":
    main()


__all__ = ["MiningConfig", "mine"]
