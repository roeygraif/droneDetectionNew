"""Train the in-scope detector on collapseprobe oracle-centered IR3D tubes.

This is the frozen network for the Q1 per-stage probing. We deliberately train on
the SAME oracle-centered crop tubes we will probe (charter Sec. 7: study
recognition given the rough location, to isolate the network from the front-end),
on the realistic IR3D noise where the whitening MF gives a real optimum gap
(Exp 04). The objective is plain binary detection (BCE on the track logit) — that
is exactly the "target present?" signal the per-stage probe measures; we do not
tune the architecture (charter: out of scope).

Train/val pools share the eval cells' sensor (same fixed pattern) but use disjoint
data seeds, so no sequence leaks between train, val, and the probing cells.

Run:  python -m collapseprobe.train_detector
Out:  collapseprobe/detector_ckpt.pt  (+ printed detector-vs-optimum gap per SNR)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collapseprobe.dataset import ProbeDataConfig, _base_seq_cfg, build_split_for_snr  # noqa: E402
from collapseprobe.ir_noise import make_fixed_pattern  # noqa: E402
from collapseprobe.probing import detectability  # noqa: E402
from models.recurrent_unet import TrackletRecurrentUNet, TrackletRecurrentUNetConfig  # noqa: E402

CKPT = ROOT / "collapseprobe" / "detector_ckpt.pt"
TRAIN_SNRS = (-3.0, -6.0, -9.0, -12.0, -15.0)
N_TRAIN_PER_CLASS = 160
N_VAL_PER_CLASS = 80
EPOCHS = 30
BATCH = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 31337                 # master seed of the eval cells (for the shared sensor)
TRAIN_DATA_SEED = 70001      # disjoint from eval rng -> different sequences
VAL_DATA_SEED = 90002


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_pool(cfg: ProbeDataConfig, data_seed: int, n_per_class: int, fixed):
    """Generate a labelled oracle-tube pool across TRAIN_SNRS. Returns tensors."""
    cfg = ProbeDataConfig(noise_model="ir3d", n_per_class=n_per_class, ir_noise=cfg.ir_noise,
                          scintillation=cfg.scintillation, seed=cfg.seed)
    rng = np.random.default_rng(data_seed)
    X, y, snr, wmf = [], [], [], []
    for s in TRAIN_SNRS:
        cell = build_split_for_snr(cfg, float(s), rng, fixed_pattern=fixed)
        X.append(cell["tubes"]); y.append(cell["labels"])
        wmf.append(cell["wmf_scores"]); snr.append(np.full(cell["labels"].shape, s))
    X = torch.from_numpy(np.concatenate(X)).float()          # (N,T,C,S,S)
    y = torch.from_numpy(np.concatenate(y)).float()          # (N,)
    return X, y, np.concatenate(snr), np.concatenate(wmf)


@torch.no_grad()
def score_all(model, X, device, batch=64, as_logit=False):
    """Per-tube detector scores. AUC/Pd are rank-based (invariant to the sigmoid);
    use ``as_logit=True`` for d′, since the sigmoid saturates and makes d′
    meaningless on the bounded [0,1] scores."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(device)
        lg = model(xb)["track_logit"]
        out.append((lg if as_logit else torch.sigmoid(lg)).cpu().numpy())
    return np.concatenate(out)


def main():
    t0 = time.time()
    device = _device()
    torch.manual_seed(SEED); np.random.seed(SEED)
    base = ProbeDataConfig(noise_model="ir3d", seed=SEED)
    # Same sensor (fixed pattern) as the eval cells: build_probe_dataset used seed+777.
    fixed = make_fixed_pattern(tuple(base.canvas_shape), base.ir_noise,
                               np.random.default_rng(base.seed + 777))

    print(f"[data] generating train/val pools on {device} ...", flush=True)
    Xtr, ytr, _, _ = build_pool(base, TRAIN_DATA_SEED, N_TRAIN_PER_CLASS, fixed)
    Xva, yva, snr_va, wmf_va = build_pool(base, VAL_DATA_SEED, N_VAL_PER_CLASS, fixed)
    print(f"[data] train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}  ({time.time()-t0:.0f}s)", flush=True)

    model = TrackletRecurrentUNet(TrackletRecurrentUNetConfig(
        in_channels=Xtr.shape[2], base_channels=16, bottleneck_channels=32,
        use_convlstm=True, crop_size=Xtr.shape[-1])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = nn.BCEWithLogitsLoss()

    n = len(Xtr)
    best_auc, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad(set_to_none=True)
            logit = model(xb)["track_logit"]
            loss = lossf(logit, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.detach()) * len(idx)
        # validation AUC for best-checkpointing
        sv = score_all(model, Xva, device)
        va = detectability(sv[yva.numpy() == 1], sv[yva.numpy() == 0])
        if va["auc"] > best_auc:
            best_auc = va["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {ep:2d} | train_loss {ep_loss/n:.4f} | val AUC {va['auc']:.3f} "
              f"d' {va['dprime']:.2f} | best {best_auc:.3f}", flush=True)

    model.load_state_dict(best_state)
    torch.save({"model_state": best_state, "in_channels": int(Xtr.shape[2]),
                "base_channels": 16, "bottleneck_channels": 32, "crop_size": int(Xtr.shape[-1]),
                "train_snrs": list(TRAIN_SNRS), "best_val_auc": best_auc}, CKPT)
    print(f"\n[ckpt] saved best-val model (AUC {best_auc:.3f}) -> {CKPT}", flush=True)

    # The headline: detector vs. the whitening optimum, per SNR (the Q1 gap).
    # Logit scores for a meaningful d′; AUC/Pd are rank-based so unaffected.
    sv = score_all(model, Xva, device, as_logit=True)
    yv = yva.numpy()
    print("\n=== detector vs. whitening optimum (val pool, per SNR) ===")
    print(f"  {'SNR':>5} | {'detector d′/AUC/Pd':^24} | {'optimum(whiten) d′/AUC/Pd':^28} | {'Pd gap':>7}")
    print("  " + "-" * 78)
    for s in TRAIN_SNRS:
        m = snr_va == s
        det = detectability(sv[m & (yv == 1)], sv[m & (yv == 0)])
        opt = detectability(wmf_va[m & (yv == 1)], wmf_va[m & (yv == 0)])
        gap = opt["pd_at_far"] - det["pd_at_far"]
        print(f"  {s:>5.0f} | {det['dprime']:>5.2f}/{det['auc']:.3f}/{det['pd_at_far']:.2f}        | "
              f"{opt['dprime']:>5.2f}/{opt['auc']:.3f}/{opt['pd_at_far']:.2f}            | {gap:>+6.2f}")
    print(f"\n(total {time.time()-t0:.0f}s on {device})")


if __name__ == "__main__":
    main()
