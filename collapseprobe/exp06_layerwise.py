"""Experiment 06: detectability vs. depth — where does the network lose it?

The Q1 figure. Freeze a trained detector, push the IR3D eval tubes through it, and
at each stage estimate the best linear detectability with a Fisher-LDA probe (same
instrument as Exp 01). Overlay the whitening optimum (`wmf_scores`). The stage
with the largest drop is the bottleneck.

Robustness: each cell's stage activations are collected once, then probed over
several train/test splits, so we report mean ± std per stage (error bars on the
cliff) rather than a single noisy split.

Caveat: detectability cannot truly *increase* with depth (data-processing
inequality), but a *linear* probe can be non-monotone (a later stage may be more
linearly separable). We read the trend and the largest drop, not tiny wiggles.

Run:  python -m collapseprobe.exp06_layerwise [--ckpt collapseprobe/detector_ckpt.pt]
Out:  console table (mean±std) + a figure next to the checkpoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from collapseprobe.dataset import DATA_DIR, _split_path, load_split  # noqa: E402
from collapseprobe.net_probe import STAGES, collect_stage_features, load_detector  # noqa: E402
from collapseprobe.probing import detectability, fisher_lda_scores, stratified_half_split  # noqa: E402

SNRS = (-6.0, -9.0, -12.0, -15.0)
SEEDS = (0, 1, 2, 3, 4)


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_cell(model, device, snr):
    """Return {stage: auc_over_seeds (np)} plus 'optimum'. Activations collected once."""
    sp = load_split(_split_path(DATA_DIR / "ir3d", snr))
    X, y, wmf = sp["tubes"], sp["labels"], sp["wmf_scores"]
    feats, _ = collect_stage_features(model, X, device)

    per_stage = {st: [] for st in STAGES}
    opt = []
    for seed in SEEDS:
        tr, te = stratified_half_split(y, seed=seed)
        yte = y[te]
        for st in STAGES:
            s = fisher_lda_scores(feats[st][tr], y[tr], feats[st][te])
            per_stage[st].append(detectability(s[yte == 1], s[yte == 0])["auc"])
        opt.append(detectability(wmf[te][yte == 1], wmf[te][yte == 0])["auc"])
    out = {st: np.array(v) for st, v in per_stage.items()}
    out["optimum"] = np.array(opt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(ROOT / "collapseprobe" / "detector_ckpt.pt"))
    args = ap.parse_args()
    ckpt = Path(args.ckpt)
    fig = ckpt.with_name(ckpt.stem.replace("detector_ckpt", "fig_cliff") if "detector_ckpt" in ckpt.stem
                         else "fig_cliff_" + ckpt.stem).with_suffix(".png")
    if fig == ckpt:
        fig = ROOT / "collapseprobe" / "fig_cliff.png"

    device = _device()
    model, ck = load_detector(ckpt, device)
    print(f"loaded {ckpt.name} (best_val_auc={ck.get('best_val_auc', float('nan')):.3f}) on {device}, "
          f"{len(SEEDS)} probe splits\n")

    results = {snr: run_cell(model, device, snr) for snr in SNRS}

    print(f"=== Exp 06: per-stage probe AUC (mean±std over {len(SEEDS)} splits) vs. optimum ===\n")
    print("  " + f"{'stage':<10}" + "".join(f"{f'{s:+.0f}dB':>14}" for s in SNRS))
    print("  " + "-" * (10 + 14 * len(SNRS)))
    for st in STAGES + ["optimum"]:
        cells = "".join(f"{results[s][st].mean():>8.3f}±{results[s][st].std():.3f}" for s in SNRS)
        print("  " + f"{st:<10}" + cells)

    print("\n  largest mean drop (stage→stage) per SNR:")
    for s in SNRS:
        m = [results[s][st].mean() for st in STAGES]
        drops = [(f"{STAGES[i]}→{STAGES[i+1]}", m[i] - m[i + 1]) for i in range(len(STAGES) - 1)]
        where, d = max(drops, key=lambda kv: kv[1])
        print(f"    {s:+.0f} dB: {where:<18} Δ={d:+.3f}   (input {m[0]:.3f} → logit {m[-1]:.3f})")

    xs = np.arange(len(STAGES))
    plt.figure(figsize=(8, 5))
    for s in SNRS:
        mean = np.array([results[s][st].mean() for st in STAGES])
        std = np.array([results[s][st].std() for st in STAGES])
        line = plt.errorbar(xs, mean, yerr=std, marker="o", capsize=2, label=f"{s:+.0f} dB")
        plt.scatter([len(STAGES) - 0.5], [results[s]["optimum"].mean()], marker="*", s=130,
                    color=line.lines[0].get_color(), zorder=5)
    plt.axhline(0.5, color="gray", lw=0.8, ls=":")
    plt.xticks(xs, STAGES, rotation=30, ha="right")
    plt.ylabel("linear-probe detectability (ROC-AUC)")
    plt.title(f"Detectability vs. depth  ({ckpt.name}, ★ = whitening optimum)")
    plt.legend(title="per-frame SNR", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig, dpi=130)
    print(f"\n  figure -> {fig}")


if __name__ == "__main__":
    main()
