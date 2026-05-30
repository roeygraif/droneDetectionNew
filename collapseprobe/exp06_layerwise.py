"""Experiment 06: detectability vs. depth — where does the network lose it?

The Q1 figure. Freeze the retrained detector (Exp 05), push the IR3D eval tubes
through it, and at each stage estimate the best linear detectability with a
Fisher-LDA probe (same instrument as Exp 01). Overlay the whitening optimum
(`wmf_scores`). The stage with the largest drop is the bottleneck.

Caveat: detectability cannot truly *increase* with depth (data-processing
inequality), but a *linear* probe can be non-monotone (a later stage may be more
linearly separable). We read the trend and the largest drop, not tiny wiggles.

Run:  python -m collapseprobe.exp06_layerwise
Out:  console table + collapseprobe/fig_cliff.png
"""
from __future__ import annotations

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

CKPT = ROOT / "collapseprobe" / "detector_ckpt.pt"
SNRS = (-6.0, -9.0, -12.0, -15.0)
FIG = ROOT / "collapseprobe" / "fig_cliff.png"


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def probe_stage(feats, labels, tr, te):
    yte = labels[te]
    s = fisher_lda_scores(feats[tr], labels[tr], feats[te])
    return detectability(s[yte == 1], s[yte == 0])


def run_cell(model, device, snr):
    sp = load_split(_split_path(DATA_DIR / "ir3d", snr))
    X, y, wmf = sp["tubes"], sp["labels"], sp["wmf_scores"]
    feats, _ = collect_stage_features(model, X, device)
    tr, te = stratified_half_split(y, seed=0)
    yte = y[te]
    out = {st: probe_stage(feats[st], y, tr, te)["auc"] for st in STAGES}
    out["optimum"] = detectability(wmf[te][yte == 1], wmf[te][yte == 0])["auc"]
    return out


def main():
    device = _device()
    model, ck = load_detector(CKPT, device)
    print(f"loaded detector (best_val_auc={ck.get('best_val_auc', float('nan')):.3f}) on {device}\n")

    results = {snr: run_cell(model, device, snr) for snr in SNRS}

    cols = list(SNRS)
    print("=== Exp 06: per-stage linear-probe AUC vs. the whitening optimum (IR3D) ===\n")
    header = "  " + f"{'stage':<10}" + "".join(f"{f'{s:+.0f}dB':>10}" for s in cols)
    print(header); print("  " + "-" * (10 + 10 * len(cols)))
    for st in STAGES + ["optimum"]:
        row = "  " + f"{st:<10}" + "".join(f"{results[s][st]:>10.3f}" for s in cols)
        print(row)

    # Largest stage-to-stage drop per SNR (the cliff).
    print("\n  largest drop (stage→stage, AUC) per SNR:")
    for s in cols:
        aucs = [results[s][st] for st in STAGES]
        drops = [(STAGES[i] + "→" + STAGES[i + 1], aucs[i] - aucs[i + 1]) for i in range(len(STAGES) - 1)]
        where, d = max(drops, key=lambda kv: kv[1])
        print(f"    {s:+.0f} dB: {where:<18} Δ={d:+.3f}   (input {aucs[0]:.3f} → logit {aucs[-1]:.3f})")

    # Figure: detectability vs depth, one line per SNR, optimum as dashed marker.
    xs = np.arange(len(STAGES))
    plt.figure(figsize=(8, 5))
    for s in cols:
        plt.plot(xs, [results[s][st] for st in STAGES], marker="o", label=f"{s:+.0f} dB")
        plt.scatter([len(STAGES) - 0.5], [results[s]["optimum"]], marker="*", s=120,
                    color=plt.gca().lines[-1].get_color(), zorder=5)
    plt.axhline(0.5, color="gray", lw=0.8, ls=":")
    plt.xticks(xs, STAGES, rotation=30, ha="right")
    plt.ylabel("linear-probe detectability (ROC-AUC)")
    plt.title("Detectability vs. depth (★ = whitening optimum)")
    plt.legend(title="per-frame SNR", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG, dpi=130)
    print(f"\n  figure -> {FIG}")


if __name__ == "__main__":
    main()
