"""Experiment 07: the C3 repair — average-pool vs. max-pool, against the optimum.

Exp 06b localized the detectability loss to the first encoder downsample. The
detection-theory reason max-pooling hurts at low SNR: it keeps the (upward-biased)
maximum of a mostly-noise patch, raising the noise floor, whereas the optimal
detector integrates. So we retrain identical detectors with average-pooling
(`--pool avg`) instead of max-pooling and measure how much of the
detector-vs-optimum gap it recovers.

Multi-seed: aggregates every available checkpoint per condition (mean ± std over
seeds), so the before/after isn't a single noisy run. This is signature figure #3.

Run:  python -m collapseprobe.exp07_repair
Out:  console table (mean±std, n seeds) + collapseprobe/fig_repair.png
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
from collapseprobe.net_probe import load_detector  # noqa: E402
from collapseprobe.probing import detectability  # noqa: E402

CP = ROOT / "collapseprobe"
SNRS = (-3.0, -6.0, -9.0, -12.0, -15.0)
# Each condition aggregates whatever seed checkpoints exist (added by the
# multi-seed firming run). Order: baseline first, repair second.
CONDS = [
    ("max-pool (baseline)", ["detector_ckpt_v2.pt", "detector_ckpt_max_1234.pt", "detector_ckpt_max_7777.pt"]),
    ("avg-pool (C3 repair)", ["detector_ckpt_avgpool.pt", "detector_ckpt_avg_1234.pt", "detector_ckpt_avg_7777.pt"]),
]
FIG = CP / "fig_repair.png"


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def logits(model, X, device, batch=64):
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().to(device)
        out.append(model(xb)["track_logit"].cpu().numpy())
    return np.concatenate(out)


def main():
    device = _device()
    cells = {s: load_split(_split_path(DATA_DIR / "ir3d", s)) for s in SNRS}

    opt = {}
    for s in SNRS:
        c = cells[s]; y = c["labels"]; w = c["wmf_scores"]
        opt[s] = detectability(w[y == 1], w[y == 0])["auc"]

    # condition -> SNR -> list of per-seed AUCs
    agg = {}
    nseeds = {}
    for name, files in CONDS:
        present = [f for f in files if (CP / f).exists()]
        nseeds[name] = len(present)
        per = {s: [] for s in SNRS}
        for f in present:
            model, _ = load_detector(CP / f, device)
            for s in SNRS:
                c = cells[s]; y = c["labels"]
                lg = logits(model, c["tubes"], device)
                per[s].append(detectability(lg[y == 1], lg[y == 0])["auc"])
        agg[name] = {s: np.array(per[s]) for s in SNRS}

    names = [n for n, _ in CONDS]
    print("=== Exp 07: C3 repair — detector AUC vs. optimum (IR3D eval), mean±std over seeds ===")
    print(f"  seeds/condition: " + ", ".join(f"{n}={nseeds[n]}" for n in names) + "\n")
    print(f"  {'SNR':>5}" + "".join(f"{n:>26}" for n in names) + f"{'optimum':>10}{'recovered':>11}")
    print("  " + "-" * (5 + 26 * len(names) + 21))
    for s in SNRS:
        b, r = agg[names[0]][s], agg[names[1]][s]
        frac = (r.mean() - b.mean()) / (opt[s] - b.mean()) if opt[s] - b.mean() > 1e-6 else float("nan")
        print(f"  {s:>5.0f}"
              f"{b.mean():>14.3f}±{b.std():.3f}"
              f"{r.mean():>14.3f}±{r.std():.3f}"
              f"{opt[s]:>10.3f}{frac:>10.0%}")

    xs = list(SNRS)
    plt.figure(figsize=(7.5, 5))
    plt.plot(xs, [opt[s] for s in SNRS], "k--*", lw=1.5, ms=11, label="optimum (whitening MF)")
    colors = {}
    for name, _ in CONDS:
        mean = np.array([agg[name][s].mean() for s in SNRS])
        std = np.array([agg[name][s].std() for s in SNRS])
        line = plt.plot(xs, mean, marker="o", label=f"{name} (n={nseeds[name]})")[0]
        colors[name] = line.get_color()
        plt.fill_between(xs, mean - std, mean + std, alpha=0.18, color=line.get_color())
    mb = np.array([agg[names[0]][s].mean() for s in SNRS])
    mr = np.array([agg[names[1]][s].mean() for s in SNRS])
    plt.fill_between(xs, mb, mr, where=(mr >= mb), alpha=0.12, color="green", label="recovered by repair")
    plt.xlabel("per-frame SNR (dB)"); plt.ylabel("detection AUC")
    plt.title("C3 repair: max-pool → avg-pool, vs. the optimum (mean±std over seeds)")
    plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(FIG, dpi=130)
    print(f"\n  figure -> {FIG}")


if __name__ == "__main__":
    main()
