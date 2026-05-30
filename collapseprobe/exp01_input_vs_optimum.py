"""Experiment 01 (naive): how detectable is the drone at the INPUT vs the OPTIMUM?

No network training. On the cached AWGN dataset we compare, at each SNR cell:
  - the optimum: the clairvoyant matched filter (stored cmf_scores),
  - a few hand-built naive statistics on the raw input tube,
  - a best linear readout of simple pooled input features (Fisher LDA, held-out).

Purpose: build and sanity-check the detectability measuring tools, and learn how
much of the optimum a simple/linear readout of the input recovers. This tells us
whether a linear probe is a fair instrument before we use it on deeper layers.

Run:  python -m collapseprobe.exp01_input_vs_optimum
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collapseprobe.dataset import DATA_DIR, _split_path, load_split  # noqa: E402
from collapseprobe.probing import detectability, fisher_lda_scores, stratified_half_split  # noqa: E402

# Channel order in the tube (from CropTubeConfig): 0 raw, 1 local_z, 2 matched,
# 3 temporal_diff, 4 evidence.
CH = {"raw": 0, "local_z": 1, "matched": 2, "temporal_diff": 3, "evidence": 4}
SNRS = (-3.0, -6.0)


def pooled_features(tubes: np.ndarray) -> np.ndarray:
    """Simple per-channel summary of each tube: [mean, max, center-3x3 mean over time]."""
    N, T, C, S, _ = tubes.shape
    c = S // 2
    feats = []
    for ch in range(C):
        x = tubes[:, :, ch]  # (N, T, S, S)
        feats.append(x.mean(axis=(1, 2, 3)))
        feats.append(x.max(axis=(1, 2, 3)))
        feats.append(x[:, :, c - 1:c + 2, c - 1:c + 2].mean(axis=(2, 3)).sum(axis=1))
    return np.stack(feats, axis=1)  # (N, 3*C)


def naive_stats(tubes: np.ndarray) -> dict[str, np.ndarray]:
    """A few obvious 'detectors' read straight off the input tube."""
    c = tubes.shape[-1] // 2
    def center_sum(ch):  # integrate a channel's central 3x3 over time
        return tubes[:, :, ch, c - 1:c + 2, c - 1:c + 2].sum(axis=(1, 2, 3))
    return {
        "naive: raw energy": (tubes[:, :, CH["raw"]] ** 2).sum(axis=(1, 2, 3)),
        "naive: matched-ch center-sum": center_sum(CH["matched"]),
        "naive: evidence-ch center-sum": center_sum(CH["evidence"]),
    }


def run_cell(snr_db: float) -> list[tuple[str, dict]]:
    split = load_split(_split_path(DATA_DIR / "awgn", snr_db))
    tubes, labels, cmf = split["tubes"], split["labels"], split["cmf_scores"]
    pos, neg = labels == 1, labels == 0
    rows: list[tuple[str, dict]] = []

    rows.append(("optimum (matched filter)", detectability(cmf[pos], cmf[neg])))

    for name, stat in naive_stats(tubes).items():
        rows.append((name, detectability(stat[pos], stat[neg])))

    feats = pooled_features(tubes)
    tr, te = stratified_half_split(labels, seed=0)
    scores_te = fisher_lda_scores(feats[tr], labels[tr], feats[te])
    yte = labels[te]
    rows.append(("linear probe on input (held-out)", detectability(scores_te[yte == 1], scores_te[yte == 0])))
    return rows


def main() -> None:
    print("=== Exp 01: input representation vs. the optimum (AWGN, oracle tubes) ===\n")
    for snr in SNRS:
        print(f"SNR = {snr:+.0f} dB")
        print(f"  {'detector':<36}{'AUC':>8}{'d-prime':>10}{'Pd@FAR=1/100':>14}")
        for name, m in run_cell(snr):
            print(f"  {name:<36}{m['auc']:>8.3f}{m['dprime']:>10.2f}{m['pd_at_far']:>14.2f}")
        print()


if __name__ == "__main__":
    main()
