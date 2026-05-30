"""Experiment 02 (naive): the *honest* gap to the optimum.

Exp 01 showed the cached tube already carries clairvoyant `matched`/`evidence`
channels (built from the known template), so a trivial readout of them hits the
optimum and there is no dynamic range. Here we remove that shortcut: probe only
the template-free channels (raw, local_z, temporal_diff) and ask how much of the
matched-filter optimum a fair (naive / best-linear) readout of the raw sensor
data recovers.

If even the honest readout is pinned at the ceiling, we must drop SNR to open
dynamic range. If it falls below the optimum, the headroom is already on the
data we have.

Run:  python -m collapseprobe.exp02_honest_channels
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

CH = {"raw": 0, "local_z": 1, "matched": 2, "temporal_diff": 3, "evidence": 4}
HONEST = ("raw", "local_z", "temporal_diff")  # template-free channels only
SNRS = (-3.0, -6.0)


def pooled_features(tubes: np.ndarray, channels) -> np.ndarray:
    """[mean, max, center-3x3 time-sum] per selected channel."""
    S = tubes.shape[-1]
    c = S // 2
    feats = []
    for name in channels:
        x = tubes[:, :, CH[name]]  # (N, T, S, S)
        feats.append(x.mean(axis=(1, 2, 3)))
        feats.append(x.max(axis=(1, 2, 3)))
        feats.append(x[:, :, c - 1:c + 2, c - 1:c + 2].mean(axis=(2, 3)).sum(axis=1))
    return np.stack(feats, axis=1)


def center_sum(tubes, ch):
    c = tubes.shape[-1] // 2
    return tubes[:, :, CH[ch], c - 1:c + 2, c - 1:c + 2].sum(axis=(1, 2, 3))


def run_cell(snr_db: float):
    split = load_split(_split_path(DATA_DIR / "awgn", snr_db))
    tubes, labels, cmf = split["tubes"], split["labels"], split["cmf_scores"]
    pos, neg = labels == 1, labels == 0
    rows = [("optimum (matched filter)", detectability(cmf[pos], cmf[neg]))]

    for ch in HONEST:
        rows.append((f"naive: {ch} center-sum", detectability(center_sum(tubes, ch)[pos],
                                                              center_sum(tubes, ch)[neg])))

    tr, te = stratified_half_split(labels, seed=0)
    yte = labels[te]
    for tag, chans in (("raw only", ("raw",)), ("honest (raw+z+tdiff)", HONEST)):
        feats = pooled_features(tubes, chans)
        s = fisher_lda_scores(feats[tr], labels[tr], feats[te])
        rows.append((f"linear probe: {tag}", detectability(s[yte == 1], s[yte == 0])))
    return rows


def main() -> None:
    print("=== Exp 02: honest (template-free) channels vs. the optimum (AWGN) ===\n")
    for snr in SNRS:
        print(f"SNR = {snr:+.0f} dB")
        print(f"  {'detector':<34}{'AUC':>8}{'d-prime':>10}{'Pd@FAR=1/100':>14}")
        for name, m in run_cell(snr):
            print(f"  {name:<34}{m['auc']:>8.3f}{m['dprime']:>10.2f}{m['pd_at_far']:>14.2f}")
        print()


if __name__ == "__main__":
    main()
