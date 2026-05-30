"""Experiment 03 (naive): find the SNR regime with dynamic range.

Exp 01/02 showed −3/−6 dB are saturated (optimum *and* honest readouts at
Pd 1.0). We generated lower-SNR cells (−9..−21). Here we sweep every cached
AWGN cell and, for each, report the matched-filter optimum next to the best
*honest* (template-free) readout. We are hunting for the band where the optimum
is still strong but a fair readout has begun to fall short — that gap is the
room a "where is detectability lost" study needs.

Run:  python -m collapseprobe.exp03_snr_sweep
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
from collapseprobe.exp02_honest_channels import HONEST, center_sum, pooled_features  # noqa: E402
from collapseprobe.probing import detectability, fisher_lda_scores, stratified_half_split  # noqa: E402

ALL_SNRS = (-3.0, -6.0, -9.0, -12.0, -15.0, -18.0, -21.0)


def run_cell(snr_db: float):
    split = load_split(_split_path(DATA_DIR / "awgn", snr_db))
    tubes, labels, cmf = split["tubes"], split["labels"], split["cmf_scores"]
    pos, neg = labels == 1, labels == 0

    opt = detectability(cmf[pos], cmf[neg])
    raw = detectability(center_sum(tubes, "raw")[pos], center_sum(tubes, "raw")[neg])

    tr, te = stratified_half_split(labels, seed=0)
    yte = labels[te]
    feats = pooled_features(tubes, HONEST)
    s = fisher_lda_scores(feats[tr], labels[tr], feats[te])
    probe = detectability(s[yte == 1], s[yte == 0])
    return opt, raw, probe


def main() -> None:
    print("=== Exp 03: SNR sweep — optimum vs. honest readout (AWGN) ===\n")
    print(f"{'SNR':>5} | {'OPTIMUM (matched filter)':^26} | {'naive raw center-sum':^26} | "
          f"{'linear probe (honest)':^26}")
    print(f"{'dB':>5} | {'AUC':>6}{'d-prime':>9}{'Pd@1e-2':>10} | "
          f"{'AUC':>6}{'d-prime':>9}{'Pd@1e-2':>10} | {'AUC':>6}{'d-prime':>9}{'Pd@1e-2':>10}")
    print("-" * 96)
    for snr in ALL_SNRS:
        try:
            opt, raw, probe = run_cell(snr)
        except FileNotFoundError:
            continue
        def fmt(m):
            return f"{m['auc']:>6.3f}{m['dprime']:>9.2f}{m['pd_at_far']:>10.2f}"
        print(f"{snr:>5.0f} | {fmt(opt)} | {fmt(raw)} | {fmt(probe)}")
    print("\n(gap = optimum Pd − honest Pd; the band where optimum is high but honest < optimum is the target regime)")


if __name__ == "__main__":
    main()
