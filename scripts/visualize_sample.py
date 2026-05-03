"""Render one synthetic sequence and save a PNG strip for eyeball verification.

    python -m scripts.visualize_sample --snr-db 5 --n 10 --out clean.png
    python -m scripts.visualize_sample --snr-db -15 --n 10 --maneuvers --out hard.png

The strip shows three rows: the observed frame (target overlaid in red), the
noise-free target render, and the ground-truth soft mask. The figure title
reports both the prescribed and rendered SNR — they should agree to within
~0.1 dB at default sigma.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from synthetic.sequence import SequenceConfig, generate_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snr-db", type=float, required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--maneuvers", action="store_true")
    parser.add_argument("--clutter-rms", type=float, default=0.0)
    parser.add_argument("--clutter-drift", type=float, default=0.0)
    parser.add_argument("--dropout-prob", type=float, default=0.0)
    parser.add_argument("--jitter-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("sample.png"))
    args = parser.parse_args()

    cfg = SequenceConfig(
        n_frames=args.n,
        snr_db=args.snr_db,
        maneuvers=args.maneuvers,
        clutter_rms=args.clutter_rms,
        clutter_drift_px_per_frame=args.clutter_drift,
        dropout_prob=args.dropout_prob,
        jitter_std_px=args.jitter_std,
        seed=args.seed,
    )
    sample = generate_sequence(cfg)

    n = sample.frames.shape[0]
    fig, axes = plt.subplots(3, n, figsize=(max(6, n * 1.6), 5.0), squeeze=False)
    row_labels = ("frame", "target only", "soft mask")
    cmaps = ("gray", "hot", "viridis")
    fields = (sample.frames, sample.target_signal, sample.mask_soft)

    for row_idx, (label, cmap, field) in enumerate(zip(row_labels, cmaps, fields)):
        for t in range(n):
            ax = axes[row_idx, t]
            ax.imshow(field[t], cmap=cmap, interpolation="nearest")
            if row_idx == 0:
                ax.plot(
                    sample.positions[t, 1],
                    sample.positions[t, 0],
                    "r+",
                    ms=10,
                    mew=1.5,
                )
                ax.set_title(f"t={t}{'' if sample.observed[t] else ' (drop)'}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row_idx, 0].set_ylabel(label, fontsize=9)

    fig.suptitle(
        f"SNR_prescribed={sample.snr_db_prescribed:.2f} dB | "
        f"SNR_measured={sample.snr_db_measured:.2f} dB | "
        f"N={n}, maneuvers={args.maneuvers}, clutter_rms={args.clutter_rms}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Wrote {args.out}")
    print(f"  prescribed SNR: {sample.snr_db_prescribed:.3f} dB")
    print(f"  measured SNR:   {sample.snr_db_measured:.3f} dB")
    print(f"  observed frames: {int(sample.observed.sum())} / {n}")


if __name__ == "__main__":
    main()
