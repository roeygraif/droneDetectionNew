"""Compute the CMF + GLRT ceilings as Pd vs SNR.

Mirrors ``scripts/train_demo.snr_sweep`` semantics:
- Same SNR grid (-20, -15, -12, -9, -6, -3, 0, +3 dB).
- Same ``runs_per_cell = 20`` (20 positive + 20 empty sequences per cell).
- Same deterministic seed strategy (``np.random.default_rng(31337)``).
- Same ``pd_at_far(target_far=1e-2, n_normalization=20)`` from
  ``src/training/metrics.py`` for the Pd computation.

Restricted to AWGN-only sequences (positive_uav / empty_background, no
clutter, no distractors) — this is the regime where the matched filter is
provably NP-optimal.

Output:
- ``results/cmf_snr_sweep.json``  — CMF ceiling
- ``results/glrt_snr_sweep.json`` — GLRT ceiling
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from synthetic.sequence import SequenceConfig, generate_sequence  # noqa: E402
from training.metrics import pd_at_far  # noqa: E402

from npceiling.theory.glrt import GLRTConfig, glrt_score  # noqa: E402
from npceiling.theory.matched_filter import cmf_score  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = ROOT / "npceiling" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CMF_SWEEP_JSON = RESULTS_DIR / "cmf_snr_sweep.json"
GLRT_SWEEP_JSON = RESULTS_DIR / "glrt_snr_sweep.json"

# Match scripts/train_demo:
SNR_GRID = (-20.0, -15.0, -12.0, -9.0, -6.0, -3.0, 0.0, 3.0)
RUNS_PER_CELL = 100   # bumped from 20 to tighten the very-low-SNR Pd estimates
N_FRAMES = 32
CANVAS = (64, 64)
TARGET_SIGMA = 1.0
NOISE_SIGMA = 1.0
FAR_TARGET = 1e-2  # FAR = 1/100
SWEEP_SEED = 31337  # same as snr_sweep in train_demo

# Base sequence config (AWGN-only, no clutter, no distractors).
BASE_SEQ_CFG = SequenceConfig(
    canvas_shape=CANVAS,
    n_frames=N_FRAMES,
    target_sigma=TARGET_SIGMA,
    noise_sigma=NOISE_SIGMA,
    motion="cv",
    speed_min_px_per_frame=0.3,
    speed_max_px_per_frame=1.5,
    boundary_margin_px=4.0,
    clutter_rms=0.0,
)


def _generate_positive(snr_db: float, seed: int):
    cfg = replace(BASE_SEQ_CFG, snr_db=snr_db, sequence_type="positive_uav", seed=seed)
    return generate_sequence(cfg)


def _generate_negative(snr_db: float, seed: int):
    cfg = replace(BASE_SEQ_CFG, snr_db=snr_db, sequence_type="empty_background", seed=seed)
    return generate_sequence(cfg)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def compute_ceilings(verbose: bool = True) -> tuple[list[dict], list[dict]]:
    """Run the full Pd-vs-SNR sweep for both CMF and GLRT."""
    cmf_rows: list[dict] = []
    glrt_rows: list[dict] = []
    glrt_cfg = GLRTConfig()

    rng = np.random.default_rng(SWEEP_SEED)
    overall_t0 = time.time()

    for snr_db in SNR_GRID:
        t_cell = time.time()
        cmf_pos: list[float] = []
        cmf_neg: list[float] = []
        glrt_pos: list[float] = []
        glrt_neg: list[float] = []

        for _ in range(RUNS_PER_CELL):
            # Deterministic seeds for this (snr, run) pair — match train_demo's pattern.
            seed_pos = int(rng.integers(1, 2 ** 31 - 1))
            seed_neg = int(rng.integers(1, 2 ** 31 - 1))

            sample_pos = _generate_positive(snr_db, seed_pos)
            sample_neg = _generate_negative(snr_db, seed_neg)

            # CMF — uses ground-truth trajectory of the positive on BOTH sequences
            # (paired-noise design: same trajectory, swap noise realization).
            cmf_pos.append(cmf_score(
                sample_pos.frames, sample_pos.positions, sample_pos.target_visible,
                sigma_psf=TARGET_SIGMA,
            ))
            cmf_neg.append(cmf_score(
                sample_neg.frames, sample_pos.positions, sample_pos.target_visible,
                sigma_psf=TARGET_SIGMA,
            ))

            # GLRT — searches velocity grid on both sequences independently.
            glrt_pos.append(glrt_score(sample_pos.frames, glrt_cfg))
            glrt_neg.append(glrt_score(sample_neg.frames, glrt_cfg))

        # Compute Pd@FAR using the project's existing function.
        cmf_pd = pd_at_far(
            cmf_pos, cmf_neg,
            target_far=FAR_TARGET,
            n_normalization=max(1, len(cmf_neg)),
        )
        glrt_pd = pd_at_far(
            glrt_pos, glrt_neg,
            target_far=FAR_TARGET,
            n_normalization=max(1, len(glrt_neg)),
        )

        cmf_row = {
            "snr_db": float(snr_db),
            "pd_at_far_1_per_100": float(cmf_pd["pd"]),
            "threshold": float(cmf_pd["threshold"]),
            "mean_pos_score": float(np.mean(cmf_pos)),
            "mean_neg_score": float(np.mean(cmf_neg)),
        }
        glrt_row = {
            "snr_db": float(snr_db),
            "pd_at_far_1_per_100": float(glrt_pd["pd"]),
            "threshold": float(glrt_pd["threshold"]),
            "mean_pos_score": float(np.mean(glrt_pos)),
            "mean_neg_score": float(np.mean(glrt_neg)),
        }
        cmf_rows.append(cmf_row)
        glrt_rows.append(glrt_row)

        dt = time.time() - t_cell
        if verbose:
            print(
                f"SNR={snr_db:+5.1f} dB | "
                f"CMF Pd={cmf_row['pd_at_far_1_per_100']:.2f} "
                f"(sep={cmf_row['mean_pos_score'] - cmf_row['mean_neg_score']:+.2f}) | "
                f"GLRT Pd={glrt_row['pd_at_far_1_per_100']:.2f} "
                f"(sep={glrt_row['mean_pos_score'] - glrt_row['mean_neg_score']:+.2f}) | "
                f"{dt:.1f}s",
                flush=True,
            )

    if verbose:
        print(f"\nTotal sweep time: {time.time() - overall_t0:.1f}s", flush=True)

    return cmf_rows, glrt_rows


def main():
    print(f"=== NP-Ceiling SNR sweep ===", flush=True)
    print(f"SNR grid: {SNR_GRID}", flush=True)
    print(f"runs/cell: {RUNS_PER_CELL}, FAR target: {FAR_TARGET}", flush=True)
    print(f"GLRT grid: 145 hypotheses (9 speeds × 16 directions + stationary)", flush=True)
    print(f"AWGN-only (clutter_rms=0, no distractors).", flush=True)
    print("", flush=True)

    cmf_rows, glrt_rows = compute_ceilings(verbose=True)

    with open(CMF_SWEEP_JSON, "w") as f:
        json.dump(cmf_rows, f, indent=2)
    with open(GLRT_SWEEP_JSON, "w") as f:
        json.dump(glrt_rows, f, indent=2)

    print(f"\nCMF sweep saved to {CMF_SWEEP_JSON}", flush=True)
    print(f"GLRT sweep saved to {GLRT_SWEEP_JSON}", flush=True)


if __name__ == "__main__":
    main()
