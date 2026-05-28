"""3-way head-to-head: Baseline (ConvLSTM) vs SNN V1 (spike-rate) vs SNN V2 (membrane).

Reads pre-computed SNR sweeps:
  - baseline_snr_sweep.json  (re-uses canonical Run M numbers — see compare.py)
  - snn_snr_sweep.json       (V1: base=16/bot=32, spike-rate readout)
  - snn_v2_snr_sweep.json    (V2: base=24/bot=48, membrane readout)

Produces:
  - comparison_3way_waterfall.png  — overlaid Pd-vs-SNR + score-separation
  - comparison_3way_table.json     — per-cell deltas for both SNN variants
  - REPORT.md                       — full written analysis (overwrites V1 report)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from spikeNN.eval.compare import CANONICAL_BASELINE, BASELINE_PARAM_COUNT  # noqa: E402

RESULTS_DIR = ROOT / "spikeNN" / "results"
V1_SWEEP = RESULTS_DIR / "snn_snr_sweep.json"
V2_SWEEP = RESULTS_DIR / "snn_v2_snr_sweep.json"
V1_HISTORY = RESULTS_DIR / "snn_history.json"
V2_HISTORY = RESULTS_DIR / "snn_v2_history.json"
PLOT_3WAY = RESULTS_DIR / "comparison_3way_waterfall.png"
TABLE_3WAY = RESULTS_DIR / "comparison_3way_table.json"
REPORT_3WAY = RESULTS_DIR / "REPORT.md"


def _read(p: Path):
    with open(p) as f:
        return json.load(f)


def plot_3way(baseline_rows, v1_rows, v2_rows, out_path):
    snrs_b = [r["snr_db"] for r in baseline_rows]
    pd_b = [r["pd_at_far_1_per_100"] for r in baseline_rows]
    pos_b = [r["mean_pos_score"] for r in baseline_rows]
    neg_b = [r["mean_neg_score"] for r in baseline_rows]

    snrs_v1 = [r["snr_db"] for r in v1_rows]
    pd_v1 = [r["pd_at_far_1_per_100"] for r in v1_rows]
    pos_v1 = [r["mean_pos_score"] for r in v1_rows]
    neg_v1 = [r["mean_neg_score"] for r in v1_rows]

    snrs_v2 = [r["snr_db"] for r in v2_rows]
    pd_v2 = [r["pd_at_far_1_per_100"] for r in v2_rows]
    pos_v2 = [r["mean_pos_score"] for r in v2_rows]
    neg_v2 = [r["mean_neg_score"] for r in v2_rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(snrs_b, pd_b, marker="o", linewidth=2.2,
                 label="Baseline (ConvLSTM, 152K)", color="#1f77b4")
    axes[0].plot(snrs_v1, pd_v1, marker="s", linewidth=2.0,
                 label="SNN V1 (spike-rate, 64K)", color="#d62728")
    axes[0].plot(snrs_v2, pd_v2, marker="^", linewidth=2.0,
                 label="SNN V2 (membrane, 143K)", color="#2ca02c")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Pd @ FAR = 1/100")
    axes[0].set_title("Detection probability — 3-way comparison")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    axes[1].plot(snrs_b, pos_b, marker="o", color="#1f77b4", label="Baseline pos")
    axes[1].plot(snrs_b, neg_b, marker="o", linestyle="--", color="#1f77b4", alpha=0.5)
    axes[1].plot(snrs_v1, pos_v1, marker="s", color="#d62728", label="V1 pos")
    axes[1].plot(snrs_v1, neg_v1, marker="s", linestyle="--", color="#d62728", alpha=0.5)
    axes[1].plot(snrs_v2, pos_v2, marker="^", color="#2ca02c", label="V2 pos")
    axes[1].plot(snrs_v2, neg_v2, marker="^", linestyle="--", color="#2ca02c", alpha=0.5)
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("mean predicted score")
    axes[1].set_title("Score separation (solid=pos, dashed=neg)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_report(baseline_rows, v1_rows, v2_rows, v1_history, v2_history, out_path):
    b_by = {r["snr_db"]: r for r in baseline_rows}
    v1_by = {r["snr_db"]: r for r in v1_rows}
    v2_by = {r["snr_db"]: r for r in v2_rows}
    snrs = sorted(set(b_by) | set(v1_by) | set(v2_by))

    n_b = BASELINE_PARAM_COUNT
    n_v1 = v1_history.get("param_count", "?")
    n_v2 = v2_history.get("param_count", "?")
    s1_v1 = v1_history.get("stage_results", {}).get("stage1_best_val_auc", float("nan"))
    s2_v1 = v1_history.get("stage_results", {}).get("stage2_best_val_auc", float("nan"))
    s3_v1 = v1_history.get("stage_results", {}).get("stage3_best_val_auc", float("nan"))
    s1_v2 = v2_history.get("stage_results", {}).get("stage1_best_val_auc", float("nan"))
    s2_v2 = v2_history.get("stage_results", {}).get("stage2_best_val_auc", float("nan"))
    s3_v2 = v2_history.get("stage_results", {}).get("stage3_best_val_auc", float("nan"))

    lines = []
    lines.append("# SNN vs ConvLSTM Drone Detection — 3-Way Comparison\n")
    lines.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_\n\n")

    lines.append("## Models\n")
    lines.append("| Model | Architecture | Params | Readout |\n")
    lines.append("|---|---|---:|---|\n")
    lines.append(f"| **Baseline** | U-Net + ConvLSTM bottleneck | {n_b:,} | continuous |\n")
    lines.append(f"| **SNN V1**   | Conv-LIF U-Net + spiking recurrent (base=16, bot=32) | {n_v1:,} | spike-rate (binary thresholded) |\n")
    lines.append(f"| **SNN V2**   | Conv-LIF U-Net + spiking recurrent (base=24, bot=48) | {n_v2:,} | **continuous membrane** (preserves amplitude) |\n")
    lines.append("\nAll three trained on the identical 3-stage curriculum (80/200/4000 videos, 6/5/10 epochs) "
                 "with the same accumulator front-end and tracklet dataset config.\n")

    lines.append("\n## Per-stage validation ROC-AUC\n")
    lines.append("| Stage | Baseline (per progress.md) | SNN V1 | SNN V2 |\n")
    lines.append("|---|---:|---:|---:|\n")
    lines.append(f"| 1 warmup (+2..+10 dB) | 0.998 | {s1_v1:.3f} | {s1_v2:.3f} |\n")
    lines.append(f"| 2 transition (−5..+2 dB) | 0.930 | {s2_v1:.3f} | {s2_v2:.3f} |\n")
    lines.append(f"| 3 finetune (−15..−3 dB) | 0.670 | {s3_v1:.3f} | {s3_v2:.3f} |\n")

    lines.append("\n## Headline waterfall (Pd @ FAR = 1/100)\n")
    lines.append("| SNR (dB) | Baseline | V1 | V2 | Δ V1 | Δ V2 |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|\n")
    v1_wins = v2_wins = 0
    v1_loss = v2_loss = 0
    for snr in snrs:
        b = b_by.get(snr, {}).get("pd_at_far_1_per_100", float("nan"))
        v1 = v1_by.get(snr, {}).get("pd_at_far_1_per_100", float("nan"))
        v2 = v2_by.get(snr, {}).get("pd_at_far_1_per_100", float("nan"))
        d1 = v1 - b
        d2 = v2 - b
        if d1 > 0.05: v1_wins += 1
        if d1 < -0.05: v1_loss += 1
        if d2 > 0.05: v2_wins += 1
        if d2 < -0.05: v2_loss += 1
        lines.append(f"| {snr:+.1f} | {b:.2f} | {v1:.2f} | {v2:.2f} | {d1:+.2f} | {d2:+.2f} |\n")

    lines.append("\n## Score separation (positive − negative mean score)\n")
    lines.append("| SNR (dB) | Baseline | V1 | V2 |\n")
    lines.append("|---:|---:|---:|---:|\n")
    for snr in snrs:
        b = b_by.get(snr, {})
        v1 = v1_by.get(snr, {})
        v2 = v2_by.get(snr, {})
        b_sep = b.get("mean_pos_score", 0.0) - b.get("mean_neg_score", 0.0)
        v1_sep = v1.get("mean_pos_score", 0.0) - v1.get("mean_neg_score", 0.0)
        v2_sep = v2.get("mean_pos_score", 0.0) - v2.get("mean_neg_score", 0.0)
        lines.append(f"| {snr:+.1f} | {b_sep:+.3f} | {v1_sep:+.3f} | {v2_sep:+.3f} |\n")

    lines.append("\n## Verdict\n")
    lines.append(f"- **SNN V1** (spike-rate readout, 64K params): "
                 f"beat baseline at {v1_wins} cells, lost at {v1_loss} cells.\n")
    lines.append(f"- **SNN V2** (membrane readout, 143K params, capacity-matched): "
                 f"beat baseline at {v2_wins} cells, lost at {v2_loss} cells.\n")

    if v2_wins > v1_wins or v2_loss < v1_loss:
        lines.append("\n**Membrane readout + capacity parity moved the SNN closer to baseline.** "
                     "How much closer depends on whether the gap fully closed or just shrunk — "
                     "see the headline table.\n")
    else:
        lines.append("\n**Even with membrane readout and capacity parity, the SNN did not catch up to baseline.** "
                     "This strengthens the V1 null result: the gap is not just about information loss at "
                     "the readout — it's intrinsic to spiking-based representations in this matched-filter "
                     "regime.\n")

    # Where does V2 do best/worst?
    v2_deltas = [(snr, v2_by.get(snr, {}).get("pd_at_far_1_per_100", 0)
                       - b_by.get(snr, {}).get("pd_at_far_1_per_100", 0)) for snr in snrs]
    if v2_deltas:
        best_snr, best_d = max(v2_deltas, key=lambda kv: kv[1])
        worst_snr, worst_d = min(v2_deltas, key=lambda kv: kv[1])
        lines.append(f"\n- V2's **best cell** vs baseline: {best_d:+.2f} at SNR={best_snr:+.1f} dB.\n")
        lines.append(f"- V2's **worst cell** vs baseline: {worst_d:+.2f} at SNR={worst_snr:+.1f} dB.\n")

    lines.append("\n## What this means\n")
    lines.append("- If V2 *did* close the gap → the V1 result was an information-readout artifact; the "
                 "underlying spiking dynamics are competitive when amplitude info is preserved at output.\n")
    lines.append("- If V2 *did not* close the gap → the gap is intrinsic to spiking thresholding in the "
                 "hidden layers, not just at the readout. Capacity + readout are not enough.\n")
    lines.append("\nThe specific Δ values in the table above tell which interpretation is correct.\n")

    lines.append("\n## Files in this run\n")
    lines.append(f"- `{PLOT_3WAY.name}` — 3-way waterfall plot\n")
    lines.append(f"- `snn_snr_sweep.json` / `snn_v2_snr_sweep.json` — raw SNR sweep results\n")
    lines.append(f"- `comparison_3way_table.json` — machine-readable per-cell deltas\n")
    lines.append(f"- `snn_history.json` / `snn_v2_history.json` — training histories\n")
    lines.append(f"- `snn_checkpoint.pt` / `snn_v2_checkpoint.pt` — trained weights\n")

    out_path.write_text("".join(lines))


def main():
    print("Loading sweep results...", flush=True)
    baseline_rows = list(CANONICAL_BASELINE)
    v1_rows = _read(V1_SWEEP)
    v2_rows = _read(V2_SWEEP)
    v1_history = _read(V1_HISTORY)
    v2_history = _read(V2_HISTORY)

    plot_3way(baseline_rows, v1_rows, v2_rows, PLOT_3WAY)
    print(f"Plot: {PLOT_3WAY}", flush=True)

    # Per-cell table.
    snrs = sorted(set(r["snr_db"] for r in baseline_rows + v1_rows + v2_rows))
    table = []
    for snr in snrs:
        b = next((r for r in baseline_rows if r["snr_db"] == snr), {})
        v1 = next((r for r in v1_rows if r["snr_db"] == snr), {})
        v2 = next((r for r in v2_rows if r["snr_db"] == snr), {})
        table.append({
            "snr_db": snr,
            "baseline_pd": b.get("pd_at_far_1_per_100", float("nan")),
            "v1_pd": v1.get("pd_at_far_1_per_100", float("nan")),
            "v2_pd": v2.get("pd_at_far_1_per_100", float("nan")),
            "delta_v1": v1.get("pd_at_far_1_per_100", 0) - b.get("pd_at_far_1_per_100", 0),
            "delta_v2": v2.get("pd_at_far_1_per_100", 0) - b.get("pd_at_far_1_per_100", 0),
        })
    with open(TABLE_3WAY, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Table: {TABLE_3WAY}", flush=True)

    write_report(baseline_rows, v1_rows, v2_rows, v1_history, v2_history, REPORT_3WAY)
    print(f"Report: {REPORT_3WAY}", flush=True)

    # Console summary.
    print(f"\n{'SNR':>6}  {'Baseline':>10}  {'V1':>10}  {'V2':>10}  {'ΔV1':>8}  {'ΔV2':>8}", flush=True)
    for r in table:
        marker = ""
        if r["delta_v2"] > 0.05: marker += " V2⬆"
        elif r["delta_v2"] < -0.05: marker += " V2⬇"
        if r["delta_v1"] > 0.05: marker += " V1⬆"
        elif r["delta_v1"] < -0.05: marker += " V1⬇"
        print(f"{r['snr_db']:+6.1f}  {r['baseline_pd']:>10.2f}  {r['v1_pd']:>10.2f}  {r['v2_pd']:>10.2f}  "
              f"{r['delta_v1']:+8.2f}  {r['delta_v2']:+8.2f}  {marker}", flush=True)


if __name__ == "__main__":
    main()
