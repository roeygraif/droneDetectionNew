"""4-way head-to-head: Baseline vs SNN V1 vs SNN V2 vs SNN V3 (LIAF).

Reads all four pre-computed SNR sweeps and renders the comparison plot + report.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from spikeNN.eval.compare import CANONICAL_BASELINE, BASELINE_PARAM_COUNT  # noqa: E402

RESULTS_DIR = ROOT / "spikeNN" / "results"
V1_SWEEP = RESULTS_DIR / "snn_snr_sweep.json"
V2_SWEEP = RESULTS_DIR / "snn_v2_snr_sweep.json"
V3_SWEEP = RESULTS_DIR / "snn_v3_snr_sweep.json"
V1_HISTORY = RESULTS_DIR / "snn_history.json"
V2_HISTORY = RESULTS_DIR / "snn_v2_history.json"
V3_HISTORY = RESULTS_DIR / "snn_v3_history.json"
PLOT_4WAY = RESULTS_DIR / "comparison_4way_waterfall.png"
TABLE_4WAY = RESULTS_DIR / "comparison_4way_table.json"
REPORT_4WAY = RESULTS_DIR / "REPORT.md"


def _read(p: Path):
    with open(p) as f:
        return json.load(f)


def plot_4way(base, v1, v2, v3, out_path):
    def _ks(rows, key):
        return [r[key] for r in rows]
    snrs_b, pd_b = _ks(base, "snr_db"), _ks(base, "pd_at_far_1_per_100")
    snrs_v1, pd_v1 = _ks(v1, "snr_db"), _ks(v1, "pd_at_far_1_per_100")
    snrs_v2, pd_v2 = _ks(v2, "snr_db"), _ks(v2, "pd_at_far_1_per_100")
    snrs_v3, pd_v3 = _ks(v3, "snr_db"), _ks(v3, "pd_at_far_1_per_100")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(snrs_b, pd_b, marker="o", linewidth=2.2,
                 label="Baseline (ConvLSTM, 152K)", color="#1f77b4")
    axes[0].plot(snrs_v1, pd_v1, marker="s", linewidth=1.8,
                 label="V1 (LIF + spike readout, 64K)", color="#d62728")
    axes[0].plot(snrs_v2, pd_v2, marker="^", linewidth=1.8,
                 label="V2 (LIF + membrane readout, 143K)", color="#2ca02c")
    axes[0].plot(snrs_v3, pd_v3, marker="D", linewidth=2.2,
                 label="V3 (LIAF analog, 143K)", color="#9467bd")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Pd @ FAR = 1/100")
    axes[0].set_title("Detection probability — 4-way")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8)

    # Score separation = pos - neg (clearer story than overlaying both lines).
    def _sep(rows):
        return [r["mean_pos_score"] - r["mean_neg_score"] for r in rows]
    axes[1].plot(snrs_b, _sep(base), marker="o", linewidth=2.2,
                 label="Baseline", color="#1f77b4")
    axes[1].plot(snrs_v1, _sep(v1), marker="s", linewidth=1.8, label="V1", color="#d62728")
    axes[1].plot(snrs_v2, _sep(v2), marker="^", linewidth=1.8, label="V2", color="#2ca02c")
    axes[1].plot(snrs_v3, _sep(v3), marker="D", linewidth=2.2, label="V3", color="#9467bd")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("score_pos − score_neg")
    axes[1].set_title("Score separation vs SNR")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_report(base, v1, v2, v3, h1, h2, h3, out_path):
    b_by = {r["snr_db"]: r for r in base}
    v1_by = {r["snr_db"]: r for r in v1}
    v2_by = {r["snr_db"]: r for r in v2}
    v3_by = {r["snr_db"]: r for r in v3}
    snrs = sorted(set(b_by) | set(v1_by) | set(v2_by) | set(v3_by))

    n_b = BASELINE_PARAM_COUNT
    n_v1 = h1.get("param_count", "?")
    n_v2 = h2.get("param_count", "?")
    n_v3 = h3.get("param_count", "?")

    def _stage(h, k):
        return h.get("stage_results", {}).get(k, float("nan"))

    lines = []
    lines.append("# SNN vs ConvLSTM Drone Detection — 4-way Comparison (incl. LIAF V3)\n")
    lines.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_\n\n")

    lines.append("## Models\n")
    lines.append("| Model | Description | Params | Hidden representation |\n")
    lines.append("|---|---|---:|---|\n")
    lines.append(f"| **Baseline** | U-Net + ConvLSTM bottleneck | {n_b:,} | continuous (LSTM gates) |\n")
    lines.append(f"| **V1** | Conv-LIF U-Net, spike-rate readout | {n_v1:,} | binary spikes |\n")
    lines.append(f"| **V2** | Conv-LIF U-Net (capacity-matched), membrane readout | {n_v2:,} | binary spikes (continuous at output only) |\n")
    lines.append(f"| **V3 (LIAF)** | Conv-LIAF U-Net — analog fire (sigmoid(mem − thr)) | {n_v3:,} | **continuous analog throughout** |\n")
    lines.append("\nAll four trained on the identical 3-stage curriculum (80/200/4000 videos, 6/5/10 epochs).\n")

    lines.append("\n## Per-stage validation ROC-AUC\n")
    lines.append("| Stage | Baseline | V1 | V2 | V3 |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    lines.append(f"| 1 warmup | 0.998 | {_stage(h1, 'stage1_best_val_auc'):.3f} | "
                 f"{_stage(h2, 'stage1_best_val_auc'):.3f} | "
                 f"{_stage(h3, 'stage1_best_val_auc'):.3f} |\n")
    lines.append(f"| 2 transition | 0.930 | {_stage(h1, 'stage2_best_val_auc'):.3f} | "
                 f"{_stage(h2, 'stage2_best_val_auc'):.3f} | "
                 f"{_stage(h3, 'stage2_best_val_auc'):.3f} |\n")
    lines.append(f"| 3 finetune | 0.670 | {_stage(h1, 'stage3_best_val_auc'):.3f} | "
                 f"{_stage(h2, 'stage3_best_val_auc'):.3f} | "
                 f"{_stage(h3, 'stage3_best_val_auc'):.3f} |\n")

    lines.append("\n## Headline waterfall (Pd @ FAR = 1/100)\n")
    lines.append("| SNR (dB) | Baseline | V1 | V2 | V3 | ΔV3-Baseline |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|\n")
    v3_wins = v3_loss = 0
    for snr in snrs:
        b = b_by.get(snr, {}).get("pd_at_far_1_per_100", float("nan"))
        v1v = v1_by.get(snr, {}).get("pd_at_far_1_per_100", float("nan"))
        v2v = v2_by.get(snr, {}).get("pd_at_far_1_per_100", float("nan"))
        v3v = v3_by.get(snr, {}).get("pd_at_far_1_per_100", float("nan"))
        d3 = v3v - b
        if d3 > 0.05: v3_wins += 1
        elif d3 < -0.05: v3_loss += 1
        lines.append(f"| {snr:+.1f} | {b:.2f} | {v1v:.2f} | {v2v:.2f} | {v3v:.2f} | {d3:+.2f} |\n")

    lines.append("\n## Score separation (positive − negative mean score)\n")
    lines.append("| SNR (dB) | Baseline | V1 | V2 | V3 |\n")
    lines.append("|---:|---:|---:|---:|---:|\n")
    for snr in snrs:
        def _s(d):
            return d.get("mean_pos_score", 0.0) - d.get("mean_neg_score", 0.0)
        lines.append(f"| {snr:+.1f} | {_s(b_by.get(snr, {})):+.3f} | "
                     f"{_s(v1_by.get(snr, {})):+.3f} | {_s(v2_by.get(snr, {})):+.3f} | "
                     f"{_s(v3_by.get(snr, {})):+.3f} |\n")

    lines.append("\n## Verdict\n")
    lines.append(f"- **V3 (LIAF)** beat baseline at {v3_wins} cells, lost at {v3_loss} cells.\n")

    # Did V3 close the V2 residual gap?
    v2_loss_6 = (b_by.get(-6.0, {}).get("pd_at_far_1_per_100", 0)
                 - v2_by.get(-6.0, {}).get("pd_at_far_1_per_100", 0))
    v3_loss_6 = (b_by.get(-6.0, {}).get("pd_at_far_1_per_100", 0)
                 - v3_by.get(-6.0, {}).get("pd_at_far_1_per_100", 0))
    v2_loss_3 = (b_by.get(-3.0, {}).get("pd_at_far_1_per_100", 0)
                 - v2_by.get(-3.0, {}).get("pd_at_far_1_per_100", 0))
    v3_loss_3 = (b_by.get(-3.0, {}).get("pd_at_far_1_per_100", 0)
                 - v3_by.get(-3.0, {}).get("pd_at_far_1_per_100", 0))

    lines.append(f"\n### Did LIAF close the V2 residual gap?\n")
    lines.append(f"| SNR | V2 gap to baseline | V3 gap to baseline | Δ (V3 − V2) |\n")
    lines.append(f"|---:|---:|---:|---:|\n")
    lines.append(f"| −6 dB | {v2_loss_6:+.2f} | {v3_loss_6:+.2f} | "
                 f"{v3_loss_6 - v2_loss_6:+.2f} |\n")
    lines.append(f"| −3 dB | {v2_loss_3:+.2f} | {v3_loss_3:+.2f} | "
                 f"{v3_loss_3 - v2_loss_3:+.2f} |\n")

    lines.append("\n## Architectural lessons\n")
    lines.append("- **V1 → V2** (membrane readout, capacity parity): closed most of the gap at −3 / 0 dB.\n")
    lines.append("- **V2 → V3** (analog firing throughout): tests whether residual gap at −6 dB is from "
                 "spike thresholding *inside* the hidden layers, not just at the readout.\n")
    lines.append("- A meaningful V3 lift at −6 dB confirms the hidden-layer-thresholding diagnosis. "
                 "Otherwise the gap is something else (e.g. ConvLSTM's gating, BN dynamics, etc.).\n")

    lines.append("\n## Files in this run\n")
    lines.append(f"- `{PLOT_4WAY.name}` — 4-way waterfall plot\n")
    lines.append(f"- `snn_snr_sweep.json`, `snn_v2_snr_sweep.json`, `snn_v3_snr_sweep.json`\n")
    lines.append(f"- `comparison_4way_table.json` — machine-readable per-cell deltas\n")
    lines.append(f"- `snn_*_history.json` — per-stage val histories\n")
    lines.append(f"- `snn_*_checkpoint.pt` — trained weights\n")

    out_path.write_text("".join(lines))


def main():
    print("Loading sweep results...", flush=True)
    base = list(CANONICAL_BASELINE)
    v1 = _read(V1_SWEEP)
    v2 = _read(V2_SWEEP)
    v3 = _read(V3_SWEEP)
    h1 = _read(V1_HISTORY)
    h2 = _read(V2_HISTORY)
    h3 = _read(V3_HISTORY)

    plot_4way(base, v1, v2, v3, PLOT_4WAY)
    print(f"Plot: {PLOT_4WAY}", flush=True)

    snrs = sorted({r["snr_db"] for r in base + v1 + v2 + v3})
    table = []
    for snr in snrs:
        b = next((r for r in base if r["snr_db"] == snr), {})
        a = next((r for r in v1 if r["snr_db"] == snr), {})
        c = next((r for r in v2 if r["snr_db"] == snr), {})
        d = next((r for r in v3 if r["snr_db"] == snr), {})
        table.append({
            "snr_db": snr,
            "baseline_pd": b.get("pd_at_far_1_per_100", float("nan")),
            "v1_pd": a.get("pd_at_far_1_per_100", float("nan")),
            "v2_pd": c.get("pd_at_far_1_per_100", float("nan")),
            "v3_pd": d.get("pd_at_far_1_per_100", float("nan")),
            "delta_v3_baseline": d.get("pd_at_far_1_per_100", 0) - b.get("pd_at_far_1_per_100", 0),
            "delta_v3_v2": d.get("pd_at_far_1_per_100", 0) - c.get("pd_at_far_1_per_100", 0),
        })
    with open(TABLE_4WAY, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Table: {TABLE_4WAY}", flush=True)

    write_report(base, v1, v2, v3, h1, h2, h3, REPORT_4WAY)
    print(f"Report: {REPORT_4WAY}", flush=True)

    print(f"\n{'SNR':>6}  {'Base':>6}  {'V1':>6}  {'V2':>6}  {'V3':>6}  {'ΔV3-B':>7}  {'ΔV3-V2':>7}",
          flush=True)
    for r in table:
        marker = ""
        if r["delta_v3_baseline"] > 0.05: marker += "  V3⬆base"
        elif r["delta_v3_baseline"] < -0.05: marker += "  V3⬇base"
        if r["delta_v3_v2"] > 0.05: marker += "  V3⬆V2"
        print(f"{r['snr_db']:+6.1f}  {r['baseline_pd']:>6.2f}  {r['v1_pd']:>6.2f}  "
              f"{r['v2_pd']:>6.2f}  {r['v3_pd']:>6.2f}  "
              f"{r['delta_v3_baseline']:+7.2f}  {r['delta_v3_v2']:+7.2f}{marker}", flush=True)


if __name__ == "__main__":
    main()
