# SNN vs ConvLSTM Drone Detection — 4-way Comparison (incl. LIAF V3)
_Generated: 2026-05-25 15:15:33_

## Models
| Model | Description | Params | Hidden representation |
|---|---|---:|---|
| **Baseline** | U-Net + ConvLSTM bottleneck | 151,763 | continuous (LSTM gates) |
| **V1** | Conv-LIF U-Net, spike-rate readout | 63,828 | binary spikes |
| **V2** | Conv-LIF U-Net (capacity-matched), membrane readout | 142,588 | binary spikes (continuous at output only) |
| **V3 (LIAF)** | Conv-LIAF U-Net — analog fire (sigmoid(mem − thr)) | 142,588 | **continuous analog throughout** |

All four trained on the identical 3-stage curriculum (80/200/4000 videos, 6/5/10 epochs).

## Per-stage validation ROC-AUC
| Stage | Baseline | V1 | V2 | V3 |
|---|---:|---:|---:|---:|
| 1 warmup | 0.998 | 0.995 | 0.989 | 0.992 |
| 2 transition | 0.930 | 0.894 | 0.907 | 0.921 |
| 3 finetune | 0.670 | 0.607 | 0.655 | 0.656 |

## Headline waterfall (Pd @ FAR = 1/100)
| SNR (dB) | Baseline | V1 | V2 | V3 | ΔV3-Baseline |
|---:|---:|---:|---:|---:|---:|
| -20.0 | 0.00 | 0.00 | 0.10 | 0.10 | +0.10 |
| -15.0 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 |
| -12.0 | 0.00 | 0.00 | 0.05 | 0.00 | +0.00 |
| -9.0 | 0.00 | 0.05 | 0.00 | 0.05 | +0.05 |
| -6.0 | 0.25 | 0.00 | 0.05 | 0.15 | -0.10 |
| -3.0 | 0.65 | 0.30 | 0.55 | 0.45 | -0.20 |
| +0.0 | 1.00 | 0.95 | 1.00 | 1.00 | +0.00 |
| +3.0 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 |

## Score separation (positive − negative mean score)
| SNR (dB) | Baseline | V1 | V2 | V3 |
|---:|---:|---:|---:|---:|
| -20.0 | +0.000 | +0.001 | -0.005 | +0.007 |
| -15.0 | +0.000 | +0.048 | -0.006 | -0.050 |
| -12.0 | +0.000 | -0.014 | +0.002 | -0.018 |
| -9.0 | +0.000 | -0.005 | -0.026 | -0.016 |
| -6.0 | +0.100 | +0.006 | +0.009 | +0.028 |
| -3.0 | +0.200 | +0.062 | +0.157 | +0.205 |
| +0.0 | +0.370 | +0.230 | +0.385 | +0.367 |
| +3.0 | +0.460 | +0.377 | +0.554 | +0.440 |

## Verdict
- **V3 (LIAF)** beat baseline at 1 cells, lost at 2 cells.

### Did LIAF close the V2 residual gap?
| SNR | V2 gap to baseline | V3 gap to baseline | Δ (V3 − V2) |
|---:|---:|---:|---:|
| −6 dB | +0.20 | +0.10 | -0.10 |
| −3 dB | +0.10 | +0.20 | +0.10 |

## Architectural lessons
- **V1 → V2** (membrane readout, capacity parity): closed most of the gap at −3 / 0 dB.
- **V2 → V3** (analog firing throughout): tests whether residual gap at −6 dB is from spike thresholding *inside* the hidden layers, not just at the readout.
- A meaningful V3 lift at −6 dB confirms the hidden-layer-thresholding diagnosis. Otherwise the gap is something else (e.g. ConvLSTM's gating, BN dynamics, etc.).

## Files in this run
- `comparison_4way_waterfall.png` — 4-way waterfall plot
- `snn_snr_sweep.json`, `snn_v2_snr_sweep.json`, `snn_v3_snr_sweep.json`
- `comparison_4way_table.json` — machine-readable per-cell deltas
- `snn_*_history.json` — per-stage val histories
- `snn_*_checkpoint.pt` — trained weights
