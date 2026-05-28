# SNN vs ConvLSTM Drone Detection

Isolated experiment comparing a Spiking Neural Network (SNN) version of the
tracklet detector against the existing ConvLSTM-based recurrent U-Net.

All code in this folder. The main project (`src/`, `scripts/`, `tests/`,
`demo_outputs/`) is read-only.

## Quick run

```bash
# from repo root
.venv/bin/pip install -r spikeNN/requirements.txt
.venv/bin/python -m pytest spikeNN/tests/ -v
.venv/bin/python -m spikeNN.training.train_snn
.venv/bin/python -m spikeNN.eval.compare
```

Artifacts land in `spikeNN/results/`:
- `snn_checkpoint.pt` — trained weights
- `snn_history.json` — per-epoch training history
- `snn_snr_sweep.json` — SNR sweep results (same schema as baseline)
- `comparison_waterfall.png` — headline figure: SNN vs baseline Pd vs SNR
- `REPORT.md` — written analysis

## Design

Mirrors `TrackletRecurrentUNet` I/O contract exactly so loss, dataset, metrics,
and SNR sweep are reused unchanged from the main project. Only the model
architecture differs: per-frame Conv-LIF encoder/decoder + a spiking recurrent
bottleneck (membrane potential carries across the T=32 frames, like ConvLSTM
gating but native to LIF dynamics).
