# NP-Ceiling — Neyman-Pearson optimal detector for low-SNR drone detection

Computes the theoretical upper bound on detection probability for the synthetic
data generator, to overlay on every existing detector waterfall.

Two ceilings:
- **CMF** (Clairvoyant Matched Filter): cheats with ground-truth trajectory →
  absolute upper bound, "what's the most signal we could ever pull out of these frames."
- **GLRT** (Generalized Likelihood Ratio Test): estimates trajectory from data via
  grid search over (initial_position, velocity) → realistic upper bound under the
  composite hypothesis "target somewhere with unknown velocity."

## Quick run

```bash
# from repo root
.venv/bin/python -m pytest npceiling/tests/ -v
.venv/bin/python -m npceiling.eval.compute_ceiling
.venv/bin/python -m npceiling.compare.plot_with_ceiling
```

Artifacts land in `npceiling/results/`:
- `cmf_snr_sweep.json`, `glrt_snr_sweep.json` — raw Pd-vs-SNR ceilings
- `ceiling_waterfall.png` — 6-line plot: baseline + V1/V2/V3 + GLRT + CMF
- `REPORT.md` — written analysis: gap-to-optimal per SNR cell

## Design notes

- AWGN-only: sequences restricted to `positive_uav` and `empty_background`,
  `clutter_rms=0`, no distractors. The matched filter is NP-optimal *only* under
  pure AWGN. Clutter robustness is out-of-scope future work.
- Template matches the project's PSF rendering exactly (sub-pixel
  normalization compensation per `src/synthetic/targets.py`).
- Reuses `pd_at_far` from `src/training/metrics.py` so the ceiling Pd is
  computed identically to the existing detector sweep.
- Pure numpy + matplotlib. No torch. No new dependencies.
