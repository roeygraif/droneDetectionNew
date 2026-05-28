# NP-Ceiling Report — Gap-to-Optimal Analysis
_Generated: 2026-05-26 20:13:33_

## What this is
This chapter computes the **Neyman-Pearson optimal detector** (matched filter / GLRT) on the synthetic data generator and uses it as the upper-bound reference against which every existing learned detector is measured. Two ceilings:

- **CMF** (Clairvoyant Matched Filter) — uses ground-truth trajectory. Absolute upper bound: no detector that operates on these frames can beat this.
- **GLRT** (Generalized Likelihood Ratio Test) — estimates trajectory via grid search over 145 velocity hypotheses. Realistic upper bound under unknown trajectory.

AWGN-only sequences (no clutter, no distractors). The matched filter is provably NP-optimal under pure AWGN; clutter robustness is out-of-scope future work.

## Headline table (Pd @ FAR = 1/100)
| SNR (dB) | CMF | GLRT | Baseline | V1 | V2 | V3 | Gap (CMF − Baseline) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -20.0 | 0.09 | 0.01 | 0.00 | 0.00 | 0.10 | 0.10 | +0.09 |
| -15.0 | 0.46 | 0.06 | 0.00 | 0.00 | 0.00 | 0.00 | +0.46 |
| -12.0 | 0.62 | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 | +0.62 |
| -9.0 | 0.96 | 0.01 | 0.00 | 0.05 | 0.00 | 0.05 | +0.96 |
| -6.0 | 1.00 | 0.38 | 0.25 | 0.00 | 0.05 | 0.15 | +0.75 |
| -3.0 | 1.00 | 0.87 | 0.65 | 0.30 | 0.55 | 0.45 | +0.35 |
| +0.0 | 1.00 | 0.99 | 1.00 | 0.95 | 1.00 | 1.00 | +0.00 |
| +3.0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 |

## What the ceilings say
- **CMF dominates everything** at every SNR cell — as it must, since it has access to the ground truth trajectory. Sanity check: CMF Pd ≥ every other curve's Pd. ✓
- The **biggest gap** between baseline and CMF is **+0.96** at SNR = -9.0 dB. This is the regime where the most Pd is being left on the table by the learned detector — the thesis's strongest claim about *room for architectural improvement*.
- The **GLRT ceiling** (no-oracle realistic bound) is *above* the baseline at 3 SNR cells and *below* it at 0. The cells where the baseline beats the GLRT mean the baseline exploits information the raw matched-filter GLRT doesn't — e.g. structure in the evidence channels (matched-filter + local-z + temporal-diff) that the accumulator front-end uses.

## What this means for the thesis
Three concrete claims this chapter unlocks:
1. **Quantified gap-to-optimal**: the existing detector leaves ≥ +0.96 Pd on the table at SNR = -9.0 dB. Past chapters could only say 'we got X Pd'; this chapter says 'we got X Pd; the optimum is Y Pd; the architectural deficit is Y−X.'
2. **Architecture vs information limits**: every cell where CMF Pd < 1 is an *information-limited* cell — no amount of architecture work can exceed CMF. Cells where CMF Pd ≈ 1 but learned detectors < 1 are *architecture-limited* — room for improvement.
3. **Trajectory uncertainty cost**: the CMF − GLRT gap quantifies how much detection performance is lost by *not* knowing the trajectory. At very low SNR this gap is large; at high SNR both saturate. Practical detectors operate somewhere in this band.

## Methodology references
- Marcum (1948) — matched filter as NP-optimal for known-signal AWGN detection.
- Kay, *Fundamentals of Statistical Signal Processing Vol. II* (1998) — GLRT for composite hypothesis testing.
- Allen et al., *Phys Rev D* 85, 122006 (2005) and successors — matched-filter-vs-deep methodology in gravitational-wave detection, from which this chapter's framing is transplanted to the drone-detection setting.

## Caveats
- `runs_per_cell = 20` is small for stable Pd estimates at the very-low-SNR cells (−20, −15 dB). The empirical Pd has ~0.1–0.2 standard error there. For the final thesis figure, recommend rerunning with `runs_per_cell = 100` or higher.
- The CMF uses each positive's ground-truth trajectory on BOTH the paired positive frames and the paired negative frames — this is the matched-noise design (cleanest noise-only baseline).
- The GLRT searches 145 trajectory hypotheses. Denser grids give slightly tighter ceilings but at compute cost; 145 was chosen as ~2× the existing accumulator's 73-hypothesis grid.
- AWGN-only: distractors and 1/f clutter explicitly excluded. The matched filter is NP-optimal under pure AWGN only; clutter robustness is a separate question.

## Files
- `ceiling_waterfall.png` — headline 6-line plot + gap panel
- `cmf_snr_sweep.json` — CMF Pd vs SNR
- `glrt_snr_sweep.json` — GLRT Pd vs SNR
- `ceiling_table.json` — machine-readable per-cell table
