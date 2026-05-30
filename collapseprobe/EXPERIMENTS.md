# collapseprobe experiment log

A running lab notebook. Append-only: one entry per experiment, newest at the
bottom of its section. The point is so we never repeat a dead end and always
know *why* we did the next thing. Every entry: **hypothesis → method → command →
result → what we learned → decision/next**. Numbers must come from a run, not
intuition (see `RESEARCH_CHARTER.md`, guardrail 1).

Anchor question (from the charter): *where, and why, does a deep low-SNR drone
detector fall short of the optimal detector; can it be repaired; what limits
temporal integration.* The tool is layer-wise detectability probing against a
computable matched-filter ceiling.

---

## Conventions

- **Detectability metrics.** For every "detector" (the optimum, a naive
  statistic, or a linear probe) we report three numbers on the same paired
  positive/negative set: **ROC-AUC**, **d′** (detection-theory deflection,
  `(μ⁺−μ⁻)/pooled-std`), and **Pd@FAR=1/100** (probability of detection at a
  false-alarm rate of 1 in 100). Code: `collapseprobe/probing.py`.
- **The optimum.** AWGN cells: the plain matched filter `cmf_scores` (NP-optimal
  in white noise). IR3D cells: the **whitening** matched filter `wmf_scores`
  (NP-optimal in correlated noise; `collapseprobe/whitening_mf.py`), now stored
  next to `cmf_scores` by the dataset build. **Do not use `cmf_scores` as the
  optimum for IR3D** — it understates it by 1.2–1.4× in d′ (Exp 04).
- **Tube channels** (5, from `CropTubeConfig`): `0 raw`, `1 local_z`,
  `2 matched`, `3 temporal_diff`, `4 evidence`. **Caveat (see Exp 01):**
  `matched` (2) and `evidence` (4) are *clairvoyant* — built from the known
  target template by `compute_evidence_maps`. `raw`/`local_z`/`temporal_diff`
  are template-free ("honest") readouts of the sensor data.
- **Cached cells** (oracle-centered, paired pos/neg, 200/class): `awgn` at
  SNR = −3..−21 dB; `ir3d` at −3..−18 dB (each carries `wmf_scores`). Regenerate
  with `python -m collapseprobe.dataset --noise-model {awgn,ir3d} --snr=...`.

---

## Open TODO / known gaps (so we don't forget)

- [x] ~~**No dynamic range yet.**~~ **Resolved (Exp 03).** Generated −9..−21 dB
  AWGN cells; the optimum lifts off saturation below −9 dB. Working regime for
  the network study: **−9 to −15 dB** (optimum Pd 0.5–0.94, headroom below).
- [ ] **Coarse Pd@FAR=1/100.** Only 200 neg/class → the 1/100 false-alarm
  threshold rests on ~2 negatives, so Pd@1e-2 is noisy (see −15 dB row). AUC/d′
  are smooth; trust those for trends, bump n/class before final figures.
- [ ] **Open question (Exp 03).** In clean AWGN with oracle centering, even a
  naive raw center-sum is *nearly* matched-filter-optimal (honest readout tracks
  the optimum within ~0.3–0.5 d′). So the input representation has little room to
  lose signal here — the interesting losses must come from (a) the network's
  internal processing, and/or (b) realistic IR3D noise where a plain readout is
  genuinely suboptimal. Steers the next phase.
- [x] ~~**Whitening matched filter for IR3D.**~~ **Built & validated (Exp 04).**
  `whitening_mf.py` (exact covariance operator + conjugate-gradient solve).
  **Consequence:** the dataset's stored `cmf_scores` is the *wrong* ceiling for
  IR3D cells — it understates the optimum by 1.4–2.3× in d′. Any "gap to optimum"
  in the IR3D regime must use the whitening MF, not `cmf_scores`.
- [x] ~~**Wire `wmf_score` into the dataset build.**~~ **Done.** IR3D cells now
  store `wmf_scores`; regenerated −3..−18 dB. Verified on the stored cells: the
  whitening ceiling beats the plain MF by 1.19–1.42× in d′ (cleanest in the
  −9..−15 band). `pytest collapseprobe/tests` green.
- [x] ~~**The network is not in the loop yet.**~~ **Detector retrained (Exp 05).**
  `collapseprobe/detector_ckpt.pt` — `TrackletRecurrentUNet`, trained on
  oracle-centered IR3D tubes. Frozen, with a clear gap to the whitening optimum
  that widens at low SNR (AUC 0.94→0.63 over −3..−15 dB vs optimum 1.0→0.91).
  SNN dropped (abandoned by the user — never worked).
- [x] ~~**Per-stage probing not built yet.**~~ **Done (Exp 06; refined by 06b).**
  `net_probe.py` + `exp06_layerwise.py`. Exp 06 (v1 detector) saw a U-shape; the
  robustness pass (Exp 06b, v2 detector) showed the robust part is a **single
  cliff at the first max-pool (enc1→enc2)** — the ConvLSTM-trough / head-discards
  parts of the U were v1 overfitting artifacts. Use the 06b reading.
- [x] ~~**Confirm the U-shape is not an overfit artifact.**~~ **Done (Exp 06b) —
  and it refined the finding.** Multi-split error bars + a better detector (v2:
  ~1.9× data, more reg, different init). **Robust:** input near-optimal, then a
  single sharp cliff at the **first max-pool (enc1→enc2, 31→15 px)** — largest
  drop at every SNR, both detectors. **Refuted as v1 artifacts:** the deep
  ConvLSTM trough and the "head discards decoder recovery" story (gone in v2).
- [x] ~~**C3 repair hypothesis (revised, from Exp 06b).**~~ **Prototyped, positive
  (Exp 07).** Swapped the encoder max-pools for **average-pooling** (same
  resolution reduction, but integrates instead of taking the noise-inflated max).
  Recovers 40–61% of the detector-vs-optimum AUC gap at −3/−6/−9 dB, 18% at −15;
  the targeted enc1→enc2 cliff roughly halved. One caveat: −12 dB regressed
  (−24%), almost certainly single-run-per-condition noise (val pool showed −12
  *improving*). → firm up with multi-seed before it's signature figure #3.
- [ ] **C3 multi-seed firming.** Exp 07 is one training run per condition; the
  −12 dB sign flip is within run-to-run noise. Train 2–3 seeds each of max/avg,
  report mean±std of the recovered gap, before finalizing figure #3.
- [ ] **Detector overfits a little.** train_loss → 0.03 on 1600 tubes; best-val
  checkpointing rescues a good epoch but the val curve is noisy. Adequate for a
  *representative* frozen detector (charter: tuning out of scope); revisit with
  more data only if the probing result hinges on it.

---

## Experiments

### Exp 01 — input representation vs. the optimum (naive baseline)

**Date:** 2026-05-30 · **Noise:** awgn · **Cells:** −3, −6 dB · **Net:** none (input only)
**Script:** `exp01_input_vs_optimum.py` · **Run:** `python -m collapseprobe.exp01_input_vs_optimum`

**Hypothesis.** Before trusting a linear probe as a per-layer ruler, check it on
the input: a best linear readout of simple pooled input features should land
between obviously-bad naive statistics and the matched-filter optimum. If it
does, the probe is a fair instrument.

**Method.** No training. On each cached AWGN cell, compute detectability of:
(a) the optimum (`cmf_scores`); (b) three naive statistics read straight off the
tube (raw energy; center-3×3 time-sum of the `matched` channel; same of the
`evidence` channel); (c) a Fisher-LDA linear probe on pooled per-channel
features (mean, max, center-3×3 time-sum), trained on a stratified half and
scored on the held-out half.

**Result.**

| SNR | detector | AUC | d′ | Pd@FAR=1/100 |
|----:|----------|----:|---:|-------------:|
| −3 | optimum (matched filter)        | 1.000 | 7.63 | 1.00 |
| −3 | naive: raw energy               | 0.502 | 0.01 | 0.04 |
| −3 | naive: matched-ch center-sum    | 1.000 | 7.32 | 1.00 |
| −3 | naive: evidence-ch center-sum   | 1.000 | 7.27 | 1.00 |
| −3 | linear probe on input (held-out)| 1.000 | 6.72 | 1.00 |
| −6 | optimum (matched filter)        | 1.000 | 5.47 | 1.00 |
| −6 | naive: raw energy               | 0.520 | 0.03 | 0.02 |
| −6 | naive: matched-ch center-sum    | 1.000 | 5.05 | 1.00 |
| −6 | naive: evidence-ch center-sum   | 1.000 | 5.08 | 1.00 |
| −6 | linear probe on input (held-out)| 1.000 | 4.98 | 1.00 |

**What we learned.**
1. **The measuring tools are sound.** Raw energy sits at chance (AUC ≈ 0.50) as
   it should; the matched filter is the ceiling; the linear probe lands just
   below the optimum (d′ 6.72 vs 7.63 at −3) — exactly the "fair instrument"
   ordering we wanted. The probe is trustworthy.
2. **Detection-theory sanity check passes.** Optimum d′ drops 7.63 → 5.47 from
   −3 to −6 dB. That ratio (1.39) ≈ √2 (1.41) = the √(SNR-power) scaling the
   matched filter must obey (Kay; Richards). The ruler behaves as theory says.
3. **The input is *already* near-optimal — and trivially so.** A plain
   center-sum of the `matched`/`evidence` channels hits Pd = 1.00. That is not
   the network being clever; those channels are **clairvoyant** (built from the
   known template, see Conventions). So "detectability at the input" is pinned
   at the ceiling and there is **no dynamic range** at −3/−6 dB.

**Decision / next.** Two consequences for the plan:
- We need a regime with headroom — the optimum high but **not** saturated —
  before any "where is detectability lost" figure can exist. Push SNR lower
  and/or restrict to template-free channels. → **Exp 02**: probe the *honest*
  (template-free) channels only, to see the real gap to the optimum on the data
  we already have, before paying to generate new cells.
- When the network goes in the loop, be explicit about which channels it
  ingests; if it gets the clairvoyant `evidence` channel, Q1 is trivial by
  construction. Flag for the architecture step.

---

### Exp 02 — the *honest* gap to the optimum (template-free channels)

**Date:** 2026-05-30 · **Noise:** awgn · **Cells:** −3, −6 dB · **Net:** none
**Script:** `exp02_honest_channels.py` · **Run:** `python -m collapseprobe.exp02_honest_channels`

**Hypothesis.** Exp 01's near-optimal input was an artifact of the clairvoyant
`matched`/`evidence` channels. Remove them and a fair readout of the
template-free channels (`raw`, `local_z`, `temporal_diff`) should fall clearly
below the optimum, revealing headroom on the data we already have.

**Method.** Same metrics. Naive center-3×3 time-sum of each honest channel, plus
Fisher-LDA probes on (raw only) and (raw+local_z+temporal_diff) pooled features.

**Result.**

| SNR | detector | AUC | d′ | Pd@1e-2 |
|----:|----------|----:|---:|--------:|
| −3 | optimum                       | 1.000 | 7.63 | 1.00 |
| −3 | naive: raw center-sum         | 1.000 | 7.00 | 1.00 |
| −3 | naive: local_z center-sum     | 1.000 | 6.56 | 1.00 |
| −3 | naive: temporal_diff center-sum | 0.528 | 0.11 | 0.03 |
| −3 | linear probe: raw only        | 1.000 | 6.41 | 1.00 |
| −3 | linear probe: honest          | 1.000 | 6.24 | 1.00 |
| −6 | optimum                       | 1.000 | 5.47 | 1.00 |
| −6 | naive: raw center-sum         | 1.000 | 4.83 | 0.99 |
| −6 | naive: local_z center-sum     | 0.999 | 4.27 | 0.98 |
| −6 | naive: temporal_diff center-sum | 0.556 | 0.23 | 0.04 |
| −6 | linear probe: raw only        | 1.000 | 4.67 | 0.99 |
| −6 | linear probe: honest          | 0.999 | 4.57 | 0.99 |

**What we learned.**
1. **The hypothesis was wrong** — and that is the finding. Even the honest raw
   center-sum reaches Pd ≈ 1.0 at −3/−6 dB. Removing the clairvoyant channels
   did *not* open headroom. The saturation is not a channel artifact; it is the
   **regime**: oracle-centering + 32-frame integration makes the post-stacking
   SNR huge, so per-frame −3/−6 dB is trivially easy.
2. `temporal_diff` alone is at chance (AUC ≈ 0.53) — frame differencing removes
   the slowly-varying target signal. Not useful as a standalone readout here.

**Decision / next.** The blocker is SNR, not channels. → **Exp 03**: generate and
sweep much lower SNR cells to find the band with dynamic range.

---

### Exp 03 — SNR sweep to find dynamic range

**Date:** 2026-05-30 · **Noise:** awgn · **Cells:** −3..−21 dB · **Net:** none
**Script:** `exp03_snr_sweep.py` · **Setup:** generated −9..−21 dB cells via
`python -m collapseprobe.dataset --noise-model awgn --snr=-9,-12,-15,-18,-21`
(200/class, ~11 s/cell) · **Run:** `python -m collapseprobe.exp03_snr_sweep`

**Hypothesis.** d′ ∝ √(SNR-power) (Kay; Richards), so from the −6 dB optimum
(d′ 5.47) the optimum should fall to ≈1–4 over −9..−21 dB, crossing out of
saturation and opening a gap to the honest readout.

**Result.**

| SNR | OPT AUC | OPT d′ | OPT Pd | raw d′ | raw Pd | probe d′ | probe Pd |
|----:|--------:|-------:|-------:|-------:|-------:|---------:|---------:|
| −3  | 1.000 | 7.63 | 1.00 | 7.00 | 1.00 | 6.24 | 1.00 |
| −6  | 1.000 | 5.47 | 1.00 | 4.83 | 0.99 | 4.57 | 0.99 |
| −9  | 0.998 | 3.87 | 0.94 | 3.39 | 0.91 | 3.22 | 0.84 |
| −12 | 0.972 | 2.78 | 0.67 | 2.49 | 0.68 | 2.16 | 0.70 |
| −15 | 0.926 | 2.05 | 0.51 | 1.86 | 0.41 | 1.74 | 0.20 |
| −18 | 0.811 | 1.26 | 0.24 | 1.10 | 0.15 | 1.01 | 0.16 |
| −21 | 0.766 | 1.05 | 0.18 | 0.98 | 0.16 | 1.03 | 0.14 |

**What we learned.**
1. **Dynamic range found: −9 to −15 dB.** Optimum Pd runs 0.94 → 0.51 there with
   clear room below. This is the band to put the network in the loop.
2. **The ruler is exact.** Predicted optimum d′ from √SNR scaling vs measured:
   −9 3.87/3.87, −12 2.74/2.78, −15 1.94/2.05, −18 1.37/1.26, −21 0.97/1.05.
   Strong evidence the matched-filter ceiling is implemented correctly.
3. **Caveat that shapes the thesis (now a TODO):** with oracle centering the
   honest raw readout tracks the optimum to within ~0.3–0.5 d′ across the whole
   sweep. In clean AWGN the *input representation* barely loses signal, so a
   dramatic input-side "collapse" is not expected here. The losses worth
   localizing must live in (a) the network's internal stages, (b) the IR3D
   correlated-noise regime where a plain readout is genuinely suboptimal — which
   is exactly where the whitening matched filter becomes the right ceiling.

**Decision / next.** Two tracks, both grounded by this sweep:
- **Put the frozen network in the loop** at −9..−15 dB and measure per-stage
  detectability vs the optimum (the actual Q1 figure).
- **Build the IR3D whitening matched filter** and rerun the sweep there; expect
  the honest readout to fall further below the (correct) ceiling than in AWGN.

**Track chosen (2026-05-30): IR3D whitening matched filter.** Rationale: Exp 03
showed the AWGN input barely loses signal vs the optimum, so the genuine gap
lives in the correlated/fixed-pattern (IR3D) regime where the plain MF is *not*
the optimum. The in-scope detector checkpoint was cleaned up (only `spikeNN/`
checkpoints remain, out of scope), so "network in the loop" would need a retrain
first — deferred. → Exp 04.

---

### Exp 04 — the right ceiling for IR3D: whitening vs. plain matched filter

**Date:** 2026-05-30 · **Noise:** ir3d (regenerated inline) · **Cells:** −9, −12, −15 dB · **Net:** none
**New module:** `whitening_mf.py` · **Script:** `exp04_whitening_vs_plain.py`
**Run:** `python -m collapseprobe.exp04_whitening_vs_plain` (150/class, ~11 s total)

**Hypothesis.** Under the IR 3-D noise model the noise is correlated (fixed
pattern + stripes + flicker), so the plain matched filter is suboptimal; the
whitening MF `w = C⁻¹s` (Kay §5) is the true optimum and should give higher d′,
by a margin that grows with the fixed-pattern strength. Sanity: with only the
white floor, the two must coincide.

**Method.** `whitening_mf.py` applies the exact IR3D covariance as a structured
operator (white floor + per-pixel FPN + column/row stripes + flicker, read
straight from `IR3DNoiseConfig`) and solves `C w = s` by conjugate gradient — no
131k×131k matrix is ever formed. The signal template reuses
`npceiling … render_unit_template` (physics not forked). Regenerate paired
pos/neg IR3D frames; score each with the plain MF (`cmf_score`), the whitening MF
(`wᵀx`, `w` solved once per trajectory and applied to both members of the pair),
and an honest location-only temporal integrator.

**Module validation (standalone, before the experiment).**
- White-noise limit (`σ_vh=…=0`): CG converges in **1 iter**, `w = s/σ²` to 4e-16.
- Operator symmetry `⟨Cu,v⟩=⟨u,Cv⟩` to 4e-13 (valid covariance).
- IR3D solve: **6 iters, 4 ms**, residual 3e-15 (well-conditioned, exact).

**Result.**

*Sanity — pure white floor (whiten must == plain):*

| SNR | plain d′ | whiten d′ | gain | CG it |
|----:|---------:|----------:|-----:|------:|
| −9  | 3.70 | 3.70 | 1.000× | 1 |
| −12 | 2.60 | 2.60 | 1.000× | 1 |
| −15 | 1.81 | 1.81 | 1.000× | 1 |

*IR3D default (σ_vh=0.30 FPN, σ_v=0.12, σ_h=0.08, σ_t=0.05; scintillation on):*

| SNR | plain d′ / AUC / Pd | whiten d′ / AUC / Pd | honest d′ / AUC / Pd | d′ gain |
|----:|--------------------:|---------------------:|---------------------:|--------:|
| −9  | 2.46 / 0.960 / 0.71 | 3.36 / 0.994 / 0.89 | 2.24 / 0.944 / 0.68 | 1.368× |
| −12 | 1.73 / 0.890 / 0.47 | 2.39 / 0.956 / 0.64 | 1.56 / 0.867 / 0.39 | 1.382× |
| −15 | 1.21 / 0.803 / 0.30 | 1.69 / 0.881 / 0.37 | 1.08 / 0.779 / 0.24 | 1.394× |

*IR3D heavy fixed-pattern (3× FPN: σ_vh=0.60, σ_v=0.25, σ_h=0.18):*

| SNR | plain d′ / AUC / Pd | whiten d′ / AUC / Pd | honest d′ / AUC / Pd | d′ gain |
|----:|--------------------:|---------------------:|---------------------:|--------:|
| −9  | 1.50 / 0.869 / 0.41 | 3.31 / 0.993 / 0.86 | 1.40 / 0.856 / 0.21 | 2.208× |
| −12 | 1.06 / 0.783 / 0.26 | 2.36 / 0.954 / 0.57 | 0.97 / 0.767 / 0.13 | 2.241× |
| −15 | 0.74 / 0.706 / 0.17 | 1.67 / 0.880 / 0.29 | 0.67 / 0.692 / 0.04 | 2.264× |

**What we learned.**
1. **The whitening MF is correct** (sanity row is exact) and **fast** (6 CG iters).
2. **The plain MF is the wrong ceiling under IR3D** — it understates the optimum
   by **1.37–1.39×** (default) and **2.2–2.3×** (heavy FPN) in d′. The dataset's
   stored `cmf_scores` cannot be used as the optimum for IR3D cells.
3. **The gain scales with fixed-pattern strength**, as the physics demands:
   whitening's whole job is to reject the fixed pattern temporal averaging leaves
   behind. At −9 dB heavy-FPN the optimum rises from Pd 0.41 (plain) to 0.86
   (whiten).
4. **Dynamic range, finally.** Unlike AWGN (Exp 03, honest ≈ optimum), under IR3D
   the honest location-only integrator sits clearly **below** the whitening
   optimum (−9 dB heavy-FPN: 0.21 vs 0.86 Pd). This is the headroom a "where is
   detectability lost" study needs — and the IR3D regime is where it lives.

**Decision / next.** The ruler for the realistic regime is ready. Two follow-ups:
- **Persist it:** wire `wmf_score` into `dataset.build_split_for_snr` so IR3D
  cells store `wmf_scores`, and generate the low-SNR IR3D cells (−9..−15 dB).
- **Then the IR3D sweep / network:** repeat Exp 03's optimum-vs-readout sweep
  against the *correct* (whitening) ceiling, and/or put the (to-be-retrained)
  network in the loop at these SNRs, where there is now real room to lose signal.

---

### Exp 05 — retrain the in-scope detector (the frozen network for Q1)

**Date:** 2026-05-30 · **Noise:** ir3d · **Train SNRs:** −3..−15 dB · **Net:** TrackletRecurrentUNet
**Script:** `train_detector.py` · **Run:** `python -m collapseprobe.train_detector`
**Checkpoint:** `collapseprobe/detector_ckpt.pt`

**Why.** Q1 ("where does the network lose detectability vs the optimum?") needs a
frozen trained detector. The old demo checkpoint was cleaned up; the SNN is
abandoned (user: never worked, out of scope). So we retrain the in-scope
`TrackletRecurrentUNet` (per-frame U-Net encoder → ConvLSTM bottleneck → decoder,
`track_logit` head; base=16, bottleneck=32, 5-channel input).

**Method.** Train on the SAME oracle-centered crop tubes we will probe (charter
Sec. 7 — isolate the network from the front-end), on IR3D where the whitening
optimum has real headroom (Exp 04). Train/val pools share the eval cells' sensor
(same fixed pattern) but use disjoint data seeds (70001/90002 vs the eval rng),
so no sequence leaks across train/val/probe. Objective: plain BCE on `track_logit`
(the "target present?" signal the probe measures); no architecture tuning
(out of scope). 1600 train / 800 val tubes pooled over 5 SNRs, AdamW lr 1e-3,
30 epochs, best-val-AUC checkpoint, MPS. Total 652 s.

**Metric note.** AUC and Pd@FAR are rank-based, so unaffected by the output
sigmoid; **d′ must be computed from the logits** (the sigmoid saturates → bogus
d′). The table below uses logit d′, evaluated on the **200/class eval cells**
(disjoint from training) — the trustworthy version.

**Result — frozen detector vs. whitening optimum (eval cells, 200/class):**

| SNR | detector d′ / AUC / Pd | optimum d′ / AUC / Pd | AUC gap |
|----:|-----------------------:|----------------------:|--------:|
| −3  | 1.38 / 0.941 / 0.48 | 6.60 / 1.000 / 1.00 | +0.059 |
| −6  | 0.89 / 0.835 / 0.14 | 5.02 / 1.000 / 1.00 | +0.165 |
| −9  | 0.63 / 0.729 / 0.09 | 3.84 / 0.996 / 0.94 | +0.266 |
| −12 | 0.74 / 0.728 / 0.14 | 2.54 / 0.963 / 0.46 | +0.236 |
| −15 | 0.42 / 0.626 / 0.07 | 1.86 / 0.908 / 0.10 | +0.282 |

**What we learned.**
1. **The Q1 premise holds, measured.** The whitening optimum keeps AUC ≈ 1.0 down
   to −9 dB; the detector falls to 0.73 there and the **gap widens as SNR drops**
   (+0.06 → +0.28 AUC). The signal is present (optimum finds it) but the network
   discards it — the thing Q1 sets out to localize.
2. **There is a real input→output drop to localize.** The input tube carries the
   near-optimal `evidence` channel (AUC ≈ 1.0, Exp 01), yet the detector output is
   only AUC 0.73 at −9 dB. So detectability is lost *inside* the network — the
   per-stage probe can now find where.
3. The detector overfits mildly on 1600 tubes (train loss → 0.03, noisy val);
   best-val checkpointing rescues a representative model. Good enough as a frozen
   subject (charter: detector tuning out of scope).

**Decision / next.** Build the per-stage probe (Exp 06): freeze this checkpoint,
hook the 8 stages (input → enc1 → enc2 → enc3 → ConvLSTM → dec2 → dec1 → logit),
take a best-linear-probe detectability at each (same instrument as Exp 01),
overlay the whitening optimum, and read off the cliff. That is signature
figure #2 (detectability vs depth) and the heart of the thesis.

---

### Exp 06 — detectability vs. depth (the Q1 cliff figure)

**Date:** 2026-05-30 · **Noise:** ir3d eval cells · **SNRs:** −6..−15 dB · **Net:** frozen Exp-05 detector
**New module:** `net_probe.py` · **Script:** `exp06_layerwise.py` · **Figure:** `collapseprobe/fig_cliff.png`
**Run:** `python -m collapseprobe.exp06_layerwise`

**Hypothesis.** Push the eval tubes through the frozen detector; a Fisher-LDA probe
on pooled features at each stage (same instrument as Exp 01) estimates the best
linear detectability surviving there. Somewhere it should drop below the
whitening optimum — that stage is the bottleneck.

**Method.** Hook `enc1, enc2, enc3, ConvLSTM, dec2, dec1`; pool each activation to
per-tube features (per-channel mean, max, center-region time-sum), plus `input`
and the `logit`. Stratified-half train/test; probe AUC on the held-out half;
overlay the optimum (`wmf_scores`).

**Result — per-stage linear-probe AUC (held-out):**

| stage | −6 dB | −9 dB | −12 dB | −15 dB |
|-------|------:|------:|-------:|-------:|
| input    | 0.998 | 0.964 | 0.849 | 0.824 |
| enc1     | 0.993 | 0.925 | 0.830 | 0.748 |
| enc2     | 0.957 | 0.822 | 0.692 | 0.547 |
| enc3     | 0.900 | 0.724 | 0.625 | 0.467 |
| convlstm | 0.849 | 0.718 | 0.620 | 0.527 |
| dec2     | 0.870 | 0.779 | 0.657 | 0.572 |
| dec1     | 0.969 | 0.868 | 0.771 | 0.657 |
| logit    | 0.839 | 0.721 | 0.682 | 0.660 |
| **optimum** | **1.000** | **0.994** | **0.948** | **0.918** |

**What we learned — a clean U-shape, and a specific architectural culprit.**
1. **Input ≈ optimum, then erosion through the encoder.** input→enc1→enc2→enc3
   falls monotonically (−9 dB: 0.964→0.925→0.822→0.724) as the spatial map is
   max-pooled 31→15→7 px. This is "early downsampling erases the few-pixel
   target" (charter lit: Remote Sensing 13(18):3608), measured.
2. **Temporal integration does not help detection here.** enc3→ConvLSTM is flat
   or slightly *down* (−6 dB: 0.900→0.849). The ConvLSTM bottleneck is the trough,
   not a recovery — striking, since temporal integration is the thesis's theme.
3. **The U-Net decoder recovers detectability** (ConvLSTM→dec2→dec1, −9 dB:
   0.718→0.779→0.868): the skip connections re-inject the high-resolution encoder
   features the bottleneck dropped.
4. **…but the track head throws the recovery away.** The detection head reads the
   post-ConvLSTM 7×7 bottleneck (`bottleneck_flat`), **not** the decoder (which
   only feeds the heatmap head). So `logit` ≈ `convlstm` (−9 dB: 0.721 vs 0.718),
   far below `dec1` (0.868). The decision is bottlenecked at the downsampled
   ConvLSTM features even though the network has reconstructed the signal
   downstream.

**This localizes the loss (C1/C2) and hands us a concrete repair (C3).**
The detectable signal is present (optimum AUC ≈0.99 at −9 dB) and even *survives
inside the net* to dec1 (0.87), but the detection path bottlenecks it at the 7×7
ConvLSTM stage. **C3 hypothesis:** feed high-resolution decoder/skip features into
the track head (or reduce downsampling on the detection path); the optimum bounds
the recoverable gain.

**Caveats.** (a) Detector is mildly overfit (Exp 05) — the U-shape is
architecture-driven so likely robust, but confirm with a better detector + a
second probe seed before it's a thesis claim. (b) Linear-probe detectability can
be non-monotone with depth (the dec1 bump is real re-injected signal, not a DPI
violation). (c) Pd@FAR omitted here — AUC is the stable per-stage summary at
n=200/class.

**Decision / next.** (i) Robustness pass: retrain with more data/regularization,
re-run Exp 06 with a 2nd split seed, confirm the U-shape. (ii) Prototype the C3
repair (decoder-fed or reduced-downsampling track head), retrain, and re-measure
how much of the optimum it recovers — the before/after of signature figure #3.

---

### Exp 06b — robustness of the cliff (error bars + a better detector)

**Date:** 2026-05-30 · **Noise:** ir3d eval cells · **SNRs:** −6..−15 dB
**Scripts:** `exp06_layerwise.py` (now multi-split), `train_detector.py` (CLI'd)
**Detectors:** v1 `detector_ckpt.pt` (1600 tubes) and v2 `detector_ckpt_v2.pt`
(3000 tubes, weight_decay 3e-4, init seed 4242, best-val AUC 0.842)
**Figures:** `fig_cliff.png` (v1), `fig_cliff_v2.png` (v2)

**Why.** Exp 06's U-shape rested on a mildly-overfit detector. Before it becomes a
C1/C2 claim, confirm it survives (a) the probe split and (b) an independently
trained, less-overfit detector. The v2 retrain also gives the clean baseline the
C3 repair will need for a fair before/after.

**Method.** (1) Re-run the per-stage probe over 5 train/test splits → mean±std per
stage. (2) Retrain v2 (≈1.9× data, more regularization, different seed) → it is a
genuinely better detector (per-SNR AUC up vs v1, gap to optimum intact). (3) Run
the multi-split probe on v2 and compare the curve.

**Result — per-stage probe AUC at −9 dB (mean over 5 splits), v1 vs v2:**

| stage | v1 | v2 |
|-------|---:|---:|
| input    | 0.967 | 0.967 |
| enc1     | 0.937 | 0.947 |
| enc2     | 0.809 | 0.792 |
| enc3     | 0.744 | 0.776 |
| convlstm | 0.679 | **0.794** |
| dec2     | 0.763 | 0.760 |
| dec1     | 0.870 | 0.850 |
| logit    | 0.728 | **0.812** |
| optimum  | 0.995 | 0.995 |

Per-stage error bars are small (±0.01–0.03) for both detectors → the curve shape
is not probe-split noise. In v2 the **largest single drop is `enc1→enc2` at every
SNR** (Δ = +0.09/+0.16/+0.19/+0.10 over −6/−9/−12/−15 dB).

**What we learned.**
1. **Robust finding (both detectors):** input is near-optimal, then detectability
   falls off a **single cliff at the first max-pool (`enc1→enc2`, 31→15 px)** and
   stays roughly flat afterward. The loss is *front-loaded* in early spatial
   downsampling — the "downsampling erases the few-pixel target" mechanism (lit:
   Remote Sensing 13(18):3608), now localized to the **first** pool.
2. **v1 artifacts that did NOT replicate:** the deep **ConvLSTM trough**
   (convlstm 0.68→0.79) and the **"track head discards the decoder recovery"**
   story (logit 0.73→0.81, now level with dec1). These were products of the
   overfit v1 detector, not the architecture. The robustness pass caught them
   before they became claims — exactly its job (charter guardrail 1).
3. The gap to the optimum is intact and large in the better detector (v2 −9 dB:
   logit 0.81 vs optimum 0.995), so Q1's premise stands.

**Decision / next.** C1/C2 now reads cleanly: *the detector loses faint-target
detectability primarily at the first spatial downsampling, while the optimum
retains it.* **Revised C3:** replace/soften that first downsample with a
signal-preserving reduction (LDW-Pooling / anti-aliased / learned), retrain, and
re-measure how much of the optimum it recovers — the before/after of signature
figure #3, now built on the v2 baseline.

---

### Exp 07 — the C3 repair: average-pool vs. max-pool

**Date:** 2026-05-30 · **Noise:** ir3d eval cells · **SNRs:** −3..−15 dB
**Script:** `exp07_repair.py` · **Figure:** `collapseprobe/fig_repair.png`
**Detectors:** `detector_ckpt_v2.pt` (max-pool baseline) vs
`detector_ckpt_avgpool.pt` (avg-pool; trained `--pool avg`, identical data/seed/reg)

**Hypothesis.** Exp 06b localized the loss to the first encoder downsample.
Max-pooling a mostly-noise 2×2 patch keeps the upward-biased maximum → raises the
noise floor → crushes the faint target's contrast, while the optimum *integrates*.
So replacing the encoder max-pools with **average-pooling** (same 2× reduction,
noise variance ÷4 instead of a biased max) should recover detectability. Held
everything else fixed (data, seed 4242, weight_decay 3e-4) so the effect is
attributable to the pooling op alone.

**Result — detector output AUC vs. the optimum (eval cells), and % of the gap recovered:**

| SNR | max-pool | avg-pool | optimum | gap recovered |
|----:|---------:|---------:|--------:|--------------:|
| −3  | 0.985 | 0.993 | 1.000 | 55% |
| −6  | 0.932 | 0.973 | 1.000 | 61% |
| −9  | 0.805 | 0.881 | 0.996 | 40% |
| −12 | 0.796 | 0.757 | 0.963 | −24% |
| −15 | 0.649 | 0.696 | 0.908 | 18% |

Per-stage probe (avg-pool) confirms the mechanism: the **enc1→enc2 cliff
roughly halved** (−9 dB Δ 0.155→0.100; −6 dB 0.092→0.048), and every downstream
stage lifted (−9 dB logit 0.812→0.886). On the training val pool the avg-pool
detector was uniformly better (d′ at −9 dB 1.10→1.87, +70%; best val AUC
0.842→0.877).

**What we learned.**
1. **C3 is a positive result.** A targeted, theory-motivated repair at the
   localized bottleneck recovers ~40–60% of the detector-vs-optimum gap at the
   SNRs that matter (−3..−9 dB), bounded by the optimum (a residual gap remains,
   widening at low SNR — the loss is not *entirely* the pooling).
2. **The mechanism is confirmed**, not just the outcome: the very stage we
   blamed (the first downsample) is where the cliff shrank.
3. **Caveat — single run per condition.** −12 dB regressed (−24%), yet the same
   detector's val pool showed −12 *improving* (0.767→0.801). That inconsistency
   is run-to-run / sampling noise, not a real reversal. The effect at −3/−6/−9 is
   large and consistent; −12/−15 are within noise.

**Decision / next.** The diagnose→repair loop is closed and positive. Before
figure #3 is final: **multi-seed firming** — 2–3 training seeds per condition,
report mean±std of the recovered gap (smooths the −12 noise), exactly as Exp 06b
firmed up Exp 06. After that, Q3/C4 (the temporal-integration limit vs the
fixed-pattern floor) is the remaining contribution.
