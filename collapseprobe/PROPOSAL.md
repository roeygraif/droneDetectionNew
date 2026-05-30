# collapseprobe — thesis-direction proposal

**Where does the signal die?**
Localizing — and then repairing — the network stage at which a deep low-SNR
detector loses a faint target, measured against the Neyman–Pearson optimum.

_Draft proposal — 2026-05-29. This is a standalone idea kept in its own
directory; the rest of the repo (`src/`, `npceiling/`, `spikeNN/`, `scripts/`)
is read-only reference for this module._

---

## 1. One sentence

At low signal-to-noise ratio (SNR) a deep detector throws the target away
somewhere between its input and its decision. Because our data is **synthetic**,
we can compute the optimal detector (the matched filter) and measure, **stage by
stage, exactly where** the target stops being recoverable — then redesign that
one stage and **prove** the lost detectability comes back.

Acronyms used below: UAV = unmanned aerial vehicle; P_d = probability of
detection; FAR = false-alarm rate; NP = Neyman–Pearson; TBD = track-before-detect;
CMF = clairvoyant matched filter; GLRT = generalized likelihood-ratio test;
DPI = data-processing inequality; IRSTD = infrared small-target detection.

---

## 2. Motivation — a gap nobody can close on real data

Deep detectors for dim / low-SNR targets reliably **underperform the theoretical
optimum**, but on real sensor data the optimum is *uncomputable* — you do not
know the true signal-plus-noise model. So the field can only ever say "we
achieved P_d = X" and compare one network against another. It **cannot** say:

1. how far X sits below the best physically possible detector, nor
2. **where inside the network** that shortfall is actually created.

Our synthetic generator gives the **exact** signal (Gaussian-PSF point target)
and noise (additive white Gaussian) model. The matched filter is therefore the
provably NP-optimal detector (Marcum 1948; Kay 1998, Vol. II). The existing
`npceiling/` module already computes that ceiling **at the detector's output**.
This proposal pushes the same ceiling-referenced measurement **inside** the
network, to the intermediate stages.

---

## 3. The idea in plain terms (the "photocopier")

Think of the network as a chain of photocopiers: each copy is smaller and
blurrier than the last. A bright object survives the whole chain; a **faint
drone (a few pixels)** may survive copy 1, weaken at copy 2, and vanish by
copy 3. Today we only look at the final copy and ask "is the drone there?". We
never check the in-between copies. The proposal: **check every copy.**

In signal-processing terms, each "copy" is a **collapse** — a dimensionality
reduction. The matched filter is the *optimal* collapse: it reduces the entire
observation to a single scalar that is a **sufficient statistic** (zero loss of
detection information). Any other collapse can only be lossy. The governing law
is the **data-processing inequality**: for the chain
`frames → stage₁ → stage₂ → … → decision`, the target-relevant information
`I(stageₖ ; target-present)` is **non-increasing** with depth. So detectability
traces a descending staircase through the network, and the **bottleneck is the
single step with the steepest drop.** We measure the staircase and find that step.

A corollary worth stating up front: because of the DPI, **the most this whole
research program can ever buy is the gap between the network and the matched-filter
ceiling** that `npceiling/` already plots. No new collapse can exceed the optimum.
That is a feature — it bounds the contribution and makes every claim falsifiable
against a number we already compute.

---

## 4. Method

### 4.1 Where we tap (real architecture — `src/models/recurrent_unet.py`)

The trained detector is a per-frame U-Net encoder + a ConvLSTM bottleneck
(temporal integration) + heads. The natural tap points, input → output:

| # | Tap | What it is | Collapse type |
|---|---|---|---|
| 0 | input | the 5 evidence channels (raw) | — (reference) |
| 1 | after `pool1` | first 2× spatial downsample | **spatial collapse** |
| 2 | after `pool2` | second 2× spatial downsample | **spatial collapse** |
| 3 | `enc3` | per-frame bottleneck, still frame-by-frame | — |
| 4 | after `ConvLSTM` | frames merged across the T-frame tube | **temporal collapse** |
| 5 | pooled track feature | the 128-vector before the classifier | final collapse |
| 6 | `track_logit` | the model's actual decision today | — (= current P_d) |

The two spatial collapses (1, 2), the temporal collapse (4), and the final
collapse (5) are exactly the operations the "collapse it differently" intuition
is about.

### 4.2 How we measure detectability at a tap

1. **Freeze** the trained detector. Run many target-present and target-absent
   crop-tubes through it at a fixed SNR; cache the activations at each tap.
2. At each tap, fit a **fixed-family probe** (linear / logistic classifier — the
   standard linear-probe diagnostic, Alain & Bengio 2016) to separate
   present vs absent using *only* that tap's features.
3. Score the probe with the **same metric the project already uses** —
   `P_d @ FAR = 1/100` from `src/training/metrics.py` — plus the
   detection-theoretic **deflection** `d′ = (μ₁ − μ₀)/σ`.
4. The probe family is **held constant across all taps**, so differences reflect
   the *representation*, not the probe. Probe detectability is a **lower bound**
   on the information actually present (a stronger probe could do better); we
   report it as such, and the **cross-tap drop** is the quantity of interest.

### 4.3 Reference lines

CMF and GLRT from `npceiling/` are the ceiling. Tap-0 detectability is what a
probe can pull from the raw evidence channels (should sit just under the matched
filter). The gap from tap-0 down to tap-6 is the network's self-inflicted loss.

### 4.4 Headline figure

**Detectability vs network depth, with the NP ceiling on top.** The location of
the cliff *is* the result.

---

## 5. Research questions & a falsifiable hypothesis

- **RQ1 (localization).** Is there an identifiable stage where detectability
  collapses, and where is it?
- **RQ2 (mechanism).** Is the binding loss the **spatial** collapse (pooling) or
  the **temporal** collapse (ConvLSTM) — and how does that depend on SNR and on
  sequence length N?
- **RQ3 (repair).** Does a targeted redesign of the killer stage recover
  detectability toward the ceiling, and is the recovered gain bounded by the
  ceiling exactly as the DPI predicts?
- **H1 (falsifiable).** At very low SNR the binding loss is the **early spatial
  pooling** — a 2–3-pixel target is destroyed by max-pool before the temporal
  stage can integrate it — while as **N grows** the **temporal collapse**
  (ConvLSTM) becomes the binding bottleneck. Either part can be refuted by the
  Stage-1 measurement.

---

## 6. Plan

- **Stage 0 — setup (~30 min).** Retrain a baseline ConvLSTM detector checkpoint
  (the Run-M checkpoint is no longer on disk). Reuse `scripts/train_demo.py`.
- **Stage 1 — the pilot (make-or-break).** Build the per-tap detectability
  measurement on top of `eval_tracklet_detector.py`; produce the depth-vs-ceiling
  figure at **−3 dB** (clean) and **−6 dB** (larger gap). **Decision gate:** a
  clear cliff → proceed; a smooth decline or an at-input cliff → that is itself a
  result (see §11), regroup.
- **Stage 2 — repair & prove.** Redesign the collapse at the identified stage;
  re-run the same figure; show the cliff flattens and the output climbs toward
  the ceiling.
- **Stage 3 — generalize.** Sweep (SNR, N); map where the killer stage sits and
  how the repair's payoff scales. **This fuses with the temporal-horizon
  candidate** — the ConvLSTM tap *is* the temporal collapse.

**Why the pilot is safe.** `npceiling/` already shows that at −3 dB the target is
recoverable (CMF = 1.00, realistic GLRT = 0.87) yet the detector reaches only
0.65. The information is provably present at the input and provably gone at the
output, so a cliff **must** exist somewhere inside — the pilot only localizes it.

---

## 7. Has this been done? (novelty verification)

Three literature-search rounds (2026-05-28/29). The honest verdict: **the
ingredients all exist; the specific composition and application do not.**

| Ingredient of this plan | Prior art | Status |
|---|---|---|
| Probe each layer for decodable content | Linear classifier probes (Alain & Bengio 2016); information-plane analysis (Shwartz-Ziv & Tishby 2017) | **established tool** |
| "A CNN is a bank of matched filters" | Matched-filtering perspective tutorials (Stanković & Mandić 2021; 2022) | textbook framing |
| Build a NN that *approximates* the NP detector | NP neural classifiers; MLP-approx-NP radar detectors | done (different goal) |
| Matched-filter-vs-deep performance benchmark | GW astronomy (Gabbard et al. 2018); our own `npceiling/` at the **output** | done at the output |
| "Better collapse" for dim targets (wavelet / pooling / Hough-TBD) | LDW-Pooling; DWTFreqNet; 3D-Hough TBD; reduced-downsampling nets; IB-for-small-target | **crowded** |

**The gap (= the contribution).** Using layer-wise probing to measure the
**detectability** (P_d / d′) of a faint *moving point target* at each stage
**against a computable Neyman–Pearson / matched-filter ceiling**, to **localize
the exact architectural stage** where low-SNR detectability collapses — and then
**repair that stage and prove the recovery** — on synthetic data where the
optimum is computable. Targeted searches for this combination returned nothing;
the search tools themselves repeatedly noted the absence ("results don't
specifically address layer-wise signal detectability loss," "rather than
identifying which layers create bottlenecks in detection performance").

This is the safest kind of novelty for an M.Sc.: **known tools, a new
composition, a new application, enabled by a capability the field structurally
lacks on real data** (a computable optimum).

---

## 8. What previous research says (related work)

- **(a) The optimum.** Matched filter / GLRT are NP-optimal for known-signal /
  composite-hypothesis detection in AWGN (Marcum 1948; Kay 1998). This is the
  ceiling; `npceiling/` computes it for the generator.
- **(b) The diagnostic tool.** Probing intermediate layers (Alain & Bengio 2016)
  and tracking mutual information per layer / the "information plane"
  (Shwartz-Ziv & Tishby 2017; information-bottleneck method, Tishby–Pereira–Bialek
  1999) are standard for *generic* representation analysis — about input/label
  information, **not** about detectability vs a known optimal detector.
- **(c) Deep-vs-optimal benchmarking.** Gravitational-wave astronomy routinely
  benchmarks deep detectors against matched filtering (Gabbard et al. 2018) — but
  as an **output** comparison, not a per-stage internal diagnosis, and not for
  video point targets.
- **(d) Small/dim-target collapses.** A large literature attacks "small objects
  vanish under downsampling" with new front-ends — wavelet/contourlet transforms
  (DWTFreqNet), learnable wavelet pooling (LDW-Pooling), reduced-downsampling
  networks, and motion-trajectory collapses (Hough/Radon TBD — which is exactly
  our accumulator front-end). All report *output* P_d / mAP; none measure *where*
  in the network the signal is lost relative to the optimum.
- **(e) The temporal connection.** RNN temporal-learnability theory (Livi 2025,
  arXiv:2512.05790) and DNN-vs-NP gap bounds for single-sample detection (Anders,
  Kothari & Buehrer 2025, arXiv:2512.13542) motivate the N-dependence in RQ2/RQ3
  and are the link to the temporal-horizon candidate.

---

## 9. Why we can do this and the field can't

The whole program rests on two things only the synthetic setup provides:
the **computable optimum** (exact signal + noise → matched filter) and **known
ground truth at every frame**. Together they let us put a meaningful reference
line under *every* intermediate stage, and the DPI gives a hard, pre-computed
ceiling on the payoff. Real-world IRSTD groups cannot do this for the same reason
their problem is hard: they don't know the true signal.

---

## 10. Expected contributions

1. A **diagnostic method** — and the new figure "detectability vs network depth
   vs the NP ceiling" — for any low-SNR detector on synthetic data.
2. An **empirical map of where low-SNR signal dies** in a deep TBD detector, as a
   function of (SNR, N): spatial-collapse-limited vs temporal-collapse-limited.
3. A **targeted, ceiling-referenced repair** of the binding stage, with a
   proof-of-recovery (the gain is bounded by, and approaches, the computed ceiling).

---

## 11. Honest risks & caveats

- **Probe strength.** A fixed-family probe gives a *lower bound* on recoverable
  information; a different probe could read more. Mitigate by fixing the family
  across taps (fair comparison) and optionally reporting a strong-probe variant.
  The cross-tap *drop*, not the absolute value, is the claim.
- **The cliff might be at the input.** At very low SNR the matched filter itself
  fails (e.g. ≤ −9 dB, where GLRT ≈ 0) — then the loss is **information-limited,
  not architectural**, and no collapse can help. That is *also* a result. We
  therefore run the pilot in the **GLRT-positive band** (−3 / −6 dB), where the
  optimum provably succeeds, so any internal cliff is genuinely the network's.
- **The repair might not help.** If fixing the stage doesn't recover
  detectability, the loss is intrinsic to that operation — a finding, not a
  failure; redirect to the next-steepest stage.
- **Scope.** AWGN-only, synthetic-only (consistent with `npceiling/`). Clutter,
  distractors, and real-sensor transfer are explicitly out of scope / future work.

---

## 12. Relationship to the existing project

Builds directly on the `src/` detector and the `npceiling/` ceiling. It
**subsumes the spatial half** of the earlier "collapse the image differently"
idea (turning a crowded architecture bake-off into a measured, ceiling-referenced
diagnosis) and **fuses with the temporal-horizon candidate** (the ConvLSTM tap
*is* the temporal collapse). It is therefore not a fourth competing direction but
the **instrument** that ties the existing threads — detector, NP ceiling,
temporal horizon — into one headline: *localizing, in SNR and in network depth,
exactly where deep low-SNR detection falls short of the Neyman–Pearson optimum,
and why.*

---

## References

1. Marcum, J. I. (1948). *A Statistical Theory of Target Detection by Pulsed Radar.* RAND.
2. Kay, S. M. (1998). *Fundamentals of Statistical Signal Processing, Vol. II: Detection Theory.* Prentice Hall.
3. Tishby, N., Pereira, F., Bialek, W. (1999). *The Information Bottleneck Method.* — overview: https://en.wikipedia.org/wiki/Information_bottleneck_method
4. Alain, G., Bengio, Y. (2016). *Understanding intermediate layers using linear classifier probes.* https://arxiv.org/abs/1610.01644
5. Shwartz-Ziv, R., Tishby, N. (2017). *Opening the Black Box of Deep Neural Networks via Information.* https://arxiv.org/abs/1703.00810
6. Gabbard, H. et al. (2018). *Matching matched filtering with deep networks for gravitational-wave astronomy.* https://arxiv.org/abs/1712.06041
7. Stanković, L., Mandić, D. (2021). *Convolutional Neural Networks Demystified: A Matched Filtering Perspective.* https://arxiv.org/abs/2108.11663
8. *Generalized Approach to Matched Filtering using Neural Networks* (2021). https://arxiv.org/abs/2104.03961
9. *Learnable Discrete Wavelet Pooling (LDW-Pooling) for CNNs* (2021). https://arxiv.org/abs/2109.06638
10. *DWTFreqNet: Infrared Small Target Detection via Wavelet-Driven Frequency Matching* (2024). https://www.researchgate.net/publication/395431238
11. *A 3-D Hough-Transform-Based Track-Before-Detect Technique* (2019). Sensors 19(4):881. https://www.mdpi.com/1424-8220/19/4/881
12. *Learning Adjustable Reduced Downsampling Network for Small Object Detection* (2021). Remote Sensing 13(18):3608. https://www.mdpi.com/2072-4292/13/18/3608
13. *Small-target detection against information attenuation (PGI + Mamba)* (2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC11991474/
14. Livi, L. (2025). *Temporal learnability window in recurrent neural networks.* arXiv:2512.05790 _(from earlier survey)_
15. Anders, Kothari, Buehrer (2025). *DNN-vs-Neyman–Pearson gap for single-sample detection.* arXiv:2512.13542 _(from earlier survey)_
