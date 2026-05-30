# Thesis plan — *Where Does the Signal Die?*

**Localizing and repairing the temporal-integration bottleneck in deep low-SNR
drone detection, against the Neyman–Pearson optimum.**

_M.Sc. (EE) thesis plan — 2026-05-29. Companion to `PROPOSAL.md` (the research
idea) and the existing modules `src/` (detector), `npceiling/` (optimum), and
`collapseprobe/` (this direction). Acronyms: SNR = signal-to-noise ratio;
UAV = drone; Pd/FAR = probability of detection / false-alarm rate; NP =
Neyman–Pearson (the optimal detector); TBD = track-before-detect; PSF = the
blur a point source makes; FPN = fixed-pattern (sensor "smudge") noise;
IR = infrared/thermal._

---

## 0. One-line thesis
A deep low-SNR drone detector leaves a *measurable* gap below the best detector
physically possible. Using a synthetic IR testbed where that optimum is
**computable**, this thesis localizes — in SNR, in network depth, and in number
of frames — **exactly where and why** the faint moving target is lost, **repairs**
the worst bottleneck, and maps the **temporal-integration limit** that
fixed-pattern sensor noise imposes.

This sharpens (does not replace) the original thesis question — *the limits of
temporal integration for low-SNR UAV detection* — by giving every "waterfall"
curve a provable top line and a mechanism.

---

## 1. The contribution, stated honestly
**The move we use is borrowed, and we say so.** Using the optimal detector (the
"ideal observer") as a yardstick for whether processing preserves a faint signal
is established — in medical imaging for decades, and in gravitational-wave
astronomy and RF detection for deep-vs-optimal comparisons.

**What is new is the assembly + application.** No prior work combines all of:
1. the computable optimum as a yardstick **inside** the network, layer by layer;
2. to **localize** where a faint signal leaks out, **and repair** that stage;
3. for **faint moving-target IR video** (drones), not single images / other domains;
4. tied to the **temporal-integration (frame-stacking) limit**.

Concrete deliverable contributions:
- **C1 — A diagnostic instrument:** "detectability vs network depth, against the
  optimum," for any low-SNR detector on synthetic data.
- **C2 — A localization result:** where in a deep TBD detector the low-SNR signal
  dies, as a function of (SNR, frame count N, target motion).
- **C3 — A repair + proof:** redesign the worst stage; show recovery toward the
  optimum, bounded as theory predicts.
- **C4 — A temporal-limit map:** the learnable frame-stacking horizon vs the
  classical √T limit, and how fixed-pattern noise + motion set the floor.

Enabling advantage: **only computable because the data is synthetic.** Real-world
groups can't compute the optimum, so they can't do C1–C4. This is the thesis's
structural edge, and its main scope limitation (below).

---

## 2. Positioning (closest prior work, and how we differ)
| Prior work | What they do | How we differ |
|---|---|---|
| Medical "ideal observer" | Use the optimum to check if denoising/reconstruction preserves a faint signal | They judge whole images at the **output**; we go **inside the detector, per layer**, for **moving-target video**, + repair + temporal limit |
| Transformers ≈ NP test (2026) | Link a network's internals to the optimal statistic | General **in-context learning**, not faint-target video; shows nets *can* reach the optimum, not **where they lose it**; no repair |
| GW astronomy / RF detection | Deep-vs-optimal **at the output** | Sealed-box, output-only, other domains; no layer localization, no repair |
| IR small-target "arms race" | New networks, higher leaderboard scores | **No optimum reference at all**; we explain *why/where*, not who scores higher |
| Probing / information-plane | Read information per layer | No **optimum** reference, generic tasks; we anchor each layer to the computable optimum |
| Temporal-learnability theory (Livi 2025) | Max horizon RNNs can learn (general) | Our **tool** for C4, not a competitor; we instantiate it for detection |

Honest framing for the defense: *"method borrowed from the ideal-observer tradition; **first application inside a deep detector to localize-and-repair signal loss for low-SNR moving-target video**, with the frame-stacking limit as the headline."*

---

## 3. The instrument (testbed + the ruler)
- **Synthetic IR/thermal testbed** (`src/synthetic` + `collapseprobe`): Gaussian-PSF
  point target; NVESD 3-D noise model — a random floor that averages away as √T
  **plus fixed-pattern noise that does not**; lognormal scintillation (target
  twinkle); optional colored cloud clutter. SNR defined vs the random floor for
  comparability with prior waterfalls. *Validated by 9 tests + visual preview.*
- **The ruler (optimum):** matched filter (NP-optimal under white noise) →
  **whitening matched filter** (NP-optimal under the known fixed-pattern/colored
  noise). Built in `npceiling/`; the colored-noise upgrade is Phase 2 below.
- **Scope (honest):** physics-based, **no real data** (params are principled
  defaults, not fitted); **Tier-1** realism only (Gaussian, so the optimum stays
  computable); non-Gaussian clutter / hot pixels / shot noise deferred to future
  work. Framed as a **performance-bounds** study, which is exactly what synthetic
  data is good for.

---

## 4. Chapter plan (focused — each contribution chapter serves the headline)
- **Ch 1 — Introduction.** Low-SNR UAV detection as a bounded filtering problem;
  the gap nobody can measure on real data; the synthetic-data edge; contributions.
- **Ch 2 — Background & related work.** Detection theory (matched filter, NP,
  √T integration); deep small/dim-target detection; the ideal-observer and
  deep-vs-optimal precedents; probing & temporal-learnability. (§2 above, expanded.)
- **Ch 3 — Synthetic testbed and the computable optimum.** *(instrument; C-support)*
  The IR generator + the (whitening) optimum; validation against analytic Pd; the
  fixed-pattern integration floor.
- **Ch 4 — The gap to the optimum.** *(C1, output level)* Detector waterfalls
  (Pd vs SNR, vs N) measured against the optimum: gap-to-optimal as f(SNR, N).
  Turns "we got X" into "we got X; optimum is Y; deficit is Y−X."
- **Ch 5 — Localizing the loss (core).** *(C2)* Per-stage detectability-vs-depth;
  find the cliff; show it depends on SNR, N, and target motion vs fixed-pattern noise.
- **Ch 6 — Repairing the bottleneck.** *(C3)* Redesign the worst stage; re-measure;
  prove recovery toward the optimum.
- **Ch 7 — The temporal-integration limit (synthesis).** *(C4)* Learnable horizon
  vs √T; the motion-dependent fixed-pattern floor; the headline answer to
  "how many frames are worth stacking, and what stops you."
- **Ch 8 — Conclusions, limitations, future work.** Physics-based scope; path to
  real-data calibration and Tier-2 (non-Gaussian) effects.

**Appendices (preserve the work without diluting the headline):**
- **A — Spiking-network (SNN) ablation.** Honest negative/secondary result.
- **B — Detector engineering.** Accumulator front-end, SNR curriculum, hard-negative
  mining, temporal-window scaling (the Run A→M story) — the *object* we analyze.
- **C — Reproducibility.** Configs, seeds, hyperparameters.

---

## 5. Phased plan & milestones
| Phase | What | Status |
|---|---|---|
| **1** | Synthetic testbed (clean + realistic IR) + clean optimum + detector + waterfalls + probe dataset | **largely done** |
| **2** | Whitening optimum — extend the ruler to fixed-pattern/colored noise (so the IR ceiling is valid) | next |
| **3** | Retrain the baseline detector on the testbed (the object to probe) | next |
| **4** | **Localization pilot** — the "detectability vs depth, vs optimum" figure; find the cliff. On clean data first, then IR. *Make-or-break gate.* | core |
| **5** | **Repair** the identified bottleneck; prove recovery | core |
| **6** | Temporal-limit sweep (SNR × N × motion); synthesis (Ch 7) | core |
| **7** | Write-up; final high-runs figures | last |

Decision gate at Phase 4: if a clear cliff exists → proceed to repair. If the
cliff is at the input (signal truly gone) → that is itself a result; refocus on
the SNR/N band where the optimum *can* detect.

---

## 6. Risks & mitigations
- **Probe under-reads a layer.** Probe detectability is a *lower bound*; hold the
  probe family fixed across layers so the *drop* is the signal; report a strong-probe variant.
- **"Cliff at the input" (info-limited, not the network's fault).** Run the pilot
  in the band where the optimum provably detects, so any internal cliff is real.
- **Repair doesn't help.** Then the loss is intrinsic to that operation — a
  finding, not a failure; move to the next-steepest stage.
- **Domain gap / no real data.** Frame explicitly as a bounds study; state the
  physics-based caveat; future work = calibrate noise to a real IR clip.
- **Novelty challenged.** Pre-empt with the honest "borrowed tool, new application"
  framing (§1–§2); plant the flag on drones + the temporal limit, where it's open.
- **Scope creep / padding.** Keep Ch 4–7 on the headline; SNN and engineering stay
  in appendices.

---

## 7. What success looks like (the defense in three figures)
1. **Gap-to-optimal waterfall** (Ch 4): detector vs the optimum, vs SNR and N.
2. **Detectability-vs-depth with the cliff** (Ch 5): *the* signature figure — where
   the signal dies inside the network.
3. **Before/after repair + the temporal-limit map** (Ch 6–7): the fix works, and
   here is the frame-stacking horizon and the fixed-pattern floor.

Elevator pitch: *"Deep detectors fail at low SNR — but is it the physics or the
network? On a synthetic IR drone testbed where the optimal detector is computable,
I show exactly where inside the network the faint target is lost, fix that stage,
and map how far temporal integration can go before fixed-pattern sensor noise
stops it."*
