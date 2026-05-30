# collapseprobe research charter

This is the working anchor for the project. Before adding a feature, an experiment,
or a claim, check it against this document. The goal is to stay on the one
question we set out to answer and to keep every claim grounded in either a
measurement we ran or a paper someone published, never in intuition alone.

Related files in this folder: `proposal.tex` (the formal proposal, the version to
submit), `THESIS_PLAN.md` (chapter structure), `PROPOSAL.md` (earlier prose draft
of the idea). This charter consolidates the idea, the goal, the desired results,
and the literature, and is the one to keep current.

---

## How to use this document (guardrails)

1. **No claim from intuition alone.** Every empirical statement we make must trace
   to (a) a result we measured in this repo, or (b) a citation in the
   "Literature" section below. If it traces to neither, it is a hypothesis and
   must be labelled as one until tested.
2. **Check new ideas against the optimum and the literature first.** When a new
   "what if we tried X" comes up, ask: does the computable optimum say X can
   help, and has someone already done X? Only then build it.
3. **Stay in scope.** The project is the four contributions in Section 3. The
   spiking-network experiments (`spikeNN/`), detector-architecture tuning, and
   real-sensor data are out of scope (see Section 7).
4. **Keep work inside `collapseprobe/`.** Reuse `src/` and `npceiling/`; do not
   fork their physics.
5. **Update this file when the plan changes**, so it never goes stale.

---

## 1. What we are doing (summary)

Deep networks detect drones well at moderate signal levels but fail when the
target is faint, as a small drone is when seen from far away by a thermal
infrared camera. When a detector fails, we normally cannot tell whether the
target was too weak for any method to find, or whether the network had the
evidence and discarded it. We work in a synthetic infrared testbed where the
optimal detector can be computed exactly, use it as a reference inside the
network to find the stage where detectability is lost, repair that stage, and
map how far stacking frames keeps helping before fixed-pattern sensor noise stops
it.

---

## 2. Goal and research questions

**Goal.** Determine where, and why, a deep low-SNR drone detector falls short of
the optimal detector, whether the shortfall can be repaired, and what ultimately
limits temporal integration.

- **Q1 (localize).** Is there a stage in the network where detectability
  collapses relative to the optimum, and where is it?
- **Q2 (repair).** Does redesigning that stage recover detectability toward the
  optimum, by an amount consistent with theory?
- **Q3 (limit).** How far does adding frames keep improving detection before
  fixed-pattern noise caps the gain, as a function of SNR, sequence length, and
  target speed?

---

## 3. What success looks like (desired results)

**Contributions we aim to deliver.**
- **C1.** A method and figure: detectability versus network depth, with the
  optimal detector as the ceiling.
- **C2.** A map of where low-SNR signal is lost inside a deep detector, as a
  function of SNR, sequence length, and motion.
- **C3.** A redesign of the responsible stage, with a measured test of how much
  detectability it recovers.
- **C4.** A description of the temporal-integration limit set by fixed-pattern
  noise, including its dependence on target motion.

**Three signature figures** (the thesis stands on these):
1. Detector versus the optimum across SNR and number of frames (the gap).
2. Detectability versus depth, showing the stage where it drops (the cliff).
3. Before and after the repair, plus the temporal-integration-limit map.

**Success criteria, and what still counts as a valid result.** We do not assume
the rosy outcome; each likely result is reportable:
- If a sharp drop exists at one stage, C2 is a clean positive result. If the
  decline is gradual, that is still a result (the loss is distributed), it just
  makes the figure less dramatic.
- If the drop sits at the input (the optimum also fails there), the case is
  information-limited rather than network-limited. That is a finding, and tells
  us to move to an SNR or sequence-length range where the optimum succeeds.
- If the repair recovers detectability, C3 is a positive result bounded by the
  optimum. If it does not, the loss is intrinsic to that operation, which is also
  a finding worth reporting.
- For C4, we expect either a clear fixed-pattern floor or a motion-dependent one;
  either is a result.

---

## 4. Approach

1. **Testbed with a computable optimum.** Synthetic infrared sequences (Gaussian
   point target, the 3-D sensor-noise model with a random part and a
   fixed-pattern part, plus brightness fluctuation). Because the signal and noise
   are known, the matched filter (white noise) and the whitening matched filter
   (correlated noise) give an exact optimum to compare against.
2. **Find where detectability is lost.** Freeze a trained detector. At each
   internal stage, estimate the best detection performance recoverable from that
   stage with a fixed probe, and compare it to the optimum. Detectability can
   only fall with depth (data-processing inequality), so we look for the stage
   with the largest drop (the bottleneck).
3. **Repair.** Replace the bottleneck stage with a feature-reduction designed to
   keep the signal, leave the rest fixed, and re-measure. The optimum caps the
   possible gain, so the claim is testable.
4. **Map the temporal limit.** Trace detection versus number of frames for the
   optimum and the detector, across SNR and target speed, separating the
   white-noise region (gain grows with frames) from the fixed-pattern floor.

---

## 5. Literature: what past researchers established, and where we stand

Grouped by theme. For each, what it gives us, and how our work relates. Full
citations in the References section. The first set (5.1 to 5.5) is the rigorously
checked backbone; 5.6 is broader field context.

### 5.1 Detection theory: the optimum and the frame-stacking rule
- **Marcum (1947/48)** and **Kay (1998)** establish that the matched filter is the
  Neyman-Pearson optimal detector for a known signal in white Gaussian noise, and
  that the whitening (generalized) matched filter is optimal for known signal in
  correlated Gaussian noise. *We use this as our computable ceiling.*
- **Richards (2014)** and **Kay (1998)** give the temporal/pulse integration gain:
  combining frames raises detectability, growing like the square root of the
  number of frames in the white-noise case. *This is the baseline rule our Q3
  tests the limits of.*

### 5.2 Infrared sensor noise
- **D'Agostino and Webb (1991)**, the NVESD three-dimensional noise model,
  decomposes infrared sensor noise into parts that vary frame to frame and a
  fixed pattern (non-uniformity-correction residual) that repeats every frame.
  *We use this model directly; the fixed-pattern part is what produces the
  integration floor in Q3.*

### 5.3 The optimum used as a yardstick (the borrowed idea)
- **Barrett, Yao, Rolland and Myers (1993)** and **Barrett and Myers (2004)**:
  medical imaging uses the optimal "ideal observer" to judge whether image
  processing preserves a faint signal. *This is exactly the move we borrow, and
  we apply it inside a deep detector.*
- **Gabbard et al. (2018)**: in gravitational-wave astronomy a deep network is
  compared to the matched filter and shown to match it. *Closest in spirit, but
  output-level only, different domain.*
- **Anders, Kothari and Buehrer (2025)**: deep detection of signals with unknown
  parameters, compared to the matched filter, on radio-frequency data.
  *Output-level, not video, no internal localization.*
- **Chaudhry and Gadkari (2026)**: transformers approximate the likelihood-ratio
  (Neyman-Pearson) statistic in-context, linking internal representations to the
  optimum. *Closest in concept, but general in-context learning, not faint
  moving-target video, and it shows networks can reach the optimum rather than
  finding where they lose signal.*

### 5.4 Looking inside networks
- **Alain and Bengio (2016)**: linear classifier probes measure what each layer
  has linearly available. *We use probes to estimate per-stage detectability.*
- **Shwartz-Ziv and Tishby (2017)** and **Tishby, Pereira and Bialek (1999)**:
  information-plane analysis and the information bottleneck track how much
  information each layer keeps about the label. *Related method, but for general
  tasks and with no optimal-detector reference.*
- **Cover and Thomas (2006)**: the data-processing inequality, our guarantee that
  detectability cannot increase with depth. *Foundation of the "staircase" in
  step 2 of the approach.*

### 5.5 Temporal learnability
- **Livi (2025)**: the longest time span over which a recurrent network can learn
  long-range structure (a learning-based horizon). *We compare this learnable
  horizon to the classical frame-stacking bound in Q3/C4.*

### 5.6 Deep infrared small-target detection and alternative feature reductions (field context)
- **Cheng, Lai, Xia and Zhou (2024)**, a review of infrared dim small-target
  detection networks. *Shows the field is large and competes on accuracy with no
  optimal-detector reference, which is the gap we fill.*
- Work on the loss of small objects under downsampling, for example **Learning
  Adjustable Reduced Downsampling (Remote Sensing 13(18):3608)**, supports our
  hypothesis that early spatial reduction can erase a few-pixel target.
- Alternative, signal-preserving reductions such as **Learnable Discrete Wavelet
  Pooling (arXiv:2109.06638)** are candidate repairs for step 3, and a reminder
  that "try a different reduction" alone is a crowded area, which is why we apply
  it only at the proven bottleneck.
- Classical motion integration such as the **3-D Hough-transform track-before-
  detect technique (Sensors 19(4):881, 2019)** is the non-learned counterpart of
  integrating along a trajectory, and the basis of the front-end in `src/`.
- The "network as a learned matched filter" view (**Stankovic and Mandic, arXiv:
  2108.11663**) connects convolutional layers to matched filtering, supporting the
  premise that a network's representation can be measured against the optimum.

---

## 6. What is ours, honestly

Every ingredient above already exists. What we could not find anyone doing, after
several rounds of searching, is the combination: the computable optimum used
inside the network, layer by layer, to localize and then repair where a faint
signal is lost, for moving-target infrared video, and tied to the
temporal-integration limit. The honest framing is "a tool borrowed from the
ideal-observer tradition, applied for the first time in this place and for this
problem." This is a modest but defensible contribution, which is the right level
for an MSc.

One fact we rely on is our own, not borrowed: at low SNR our detector reaches
about 0.65 probability of detection where the computable optimum reaches near
1.0. That measured gap is what guarantees there is something inside the network
to find.

### Key claims and their basis (the no-made-up-things table)
| Claim we rely on | Basis |
|---|---|
| Matched filter is optimal for a known signal in white Gaussian noise | Marcum; Kay |
| Whitening matched filter is optimal for correlated Gaussian noise | Kay |
| Stacking frames gives roughly square-root gain | Richards; Kay |
| Infrared noise has a fixed pattern that does not average out | D'Agostino and Webb |
| Detectability cannot increase with network depth | Cover and Thomas (data-processing inequality) |
| The optimum can serve as a yardstick for whether processing keeps a signal | Barrett et al. |
| Deep networks can match or approach the matched filter | Gabbard et al.; Anders et al.; Chaudhry and Gadkari |
| Early downsampling can erase small targets | small-object detection literature (e.g. Remote Sensing 13(18):3608) |
| There is a real gap between our detector and the optimum at low SNR | our own measurement (`npceiling` sweep) |

---

## 7. Scope and what we are not doing

- Synthetic data only. This is deliberate: it is what makes the optimum
  computable. The cost is a gap to real footage. Noise parameters are set from
  physics, not fitted to a specific camera, because we do not assume access to
  real low-SNR drone footage.
- Realism is kept Gaussian with known covariance so the optimum stays exact.
  Non-Gaussian clutter, hot pixels, and shot noise are out of scope.
- We study recognition given the rough target location (centered crops), to
  isolate the network from the detection front-end. Full end-to-end detection is
  not the subject.
- Out of scope: the spiking-network experiments, detector-architecture tuning for
  its own sake, and calibration to or testing on real recordings. These are
  possible future work, not part of this thesis.

---

## References

1. J. I. Marcum, *A Statistical Theory of Target Detection by Pulsed Radar*, RAND RM-753/RM-754, 1947-1948.
2. S. M. Kay, *Fundamentals of Statistical Signal Processing, Vol. II: Detection Theory*, Prentice-Hall, 1998.
3. M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014.
4. J. A. D'Agostino and C. M. Webb, "Three-dimensional analysis framework and measurement methodology for imaging system noise," Proc. SPIE 1488, 1991, doi:10.1117/12.45794.
5. H. H. Barrett, J. Yao, J. P. Rolland and K. J. Myers, "Model observers for assessment of image quality," PNAS 90(21):9758-9765, 1993. (Also H. H. Barrett and K. J. Myers, *Foundations of Image Science*, Wiley, 2004.)
6. H. Gabbard, M. Williams, F. Hayes and C. Messenger, "Matching matched filtering with deep networks for gravitational-wave astronomy," Phys. Rev. Lett. 120, 141103, 2018; arXiv:1712.06041.
7. T. Anders, H. P. Kothari and R. M. Buehrer, "On the ability of deep learning to detect signals with unknown parameters," arXiv:2512.13542, 2025.
8. F. Chaudhry and S. Gadkari, "Implicit statistical inference in transformers: approximating likelihood-ratio tests in-context," arXiv:2603.10573, 2026.
9. G. Alain and Y. Bengio, "Understanding intermediate layers using linear classifier probes," arXiv:1610.01644, 2016.
10. R. Shwartz-Ziv and N. Tishby, "Opening the black box of deep neural networks via information," arXiv:1703.00810, 2017.
11. N. Tishby, F. C. Pereira and W. Bialek, "The information bottleneck method," arXiv:physics/0004057, 1999.
12. T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley, 2006.
13. L. Livi, "Learnability window in gated recurrent neural networks," arXiv:2512.05790, 2025.
14. Y. Cheng, X. Lai, Y. Xia and J. Zhou, "Infrared dim small target detection networks: a review," Sensors 24(12):3885, 2024.
15. "Learning Adjustable Reduced Downsampling Network for Small Object Detection," Remote Sensing 13(18):3608, 2021.
16. "Learnable Discrete Wavelet Pooling (LDW-Pooling) for Convolutional Networks," arXiv:2109.06638, 2021.
17. "A Three-Dimensional Hough-Transform-Based Track-Before-Detect Technique," Sensors 19(4):881, 2019.
18. L. Stankovic and D. Mandic, "Convolutional Neural Networks Demystified: A Matched Filtering Perspective," arXiv:2108.11663, 2021.
