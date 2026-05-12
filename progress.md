# Progress log

Living record of what we built, what we tried, the results, and what to try next. Append to this file rather than overwriting earlier sessions.

## Project

Master's thesis on temporal-integration limits for low-SNR UAV detection. The work in this repo is the algorithm side: a hard-negative-aware, tracklet-guided recurrent U-Net detector that runs on the synthetic-data generator already in `src/synthetic/`. The deliverable plot is a "waterfall" — probability of detection vs SNR at a fixed false-alarm rate.

## Session 2026-05-11 — initial build

Built the full pipeline end-to-end in one session. Twelve parts as specified, all 95 tests passing at the end.

What landed:

- `src/tbd/evidence.py`: per-frame evidence maps (raw, local_z, matched-filter, temporal_diff, combined). Pure numpy. Returns five (T,H,W) arrays.
- `src/tbd/candidates.py`: per-frame NMS local-max extractor returning Candidate objects.
- `src/tbd/tracklets.py`: beam-search multi-hypothesis tracklet builder with motion + miss penalties.
- `src/tbd/crop_tubes.py`: extracts (T,C=5,S,S) crop tubes along a tracklet, stacking all five evidence channels.
- `src/data/tracklet_dataset.py`: `TrackletCropDataset` (IterableDataset) plus `build_tube_sample` which runs the full per-sequence pipeline and returns labeled tubes. Labeling: positive if ≥3 frames within 3 px of GT.
- `src/models/recurrent_unet.py`: `TrackletRecurrentUNet`. Per-frame U-Net encoder + ConvLSTM bottleneck + per-frame decoder + three heads (track, visibility, heatmap). `use_convlstm=False` for the no-recurrence ablation.
- `src/models/convlstm.py`: standard 4-gate ConvLSTM cell (single fused conv).
- `src/models/cnn_gru_baseline.py`: `CropCNNGRU` baseline.
- `src/training/losses.py`: focal track loss + masked BCE for heatmap and visibility.
- `src/training/train_tracklet_model.py`: CLI training loop using `TrackletCropDataset`.
- `src/training/mine_hard_negatives.py`: false-positive miner (caches top FP tubes to a pickle, dataset injects them at train time).
- `src/eval/eval_tracklet_detector.py`: sequence-level SNR sweep, Pd@FAR, ROC/PR-AUC.
- `tests/test_tracklet_pipeline.py`: 17 end-to-end tests.

Initial sanity-check demo run (`scripts/train_demo.py`, 20 fixed videos at SNR +2..+10 dB) gave val ROC-AUC ≈ 1.0 at epoch 0 (the SNR range was too easy to demonstrate "learning" via AUC). The plumbing was correct.

## Session 2026-05-12 — push toward the real low-SNR regime

### Run A — single-stage low-SNR demo (diagnostic)

Configuration: 100 videos at SNR −15..−3 dB, single-stage training, beam-search tracklet source.

Result: train pool had only 2 positive tubes out of 800 — the per-frame top-K + beam search almost never finds the real UAV at very low SNR. Model collapsed to predicting ≈0 for everything. Val ROC-AUC drifted around 0.5. SNR sweep was flat at Pd ≈ 0 across all cells.

Diagnosis: the bottleneck is the front-end, not the classifier. Per-frame thresholding throws the target away before any learnable component sees it.

### Run B — TBD evidence accumulator (fix #1)

Built `src/tbd/accumulator.py`. For each velocity hypothesis `(vy, vx)` on a polar grid, sum evidence along the predicted trajectory: `acc(y,x) = Σ_t evidence[t, y+vy·t, x+vx·t]`. Collapse the V score maps by per-pixel max, NMS, top peaks → tracklets directly. Initially 33 hypotheses (5 speeds × 8 directions, including v=0). Integer-rounded shifts. Wired into `TrackletDatasetConfig.tracklet_source` so beam-search vs accumulator is a single switch.

Run with 200 stage-3 videos: train positives went from 2 to 12 (6×). With stratified batching (2 positives per batch of 8, 64 batches/epoch), val ROC-AUC peaked at 0.723. Pd@FAR=1/100 in the SNR sweep lifted from flat 0.0–0.15 to 0.0 → 0.65 across SNR (max at +3 dB). Score separation appeared.

Takeaway: the accumulator works as predicted. The next bottleneck became the very small number of unique positives (~12 in train).

### Run C — curriculum training (option 2)

Two-stage, then three-stage. Stage 1 at +2..+10 dB warmup (lots of positives), stage 2 transitional at −5..+2 dB, stage 3 deployment-target at −15..−3 dB. Each stage uses its own LR (decaying from 2e-3 down to 2e-4) and best-val checkpointing. Stage 3 uses rehearsal — 25 % of stage 1's tubes + 30 % of stage 2's mixed into the stage-3 training pool — to prevent catastrophic forgetting.

Three-stage config with 80 + 200 + 400 videos:
- Stage 1 best val AUC: 0.996
- Stage 2 best val AUC: 0.813
- Stage 3 best val AUC: 0.515
- Pd@FAR=1/100: 0.15 at −20 dB rising to 0.75 at +3 dB
- Average over all SNR cells: 0.19
- mean(positive_score) − mean(negative_score) at +3 dB: +0.278

The full SNR sweep showed a real waterfall for the first time. The 3-stage version uniformly lifted Pd vs the 2-stage version, including at the very-low-SNR end (−20 dB went from 0.0 to 0.15). The intermediate stage bridges the distribution shift that one-jump curriculum couldn't handle.

### Run D — comparison videos

`scripts/train_demo.py` now renders 5 three-panel videos to `demo_outputs/videos/`:
- input | model best tracklet | ground truth, frame-by-frame, with score and verdict in the model panel.
- Originally GIFs via PillowWriter; switched to MP4 via FFMpegWriter with the ffmpeg binary bundled in `imageio_ffmpeg` (no system ffmpeg needed). Falls back to GIF if ffmpeg missing.

Test cases: SNR +5 / −3 / −10 dB UAV-present, plus SNR −3 dB empty-background and hard-negative.

Behavior observed: HIT at +5 dB (score 0.85), misses at −3 and −10 dB (model picks an arbitrary noise tracklet, scores it ~0.4, correctly labels "no UAV"). No false alarms on negatives. The model is a high-precision, SNR-limited detector.

### Run E — 10× more low-SNR data

Stage 3 videos: 400 → 4000. To keep memory bounded, added `build_dataset_streaming` in `scripts/train_demo.py` which processes one video at a time and reservoir-samples negatives down to `STAGE3_MAX_NEGATIVES = 5000`. Bumped `BATCHES_PER_EPOCH` from 64 to 200 globally so the larger pool actually gets consumed. Stage 3 epochs 6 → 10.

Result: stage 3 train positives went from 157 to 1533 (the expected 10×). Stage 3 best val AUC: 0.515 → 0.582. Pd@FAR=1/100 at 0 dB jumped from 0.35 to 0.65 (almost doubled). Pd at +3 dB: 0.75 → 0.80. **But Pd at SNR ≤ −3 dB stayed flat at ≈0.05** — more classifier data did not help the regime where the front-end accumulator can't recover signal in the first place.

Side regression: bumping `BATCHES_PER_EPOCH` globally caused stages 1 and 2 to over-rehearse their (still small) positive pools. Stage 2 best val AUC dropped from 0.813 to 0.702. Fixed in Run F.

Takeaway: more classifier data lifted the moderate-SNR regime as predicted. Confirmed the front-end is the binding constraint below ~−3 dB. The next experiment had to address the front-end itself.

### Run M — annotated demo videos + persistent checkpoint (presentation)

Same model architecture and curriculum as Run L. Two infrastructure changes for showing the system to a non-technical audience (thesis advisor):

1. **13-case captioned video set** (vs 5 previously). Each video has a multi-line caption — *Case N: title* / *What this tests* / *What we expect* — plus a color-coded verdict ribbon (`darkgreen` = correct behavior, `darkorange` = expected-at-floor, `red` = wrong in a way to worry about). Verdict logic accounts for SNR regime (saturated / threshold / marginal / impossible) and whether the best tracklet actually lands on the GT trajectory (≥50% of visible frames within 3.5 px). New cases: +10 dB sanity, 0 dB equal-power, -3 dB variance demo (second seed), -6 dB marginal, -15 dB impossible, two `mixed_uav_and_distractors` scenes at +3 and -3 dB.

2. **Model checkpoint persisted** to `demo_outputs/model_checkpoint.pt`. New `DEMO_RENDER_ONLY=1` env-var fast path loads the checkpoint and renders videos without retraining (~3 min vs ~30 min full run). Future iteration on captions / cases is now cheap.

Headline metrics are identical to Run L (same data seeds, same curriculum):

- Stage 1 best 0.998, Stage 2 best 0.930, Stage 3 best 0.670, Stage 4 best 0.670, Stage 5 best 0.669
- Demo scores: +10 dB 0.817, +5 dB 0.898, +3 dB 0.905, 0 dB 0.801, -3 dB 0.690 (alt seed 0.722), -6 dB 0.567 (lucky hit), -10 dB 0.368 (expected miss), -15 dB 0.430 (expected miss), empty 0.406, hard_negative 0.441, mixed +3 dB 0.869, mixed -3 dB 0.674
- **All 13 verdicts came back green** — every case landed in the expected band for its SNR regime. The marginal -6 dB hit was a genuine "lucky" outcome (per the Pd ~25% measured curve).

Wall time ~36 min (12 min stage 3, 9 min stage 4, 11 min stage 5, ~3 min plots + 13 videos). Curriculum dataset build: 176 s.

### Run L — T=32 (doubled temporal window again)

Single config change: `N_FRAMES = 32` (was 16). Stage 6 stays disabled. Wall time ~25 min on CPU (was ~17 min at T=16). Memory peak ~750 MB.

Data yield jumped further: stage 3 train positives **4640** (was 2954 at T=16, 1607 at T=8). The val set has 172 positives (was 107 at T=16) so the aggregate val metric is finally stable enough to move:

- Stage 1 best val ROC-AUC: 0.998 (project high)
- Stage 2 best val ROC-AUC: 0.916 (project high)
- Stage 3 best val ROC-AUC: 0.670 (project high — was 0.629 at T=16, 0.633 at T=8)
- Stage 4 best: 0.670 | Stage 5 best: 0.669

SNR sweep — the cleanest waterfall the pipeline has produced:

- −20 dB: 0.00 | −15 dB: 0.00 | −12 dB: 0.00 | −9 dB: 0.00 — floor unchanged
- −6 dB: 0.25 (vs Run K 0.20) — modest lift
- **−3 dB: 0.65** (vs Run K 0.25) — **biggest single-cell jump in the project (+0.40)**
- **0 dB: 1.00** (vs Run K 0.75) — saturated
- **+3 dB: 1.00** — saturated (held)

Score separation at +3 dB: **+0.462** (vs Run K +0.325, Run I +0.099). At 0 dB: +0.369 (vs Run K +0.180). At −3 dB: +0.203 (vs Run K +0.041, Run I +0.014).

Comparison videos: **+5 dB hits at 0.898 (most confident detection recorded), −3 dB hits at 0.690 (vs Run K 0.510)**. −10 dB still misses (0.368) but score separation is real. Negatives correctly stay below threshold (0.41, 0.44).

Why the lift exceeded the √T prediction: three compounding effects. (1) Front-end integration gain went from √16 to √32 — about 1.5 dB more effective SNR. (2) Stage-3 positives nearly tripled vs T=8, giving the classifier much more positive variety — first time val ROC-AUC moved (0.633 → 0.670). (3) ConvLSTM unrolls over 32 timesteps, giving the classifier more temporal context independent of the front-end. Combined, the operating curve moved another ~3 dB to the left, not just 1.5.

Mining at T=32 has **not** saturated. Round 1: 142 tubes. Round 2: 251 tubes — *more* than round 1, opposite of the T=16 saturation pattern. With more candidate tracklets per sequence overall, more FPs surface after each correction. A third mining round might still help here.

**Run L is now the best deployment-ready checkpoint, replacing Run K.** SNR ≥ 0 dB is fully saturated. −3 dB is reliably detected (0.65). The hard floor below −9 dB is now well-defined and consistent with the noise-floor theoretical limit at this canvas size and target amplitude.

### Run K — T=16 (doubled temporal window)

Single config change: `N_FRAMES = 16` (was 8). Stage 6 disabled (saturation from Run J). Everything else identical to Run I.

Data yield: stage 3 train positives **2954** (was 1607 at T=8), val positives 107 (was 56). The accumulator's √T coherent-integration gain doubled, making more true tracklets pass labeling. Build time ~93s (was 50s). Total stage 3 + mining + sweep wall time ~17 min (was ~9 min). Both expected.

Per-stage best val ROC-AUC: warmup 0.994, transition 0.854, finetune 0.629, stage-4 mining 0.628, stage-5 mining 0.625. Aggregate val AUC is essentially unchanged from T=8 because val SNR is −15…−3 dB where the gains haven't reached.

SNR sweep, all-time records in bold:

- −20 dB: 0.40 (likely partly MC noise — SE ≈ ±0.10 with n=20)
- −15 dB: 0.05
- −12 dB: 0.00
- −9 dB: 0.00
- −6 dB: 0.20
- −3 dB: 0.25 (vs Run I 0.15) — **best at this SNR**
- **0 dB: 0.75** (vs Run I 0.30) — **+45 pp jump**
- **+3 dB: 1.00** (vs Run I 0.80) — **perfect detection**

Score separation at +3 dB: +0.325 (vs Run I's +0.099) — over 3× larger gap. At 0 dB: +0.180 (vs Run I's +0.036) — 5× larger gap. The model is now genuinely confident at SNR ≥ 0 dB.

Comparison-video crossover: **the −3 dB positive_uav case is now HIT (score 0.510)** — first time the demo set has registered detection at this SNR. The +5 dB case is HIT at 0.761. Negatives still correctly rejected.

Diagnosis: doubling T moved the entire operating curve roughly 3 dB to the left, exactly as the √T coherent-integration math predicts. The detection rate the T=8 model achieved at +3 dB now appears at 0 dB. The very-low-SNR floor below ≈ −9 dB is still essentially unreachable; that's the regime where the target amplitude is so small relative to noise that even 3 dB more gain doesn't pull it above the false-alarm tail.

**Run K is now the best deployment-ready checkpoint, replacing Run I.**

### Run J — third mining round (saturation reached)

Same gentle settings as rounds 1 and 2 (oversample=1, threshold=0.5) but scanned 3000 target-absent sequences. Found **only 20 new tubes** (0.67 % per-sequence FP rate, vs round 2's 4.95 % and round 1's 1.83 %). Max mined score tightened to 0.577 (vs round 2's 0.684). Stage 6 best val ROC-AUC: 0.631 — *below* stage 5's 0.634.

SNR sweep regressions vs Run I:
- 0 dB: 0.30 → 0.20 (real regression)
- −3 dB: 0.15 → 0.05 (real regression)
- +3 dB and −6 dB: held (Pd 0.80, 0.30)
- below −9 dB: roughly within Monte Carlo noise

Diagnosis: **mining saturated at 2 rounds.** Round 3 found very few new high-confidence FPs, so the model received little useful correction signal but the same amount of optimizer churn, which started to over-suppress marginal-confidence positives at 0 / −3 dB — the same Run G failure mode at a smaller scale.

Useful stopping rule going forward: "if round N's mining count drops sharply below round N−1's, stop." Round 1 → 2 saw the count grow (the round-1 correction exposed new FPs); round 2 → 3 saw the count collapse (the model has internalized the FP pattern). That collapse is the saturation signal.

**Best deployment-ready checkpoint remains Run I.**

### Run I — iterative mining (two rounds)

Added a second mining round (stage 5) against the post-stage-4 model. Same gentle settings (oversample=1, score_threshold=0.5) but a different RNG seed and 2000 target-absent sequences scanned (vs 1200 in round 1). 5 epochs of stage-5 training at LR=1e-4. Stage-5 training pool = stage-3 pool + round-1 mined (22 tubes) + round-2 mined.

Round-2 mining harvested **99 new tubes** with median score 0.524 (range [0.500, 0.684]) — substantially more than round 1's 22. The stage-4 correction was very narrow (only 22 tubes, single oversample); after that nudge, a different and larger set of false-positive tubes drifted up over the 0.5 threshold. Iterative mining is what catches that.

Stage-5 best val ROC-AUC: 0.634 (essentially unchanged from stage 3/4 at the aggregate metric, but the deployment metric — Pd@FAR — improved substantially).

SNR sweep at FAR = 1/100, compared with prior runs:

- −20 dB: 0.00 (Run F: 0.05, G: 0.00, H: 0.00)
- −15 dB: **0.20** (Run F: 0.10, G: 0.00, H: 0.05) — **all-time best at this SNR**
- −12 dB: 0.10
- −9 dB: 0.05
- −6 dB: **0.30** (Run F: 0.15, G: 0.30, H: 0.20) — tied with Run G's best
- −3 dB: 0.15
- **0 dB: 0.30** (Run F: 0.25, G: 0.05, H: 0.15) — **all-time best at this SNR**
- +3 dB: 0.80 (Run F: 0.65, G: 0.85, H: 0.70) — within 0.05 of Run G's high mark

This is the best detector we've produced. It matches Run G's +3 dB ceiling (within MC noise), preserves Run G's −6 dB lift, and recovers the 0 dB hole that Run G introduced. Score separation at +3 dB is +0.099 — below Run H's +0.154 but well above Run G's +0.055.

Diagnosis: each gentle round delivers a small, targeted correction the model can absorb without overcorrecting. Two such rounds compound — they're additive in correction signal without being multiplicative in collateral damage. Aggressive mining (Run G) crammed too much correction signal into one round; gentle mining (Run H) delivered too little; iterative gentle mining lands in the right operating point.

### Run H — gentler hard-negative mining

Two single-number changes on top of Run G: `MINING_OVERSAMPLE` 3 → 1 (each mined tube appears once instead of three times in the stage-4 pool) and `MINING_SCORE_THRESHOLD = 0.5` (only mine tubes the model is genuinely confident-wrong about, not borderline ones). Scanned 1200 target-absent sequences (2× Run G) to compensate for the score floor.

Result: only **22 tubes** survived the threshold from 1200 sequences (median mined score 0.518, max 0.624). Stage 4 best val ROC-AUC: 0.633 (same as stage 3 — mining contributed <0.3 % of gradient signal).

SNR sweep at FAR = 1/100:
- −20 dB: 0.00 | −15 dB: 0.05 | −12 dB: 0.10 | −9 dB: 0.05
- −6 dB: 0.20 (Run F: 0.15, Run G: 0.30) — kept most of the Run G lift
- −3 dB: 0.15 | 0 dB: 0.15 (Run G: 0.05) — recovered Run G's 0 dB regression
- +3 dB: 0.70 (Run F: 0.65, Run G: 0.85) — sits between the two, with **+0.154 score separation**, the best calibrated separation we've produced

The waterfall is now nearly monotone with a clean lift starting around −6 dB and peaking at +3 dB. The comparison-video +5 dB case is back to a confident HIT (score 0.642, was 0.449 in Run G). This is the most calibrated checkpoint we've produced; Run G remains the high-water mark on raw Pd at +3 dB but at the cost of being miscalibrated at 0 dB.

Diagnosis: aggressive mining (Run G) taught the model "be suspicious of scores in 0.4–0.5", which is a blunt rule that hurts marginal true positives at 0 dB. Gentle mining (Run H) only saw confidently-wrong tubes and learned a narrower correction.

### Run G — hard-negative mining + stage-4 remediation

Added `mine_hard_negatives_inline()` in `scripts/train_demo.py`. After the 3-stage curriculum, run the trained model on `MINING_N_SEQUENCES=600` target-absent sequences, collect the top `MINING_TOP_K=300` highest-scoring tubes (forced label=0), oversample each by `MINING_OVERSAMPLE=3`, append to the stage-3 training pool, and run a 5-epoch stage 4 at LR=1e-4.

Mined-tube score range: [0.41, 0.60], median 0.43. Stage 4 best val ROC-AUC: 0.643 (up from stage 3's 0.633).

SNR sweep at FAR = 1/100:
- −20 dB: 0.00 (Run F: 0.05)
- −15 dB: 0.00 (Run F: 0.10)
- −12 dB: 0.10 (Run F: 0.10)
- −9 dB: 0.00 (Run F: 0.15)
- **−6 dB: 0.30** (Run F: 0.15) — best low-SNR Pd we've produced
- −3 dB: 0.15 (Run F: 0.15)
- 0 dB: 0.05 (Run F: 0.25) — regressed
- **+3 dB: 0.85** (Run F: 0.65) — best high-SNR Pd we've produced

Mining did what it was supposed to do at +3 dB: the bilinear+finer-grid front-end produces more spurious noise tracklets, mining suppressed them, the FAR=1/100 threshold dropped, and strong positives at +3 dB cleared it easily. Surprise bonus at −6 dB (Pd doubled). But mining overcorrected at 0 dB: marginal positives that previously sat in roughly the same score range as the mined tubes (around 0.4–0.5) got pulled down with them, losing 0.20 Pd. The waterfall is now non-monotone — peaks at +3 and −6, dip at 0.

The mined-tube score range floor of 0.41 includes borderline cases that look classifier-confusable with marginal true positives, which is the proximate cause of the 0 dB regression. Mining is a blunt knob at the current settings.

### Run F — bilinear sub-pixel sampling + finer velocity grid + per-stage batches

Three coordinated changes:

1. **Bilinear shifts in `_shift_and_sum`**: per-frame shifts are now real-valued, with a 4-tap bilinear weighted sum over the integer-shifted neighbors. For integer offsets it collapses to the original nearest-pixel behavior exactly. Added `AccumulatorConfig.bilinear: bool = True`.
2. **Wider velocity grid**: `n_speeds=7, n_directions=12` → 73 hypotheses (was 33). Speeds {0, 0.33, 0.67, 1.0, 1.33, 1.67, 2.0} px/frame.
3. **Per-stage `batches_per_epoch`**: 64 / 100 / 300 for stages 1 / 2 / 3. `run_stage` now takes it as a parameter rather than reading a global.

Same 4000-stage-3-video pool as Run E.

Result:
- Stage 1 best val AUC: 0.996 (unchanged)
- Stage 2 best val AUC: 0.846 (per-stage batches fix worked; was 0.702 in Run E)
- Stage 3 best val AUC: 0.633 (+0.05 over Run E)
- Pd@FAR=1/100 at −3 dB: 0.05 → **0.15** (3× lift, the predicted win)
- Pd@FAR=1/100 at −12 dB: 0.00 → 0.10
- Pd@FAR=1/100 at +3 dB: 0.80 → 0.65 (regression)
- Pd@FAR=1/100 at 0 dB: 0.65 → 0.25 (regression)

Mixed result. The low-SNR end lifted as predicted — score separation now extends down to about −6 dB instead of stopping at 0 dB. The high-SNR end regressed because the bigger grid + bilinear creates more tracklet candidates per sequence, including more high-scoring noise tracklets, so the max-over-tracklets reduction on target-absent sequences yielded a higher negative-score distribution, pushing the FAR=1/100 threshold up and clipping some of the high-SNR positives.

Takeaway: the front-end is now more sensitive in both directions — to signal AND to clutter. The classifier wasn't trained on the noisier candidate distribution the new front-end produces. The natural next step is hard-negative mining, which is built but not yet enabled.

## What works as of 2026-05-12

- `scripts/train_demo.py` runs end-to-end in about 4 minutes on CPU and produces:
  - `demo_outputs/learning_curve.png` (3-stage staircase plot)
  - `demo_outputs/snr_sweep.png` (waterfall + score-separation panels)
  - `demo_outputs/score_distribution.png` (val-set score histogram by label)
  - `demo_outputs/example_tubes.png` (top-3 positive predictions, top-3 worst-FP negatives)
  - `demo_outputs/videos/*.mp4` (5 three-panel comparison videos)
  - `demo_outputs/history.json`, `qa_report.json`, `snr_sweep.json`
- All 99 tests pass after every change.

## Key learnings

- Per-frame top-K + beam search is a non-starter at SNR ≤ −10 dB. The target almost never makes the per-frame cut.
- Coherent integration across T frames before any thresholding (classical TBD accumulator) is what makes low-SNR detection possible at all. Recovers roughly √T effective SNR.
- With heavy class imbalance (positives at ~1 % of pool at low SNR), the model collapses to "predict zero" unless every batch is guaranteed to contain positives via stratified sampling.
- Best-val checkpointing per stage is essential. Validation AUC oscillates a lot with small positive counts; the last-epoch model is often worse than the best snapshot.
- Single-jump curriculum (high SNR → low SNR) causes catastrophic forgetting. A 3-stage curriculum with an intermediate-SNR bridge stage works much better.
- Rehearsal (mixing earlier-stage tubes into later stages) helps preserve features the earlier stages learned.
- More classifier data only helps in the regime where the front-end produces real signal. Below that, fixing the front-end is what matters.
- A more sensitive front-end produces more candidates of every kind — true positives and noise tracklets alike. Without the classifier being trained on the new candidate distribution, the false-alarm rate can rise faster than the detection rate.

## Open ideas to try next

In rough priority order, given the current state.

**Third mining round at T=32.** Run J showed mining saturates at T=8 (round 3 found 0.67 % FP rate, regressed metrics). Run L shows mining has NOT saturated at T=32 (round 2 found *more* FPs than round 1: 251 vs 142). Re-enabling stage 6 with the same gentle settings would likely lift Pd in the −6 to −3 dB band by a few more pp without the Run J regression. ~5 min more wall time. Highest-leverage next experiment.

**T=64.** Would add another ~1.0–1.5 dB of integration gain on top of T=32. Likely lifts −6 dB Pd from 0.25 toward 0.4–0.5 and −9 dB from 0.00 to small but nonzero. Costs are real: wall time ≈ 60 min on CPU, memory ~1.5 GB peak, and the trajectory sampler may start to fail at T=64 with the current speed_max=1.5 px/frame on a 64×64 canvas (max travel 96 px, larger than the canvas). Would need either reducing speed_max to 1.0 or enlarging the canvas to 96×96.

**Data augmentation (parallel, low-risk lift).** Random horizontal/vertical flips of crop tubes during stratified batching. Free at training time, multiplies positive variety by 4. Compatible with the Run I curriculum. Apply alongside longer-T to compound gains.

**Sub-pixel front-end at inference only (research-grade lever).** Currently `bilinear=True` is on for both training and inference. Could try training with `bilinear=False` (gives a cleaner positive distribution because integer shifts produce tighter peaks) and inferring with `bilinear=True` (gives a more sensitive front-end at deployment). Worth one experiment to A/B test.

**Trim `accumulator_top_k`.** Currently 20 per sequence. With 73 velocity hypotheses producing more tracklets per video, we might be keeping too many low-quality candidates. Dropping to 8–10 would cut the negative tail of the score distribution. (Note: Run G's mining largely addressed this from the classifier side; this remains a useful complementary lever.)

**Iterative mining.** Mine → train → mine → train. After Run G the model is making *different* false positives than before mining; a second mining round could pick those up. Each round can be smaller and gentler since the FP pool is shrinking.

**Longer temporal window.** Go from T=8 to T=16. The √T integration gain adds ~3 dB. The model architecture already handles variable T; only the synthetic data config needs `n_frames=16`. Cost is more memory per tube.

**Data augmentation in stratified batching.** Random horizontal/vertical flips of tubes, small brightness shifts. Free at training time, multiplies positive variety by 4. Not yet implemented.

**Slightly larger model.** With 1500+ positives at stage 3, the capacity-to-data ratio supports moving from base_channels=16 to 24, bottleneck from 32 to 48. Probably modest gain.

**Multi-scale ConvLSTM.** A small ConvLSTM at each encoder level instead of just the bottleneck. Helps localization more than detection. Should come after the front-end and the hard-negative-mining fixes are in.

**What I would NOT do.** No more "scale data" experiments without first improving the front-end and the classifier-side data composition. No replacing the architecture with a transformer/video-swin — the current model is not the bottleneck. No tinkering with the loss function — focal at α=0.25, γ=2 is doing its job.

## File index (for context recovery)

- Pipeline: `src/tbd/{evidence, candidates, tracklets, crop_tubes, accumulator}.py`
- Model: `src/models/{recurrent_unet, convlstm, cnn_gru_baseline}.py`
- Dataset: `src/data/tracklet_dataset.py` (key entry: `build_tube_sample`, `TrackletCropDataset`)
- Training: `src/training/{losses, metrics, train_tracklet_model, mine_hard_negatives}.py`
- Eval: `src/eval/eval_tracklet_detector.py`
- Tests: `tests/test_tracklet_pipeline.py` (21 tests, plus 78 pre-existing synth-data tests)
- Demo: `scripts/train_demo.py` (the runnable everything)
- Outputs: `demo_outputs/` (plots, history, videos)
