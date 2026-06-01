# collapseprobe — the story so far, in plain language

A readable narrative of what we're doing and why, meant to be understood without
the detection-theory jargon. The blow-by-blow with numbers lives in
`EXPERIMENTS.md`; the formal framing in `RESEARCH_CHARTER.md`. This file is the
"explain it to me simply" version.

---

## The problem

We want to detect a small drone, far away, seen by a thermal (infrared) camera.
Far away + small = the drone is **faint**: only a pixel or two, barely above the
sensor's noise. A neural-network detector often misses it.

When the network misses, we face a question we normally **cannot answer**:

> Did it miss because the drone was genuinely too faint for *anything* to detect
> (the information just isn't in the video) — or because the information *was*
> there and the network threw it away?

These call for opposite responses (give up vs. fix the network), and you can't
tell which it is by staring at the network's final yes/no.

## The trick that makes it answerable

We use **synthetic** (computer-generated) video. Because we generated it, we know
the exact drone shape and the exact noise statistics. That lets us compute the
**best detector that could possibly exist** — a mathematical ideal, a ceiling no
algorithm can beat. (For experts: the matched filter / whitening matched filter,
the Neyman–Pearson optimum.)

Now we have a referee:

- If even the ideal detector can't find the drone → it's genuinely too faint.
  Not the network's fault.
- If the ideal detector finds it but the network doesn't → **the information was
  there and the network wasted it.** Now the real question: *where inside the
  network did it get lost?*

To find *where*, we look inside the network **layer by layer**. At each layer we
ask: "could you still tell the drone is here from what's at this layer?" We watch
that number fall as we go deeper. Wherever it drops sharply is the stage that
breaks the signal. (For experts: linear probes at each layer, compared to the
optimum.)

That's the whole idea: **use the perfect detector as a ruler held up against the
insides of a real network, to find exactly where the network loses a faint
target.**

---

## What we did, step by step

1. **Checked our ruler.** Confirmed the "ideal detector" behaves exactly as
   textbook theory says (it gets predictably harder as the drone dims). So we
   trust it as a reference.

2. **Made the test hard enough to be interesting.** Our first settings were too
   easy — the drone was always findable, by both the ideal and any simple method,
   so there was nothing to study. We dimmed the drone until there was a real gap:
   the ideal still finds it, simple methods start to fail.

3. **Built the *right* ruler for a real camera.** Real thermal cameras have a
   fixed "fingerprint" of noise — the same speckle pattern burned into every
   frame, which does **not** average away no matter how many frames you stack.
   The simple ideal detector ignores this; we built the correct one that accounts
   for it. This realistic setting is where the network's gap actually lives.

4. **Trained a normal network detector** and confirmed the premise: the ideal
   detector finds the drone (near-perfect), the network falls short, and the
   **shortfall grows as the drone gets fainter.** So there's a real, growing gap
   to explain.

5. **Looked inside, layer by layer, and found the leak.** The drone is highly
   detectable at the network's input, then detectability **falls off a cliff at
   the very first step where the network shrinks the image** (it halves the
   resolution, 31→15 pixels). The one- or two-pixel drone gets blurred into the
   noise right there. After that, it never fully recovers.
   - *We also caught ourselves being wrong.* A first, over-trained network gave a
     fancier story (a dip in the middle, a quirk at the output). When we trained a
     cleaner network and repeated the measurement several times, that fancier
     story disappeared — the honest, repeatable result is the simple one: the loss
     is at the **first image-shrink**. (Keeping ourselves honest is the point of
     logging everything.)

6. **Fixed that one spot — and it worked.** The shrinking step used "keep the
   brightest pixel in each little patch" (max-pooling). For a faint target in
   noise that's a bad idea: the brightest pixel in a patch of mostly-noise is
   usually just a noise spike, so it *raises the noise floor* and drowns the drone.
   The ideal detector instead **averages** (integrating signal, cancelling noise).
   So we changed the shrink step to averaging. Re-trained, re-measured: the network
   **recovered a large chunk of the lost performance** — about two-thirds of the
   gap to the ideal at moderate faintness, less but still positive at the very
   faintest. Confirmed across several training runs so it isn't a fluke.

---

## The results, in one table

"Detection score" below is ROC-AUC: 0.5 = coin flip, 1.0 = perfect.
"SNR" = how faint the drone is per frame; more negative = fainter.

| how faint | ideal (ceiling) | network before fix | network after fix |
|-----------|:---:|:---:|:---:|
| −6 dB | 1.00 | 0.94 | 0.98 |
| −9 dB | 1.00 | 0.83 | 0.90 |
| −12 dB | 0.96 | 0.78 | 0.81 |
| −15 dB | 0.91 | 0.67 | 0.73 |

Read it as: the ideal detector (what's *possible*) stays high; the network falls
short; our fix moves it back **toward** the ceiling without ever exceeding it
(it can't — the ceiling is the limit).

---

## The one-sentence version

We built a way to hold the mathematically perfect detector up against the insides
of a real drone detector, used it to pinpoint exactly where the network throws a
faint target away (its first image-shrinking step), and showed that one simple,
principled change to that step recovers much of the lost detection — all measured
against the perfect detector as ground truth.

## What's left

**C4:** how many video frames you can usefully stack to boost detection before
the camera's fixed noise fingerprint caps the gain — the limit of "watching
longer." The tools for it (the realistic ruler, the noise model) are already
built.
