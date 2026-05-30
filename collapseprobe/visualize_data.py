"""Make a human-eyeball preview of the synthetic data: simple vs realistic IR.

Renders one frame of a faint drone at a chosen SNR, with the drone circled and a
zoom-in, for both the simple-noise and realistic-IR models, plus the fixed
"camera smudge" pattern. Saves a PNG so we can actually look at the data instead
of only trusting the numbers.

Run:  python -m collapseprobe.visualize_data
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import Circle    # noqa: E402

from synthetic.sequence import SequenceConfig, generate_sequence  # noqa: E402
from collapseprobe.ir_noise import (  # noqa: E402
    IR3DNoiseConfig, ScintillationConfig, apply_scintillation, make_fixed_pattern, total_noise,
)

SNR_DB = -3.0
T_SHOW = 16
SEED = 20260529
CANVAS = (64, 64)


CMAP = "gray"   # white-hot: the standard thermal display (hot = bright)
CLUTTER_RMS = 1.0  # soft cloud-like thermal-scene structure for the realistic frame


def _disp(img):
    """Auto-gain display range (like a thermal camera's AGC): 2nd-98th percentile."""
    return dict(vmin=float(np.percentile(img, 2)), vmax=float(np.percentile(img, 98)))


def main():
    # Simple (control): clean Gaussian-PSF + white noise, no scene.
    sample = generate_sequence(SequenceConfig(
        canvas_shape=CANVAS, n_frames=32, target_sigma=1.0, noise_sigma=1.0,
        motion="cv", speed_min_px_per_frame=0.3, speed_max_px_per_frame=1.5,
        boundary_margin_px=4.0, clutter_rms=0.0, snr_db=SNR_DB,
        sequence_type="positive_uav", seed=SEED,
    ))
    pos = sample.positions
    target_clean = np.asarray(sample.target_signal, dtype=np.float32)
    simple = np.asarray(sample.frames, dtype=np.float32)

    # Realistic IR: same drone/trajectory, but with a thermal scene (soft clouds),
    # the fixed camera smudge, the random IR noise, and target twinkle.
    sample_scene = generate_sequence(SequenceConfig(
        canvas_shape=CANVAS, n_frames=32, target_sigma=1.0, noise_sigma=1.0,
        motion="cv", speed_min_px_per_frame=0.3, speed_max_px_per_frame=1.5,
        boundary_margin_px=4.0, clutter_rms=CLUTTER_RMS, snr_db=SNR_DB,
        sequence_type="positive_uav", seed=SEED,
    ))
    clutter = np.asarray(sample_scene.clutter_signal, dtype=np.float32)  # cloud-like scene

    irc = IR3DNoiseConfig()
    fixed = make_fixed_pattern(CANVAS, irc, np.random.default_rng(777))
    rng = np.random.default_rng(SEED ^ 0x5151)
    target_twinkle = apply_scintillation(target_clean, ScintillationConfig(), rng)
    real = (target_twinkle + clutter + total_noise((32,) + CANVAS, fixed, irc, rng)).astype(np.float32)

    y, x = float(pos[T_SHOW, 0]), float(pos[T_SHOW, 1])
    yi, xi = int(round(y)), int(round(x))

    def zoom(img):
        y0, x0 = max(0, yi - 15), max(0, xi - 15)
        return img[y0:y0 + 31, x0:x0 + 31]

    fig, ax = plt.subplots(2, 4, figsize=(15, 7.6))
    rows = [("Simple noise (control)", simple[T_SHOW]),
            ("Realistic IR — white-hot thermal", real[T_SHOW])]
    for r, (label, frame) in enumerate(rows):
        d = _disp(frame)
        ax[r, 0].imshow(frame, cmap=CMAP, **d)
        ax[r, 0].set_title(f"{label}\nfull frame — drone hidden", fontsize=10)
        ax[r, 1].imshow(frame, cmap=CMAP, **d)
        ax[r, 1].add_patch(Circle((x, y), 5, fill=False, edgecolor="cyan", lw=1.8))
        ax[r, 1].set_title("same frame — drone circled", fontsize=10)
        z = zoom(frame)
        ax[r, 2].imshow(z, cmap=CMAP, **_disp(z))
        ax[r, 2].set_title("zoom on the drone", fontsize=10)

    zt = zoom(target_clean[T_SHOW])
    ax[0, 3].imshow(zt, cmap=CMAP, **_disp(zt))
    ax[0, 3].set_title("the drone alone\n(no noise)", fontsize=10)
    ax[1, 3].imshow(fixed, cmap=CMAP, **_disp(fixed))
    ax[1, 3].set_title("the 'camera smudge'\n(same every frame —\nnever averages out)", fontsize=10)

    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Synthetic faint drone at SNR = {SNR_DB:g} dB  —  white-hot thermal view", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = ROOT / "collapseprobe" / "data" / "preview_drone.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
