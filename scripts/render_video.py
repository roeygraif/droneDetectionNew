"""Render synthetic sequences as MP4 videos (or GIFs by extension).

Three modes:
    --mode single       one sequence, one panel
    --mode snr-grid     four SNR levels side-by-side (the "waterfall feel")
    --mode wildcard     four wildcard configurations side-by-side

Output container is chosen by the --out extension (.mp4, .mov, .gif).

Examples:
    python -m scripts.render_video --mode snr-grid  --out /tmp/snr_grid.mp4
    python -m scripts.render_video --mode wildcard  --out /tmp/wildcard.mp4
    python -m scripts.render_video --mode single --snr-db -10 --out /tmp/hard.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt

import imageio_ffmpeg

from synthetic.sequence import SequenceConfig, SequenceSample, generate_sequence

# Point matplotlib at the bundled ffmpeg so MP4 export works without a
# system-wide install.
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def _save_animation(anim: animation.FuncAnimation, out_path: Path, fps: int, dpi: int) -> None:
    """Pick a writer based on the output extension."""
    suffix = out_path.suffix.lower()
    if suffix in (".mp4", ".mov", ".m4v"):
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium"],
        )
        anim.save(out_path, writer=writer, dpi=dpi)
    elif suffix == ".gif":
        anim.save(out_path, writer="pillow", fps=fps, dpi=dpi)
    else:
        raise ValueError(f"Unsupported output extension: {suffix!r} (use .mp4, .mov, or .gif)")


def _make_grid_gif(
    samples: list[SequenceSample],
    labels: list[str],
    out_path: Path,
    fps: int,
    ncols: int,
) -> None:
    n = len(samples)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 3.3), squeeze=False)
    flat = axes.flatten()

    ims, points, titles = [], [], []
    for i, (s, label) in enumerate(zip(samples, labels)):
        ax = flat[i]
        # Lock display range across all frames so brightness flicker doesn't mask SNR feel.
        vmin, vmax = float(s.frames.min()), float(s.frames.max())
        im = ax.imshow(s.frames[0], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        (pt,) = ax.plot([], [], "r+", ms=12, mew=1.6)
        title = ax.set_title(f"{label}  t=0", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ims.append(im); points.append(pt); titles.append(title)
    for j in range(n, len(flat)):
        flat[j].axis("off")

    n_frames = max(s.frames.shape[0] for s in samples)

    def update(t: int):
        artists = []
        for s, im, pt, title, label in zip(samples, ims, points, titles, labels):
            t_clip = min(t, s.frames.shape[0] - 1)
            im.set_data(s.frames[t_clip])
            pt.set_data([s.positions[t_clip, 1]], [s.positions[t_clip, 0]])
            title.set_text(f"{label}  t={t_clip}")
            artists.extend([im, pt, title])
        return artists

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000 / fps)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_animation(anim, out_path, fps=fps, dpi=120)
    plt.close(fig)


def _make_single_gif(sample: SequenceSample, out_path: Path, fps: int) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    vmin, vmax = float(sample.frames.min()), float(sample.frames.max())
    im = ax.imshow(sample.frames[0], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    (pt,) = ax.plot([], [], "r+", ms=15, mew=2.0)
    title = ax.set_title("")
    ax.set_xticks([]); ax.set_yticks([])

    def update(t: int):
        im.set_data(sample.frames[t])
        pt.set_data([sample.positions[t, 1]], [sample.positions[t, 0]])
        title.set_text(
            f"t={t}  SNR_prescribed={sample.snr_db_prescribed:+.1f} dB  "
            f"observed={'yes' if sample.observed[t] else 'NO'}"
        )
        return im, pt, title

    anim = animation.FuncAnimation(
        fig, update, frames=sample.frames.shape[0], blit=False, interval=1000 / fps
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_animation(anim, out_path, fps=fps, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["single", "snr-grid", "wildcard"], default="snr-grid")
    parser.add_argument("--out", type=Path, default=Path("/tmp/sample.mp4"))
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snr-db", type=float, default=-5.0, help="for --mode single")
    args = parser.parse_args()

    if args.mode == "single":
        cfg = SequenceConfig(
            n_frames=args.n,
            snr_db=args.snr_db,
            speed_max_px_per_frame=1.0,
            seed=args.seed,
        )
        _make_single_gif(generate_sequence(cfg), args.out, args.fps)

    elif args.mode == "snr-grid":
        snrs = [5.0, -5.0, -10.0, -15.0]
        samples = [
            generate_sequence(
                SequenceConfig(
                    n_frames=args.n,
                    snr_db=snr,
                    speed_max_px_per_frame=1.0,
                    seed=args.seed,
                )
            )
            for snr in snrs
        ]
        labels = [f"SNR = {snr:+.0f} dB" for snr in snrs]
        _make_grid_gif(samples, labels, args.out, args.fps, ncols=2)

    elif args.mode == "wildcard":
        configs = [
            (
                "clean +5 dB",
                SequenceConfig(
                    n_frames=args.n, snr_db=5.0, speed_max_px_per_frame=1.0, seed=args.seed
                ),
            ),
            (
                "maneuvers, -5 dB",
                SequenceConfig(
                    n_frames=args.n,
                    snr_db=-5.0,
                    maneuvers=True,
                    maneuver_prob=0.15,
                    heading_std_rad=0.6,
                    speed_max_px_per_frame=1.0,
                    seed=args.seed,
                ),
            ),
            (
                "drifting clutter, -5 dB",
                SequenceConfig(
                    n_frames=args.n,
                    snr_db=-5.0,
                    clutter_rms=0.6,
                    clutter_drift_px_per_frame=0.4,
                    speed_max_px_per_frame=1.0,
                    seed=args.seed,
                ),
            ),
            (
                "30% dropout, 0 dB",
                SequenceConfig(
                    n_frames=args.n,
                    snr_db=0.0,
                    dropout_prob=0.3,
                    speed_max_px_per_frame=1.0,
                    seed=args.seed,
                ),
            ),
        ]
        samples = [generate_sequence(cfg) for _, cfg in configs]
        labels = [label for label, _ in configs]
        _make_grid_gif(samples, labels, args.out, args.fps, ncols=2)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
