"""collapseprobe synthetic dataset builder.

Design decision (see PROPOSAL.md): we **reuse** the validated synthetic
generator (`src/synthetic`) and the **exact** evidence + crop-tube pipeline
(`src/tbd`) the trained detector was built on, so the network sees
identically-formatted input. We do NOT fork the physics — the matched-filter
ceiling in `npceiling/` is calibrated to that generator.

Two noise models (`--noise-model`):

  - **awgn**  — clean Gaussian-PSF target + white Gaussian noise. The regime
    where the plain matched filter is provably Neyman-Pearson optimal, so the
    current `npceiling` ceiling is valid. Use this for ceiling-referenced work
    until the whitening-ceiling upgrade lands.

  - **ir3d**  — realistic IR/thermal: reuses the validated target render +
    (optional) colored-Gaussian clutter, but replaces the noise with the NVESD
    3-D noise model (random floor + FIXED-pattern noise that does not integrate
    away) and scintillates the target (lognormal twinkle). Still
    Gaussian-with-known-covariance, so the *whitening* matched filter remains
    the computable optimum (npceiling upgrade — next step). The fixed pattern is
    a single shared realization across the whole dataset (one sensor).

Dataset design (both models): AWGN-only-style **paired** positive / empty
sequences, **oracle ground-truth-centered crop-tubes** (so per-layer probing
isolates where the *network* loses the target, not the front-end), negatives
borrowing the paired positive's trajectory (paired-noise design = the CMF).

Output (git-ignored — regenerate with `python -m collapseprobe.dataset`):

  collapseprobe/data/{noise_model}/probe_snr{snr}.npz   — one per SNR cell
  collapseprobe/data/{noise_model}/manifest.json

Each .npz holds:
  tubes        (2N, T, C, S, S) float32 — network-ready input
  labels       (2N,)            int64   — 1 = target present, 0 = empty
  cmf_scores   (2N,)            float64 — PLAIN matched-filter statistic
                                          (valid ceiling for awgn; reference-only
                                          for ir3d until the whitening MF lands)
  gt_positions (2N, T, 2)       float32 — trajectory each tube was centered on
  snr_db       scalar           float64
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

# Make src/ and the repo root importable, exactly as npceiling does.
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from synthetic.sequence import SequenceConfig, generate_sequence  # noqa: E402
from tbd.crop_tubes import CropTubeConfig, extract_crop_tube  # noqa: E402
from tbd.evidence import EvidenceConfig, compute_evidence_maps  # noqa: E402
from tbd.tracklets import Tracklet  # noqa: E402

from npceiling.theory.matched_filter import cmf_score  # noqa: E402

from collapseprobe.whitening_mf import whitened_template  # noqa: E402

from collapseprobe.ir_noise import (  # noqa: E402
    IR3DNoiseConfig,
    ScintillationConfig,
    apply_scintillation,
    make_fixed_pattern,
    total_noise,
)


DATA_DIR = ROOT / "collapseprobe" / "data"


@dataclass
class ProbeDataConfig:
    """Knobs for the collapseprobe dataset build."""

    # SNR cells. Default = the two pilot cells where npceiling shows the signal
    # survives to the matched filter (CMF Pd 1.00) but the detector underperforms.
    snr_grid: tuple[float, ...] = (-6.0, -3.0)
    n_per_class: int = 200

    # Generator geometry — MUST match npceiling.eval.compute_ceiling.
    n_frames: int = 32
    canvas_shape: tuple[int, int] = (64, 64)
    target_sigma: float = 1.0
    noise_sigma: float = 1.0   # white-noise std for noise_model="awgn"

    # Crop tube — MUST match the trained model (5 channels, 31 px).
    crop_size: int = 31
    channels: tuple[str, ...] = ("raw", "local_z", "matched", "temporal_diff", "evidence")

    # Noise model: "awgn" (clean, npceiling-valid) | "ir3d" (realistic IR/thermal).
    noise_model: str = "awgn"
    ir_noise: IR3DNoiseConfig = field(default_factory=IR3DNoiseConfig)
    scintillation: ScintillationConfig = field(default_factory=ScintillationConfig)
    clutter_rms: float = 0.0   # colored-Gaussian clutter (used only when noise_model="ir3d")

    seed: int = 31337  # same master seed as the npceiling sweep
    out_dir: Path = field(default=DATA_DIR)


def _base_seq_cfg(cfg: ProbeDataConfig) -> SequenceConfig:
    """Base generator config. SNR is defined vs the random noise floor in both
    models (= ``noise_sigma`` for awgn, = ``ir_noise.sigma_tvh`` for ir3d), so
    the SNR axis stays comparable to the existing waterfalls."""
    if cfg.noise_model == "ir3d":
        noise_sigma = cfg.ir_noise.sigma_tvh
        clutter_rms = cfg.clutter_rms
    else:
        noise_sigma = cfg.noise_sigma
        clutter_rms = 0.0
    return SequenceConfig(
        canvas_shape=cfg.canvas_shape,
        n_frames=cfg.n_frames,
        target_sigma=cfg.target_sigma,
        noise_sigma=noise_sigma,
        motion="cv",
        speed_min_px_per_frame=0.3,
        speed_max_px_per_frame=1.5,
        boundary_margin_px=4.0,
        clutter_rms=clutter_rms,
    )


def _oracle_tracklet(positions: np.ndarray) -> Tracklet:
    """A tracklet following the ground-truth trajectory on every frame, fed to
    the project's ``extract_crop_tube`` so the tube is byte-format-identical to
    what the detector trained on — just centered on truth."""
    T = int(positions.shape[0])
    return Tracklet(
        positions=positions.astype(np.float32, copy=True),
        candidate_scores=np.ones(T, dtype=np.float32),
        start_t=0,
        end_t=T,
        score=1.0,
        miss_count=0,
        metadata={"n_observed": T, "oracle": True},
    )


def _oracle_tube(frames, positions, evidence_cfg, crop_cfg) -> np.ndarray:
    """(T,C,S,S) crop tube centered on ``positions`` via the project pipeline."""
    evidence = compute_evidence_maps(frames, evidence_cfg)
    tube = extract_crop_tube(frames, evidence, _oracle_tracklet(positions), crop_cfg)
    return tube["crop_tube"].numpy().astype(np.float32, copy=False)


def _compose_observation(sample, fixed_pattern, cfg: ProbeDataConfig, noise_seed: int) -> np.ndarray:
    """Observed frames for one sequence under the chosen noise model.

    awgn: the generator's composed frames as-is (npceiling-valid clean case).
    ir3d: validated target render (scintillated) + colored clutter + 3-D IR noise
          (shared fixed pattern + per-sequence random). For empty sequences the
          target is zero, so this is clutter + IR noise only.
    """
    if cfg.noise_model != "ir3d":
        return np.asarray(sample.frames, dtype=np.float32)
    rng = np.random.default_rng(noise_seed)
    T, H, W = sample.frames.shape
    target = apply_scintillation(np.asarray(sample.target_signal, dtype=np.float32), cfg.scintillation, rng)
    clutter = np.asarray(sample.clutter_signal, dtype=np.float32)  # zeros if clutter_rms=0
    noise = total_noise((T, H, W), fixed_pattern, cfg.ir_noise, rng)
    return (target + clutter + noise).astype(np.float32)


def build_split_for_snr(cfg, snr_db, rng, fixed_pattern=None) -> dict:
    """Build the label-balanced tube set for one SNR cell (paired pos/neg)."""
    base = _base_seq_cfg(cfg)
    evidence_cfg = EvidenceConfig()
    crop_cfg = CropTubeConfig(crop_size=cfg.crop_size, channels=tuple(cfg.channels))

    # The plain MF (cmf) is the NP-optimum only in white noise. Under ir3d the
    # noise is correlated, so the *whitening* MF is the valid ceiling — solve it
    # per trajectory and store it next to cmf. For awgn the two coincide.
    is_ir3d = cfg.noise_model == "ir3d"

    tubes: list[np.ndarray] = []
    labels: list[int] = []
    cmf_scores: list[float] = []
    wmf_scores: list[float] = []
    gt_positions: list[np.ndarray] = []

    for _ in range(cfg.n_per_class):
        seed_pos = int(rng.integers(1, 2**31 - 1))
        seed_neg = int(rng.integers(1, 2**31 - 1))
        pos = generate_sequence(replace(base, snr_db=snr_db, sequence_type="positive_uav", seed=seed_pos))
        neg = generate_sequence(replace(base, snr_db=snr_db, sequence_type="empty_background", seed=seed_neg))

        traj = pos.positions  # (T,2) finite for AWGN-only positives (no dropout)
        obs_pos = _compose_observation(pos, fixed_pattern, cfg, noise_seed=seed_pos ^ 0x5151)
        obs_neg = _compose_observation(neg, fixed_pattern, cfg, noise_seed=seed_neg ^ 0x5151)

        # Whitening template, solved once on the (shared) positive trajectory and
        # applied to both members of the paired pos/neg, mirroring cmf's reuse.
        if is_ir3d:
            w_white, _, _ = whitened_template(tuple(cfg.canvas_shape), pos.positions,
                                              pos.target_visible, cfg.target_sigma, cfg.ir_noise)

        # Positive: target present, centered on the true trajectory.
        tubes.append(_oracle_tube(obs_pos, traj, evidence_cfg, crop_cfg))
        labels.append(1)
        cmf_p = cmf_score(obs_pos, pos.positions, pos.target_visible, sigma_psf=cfg.target_sigma)
        cmf_scores.append(cmf_p)
        wmf_scores.append(float((w_white * obs_pos).sum()) if is_ir3d else cmf_p)
        gt_positions.append(traj.astype(np.float32))

        # Negative: noise-only frames, SAME (borrowed) trajectory — paired-noise design.
        tubes.append(_oracle_tube(obs_neg, traj, evidence_cfg, crop_cfg))
        labels.append(0)
        cmf_n = cmf_score(obs_neg, pos.positions, pos.target_visible, sigma_psf=cfg.target_sigma)
        cmf_scores.append(cmf_n)
        wmf_scores.append(float((w_white * obs_neg).sum()) if is_ir3d else cmf_n)
        gt_positions.append(traj.astype(np.float32))

    return {
        "tubes": np.stack(tubes).astype(np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "cmf_scores": np.asarray(cmf_scores, dtype=np.float64),
        "wmf_scores": np.asarray(wmf_scores, dtype=np.float64),
        "gt_positions": np.stack(gt_positions).astype(np.float32),
        "snr_db": np.float64(snr_db),
    }


def _split_path(out_dir: Path, snr_db: float) -> Path:
    return Path(out_dir) / f"probe_snr{snr_db:+05.1f}.npz"


def build_probe_dataset(cfg: ProbeDataConfig | None = None, verbose: bool = True) -> dict:
    """Build + cache the full collapseprobe dataset. Returns the manifest dict."""
    cfg = cfg or ProbeDataConfig()
    out_dir = Path(cfg.out_dir) / cfg.noise_model
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    # ir3d uses ONE shared fixed pattern across all sequences/SNRs (one sensor).
    fixed_pattern = None
    if cfg.noise_model == "ir3d":
        fixed_pattern = make_fixed_pattern(tuple(cfg.canvas_shape), cfg.ir_noise,
                                           np.random.default_rng(cfg.seed + 777))

    cells: list[dict] = []
    t_all = time.time()
    for snr_db in cfg.snr_grid:
        t0 = time.time()
        split = build_split_for_snr(cfg, float(snr_db), rng, fixed_pattern=fixed_pattern)
        path = _split_path(out_dir, float(snr_db))
        np.savez_compressed(path, **split)

        lab, cmf, wmf = split["labels"], split["cmf_scores"], split["wmf_scores"]
        pos_mean, neg_mean = float(cmf[lab == 1].mean()), float(cmf[lab == 0].mean())
        wmf_sep = float(wmf[lab == 1].mean() - wmf[lab == 0].mean())
        size_mb = path.stat().st_size / 1e6
        cells.append({
            "snr_db": float(snr_db), "file": path.name, "n_tubes": int(lab.shape[0]),
            "tube_shape": list(split["tubes"].shape[1:]),
            "cmf_mean_pos": pos_mean, "cmf_mean_neg": neg_mean, "cmf_separation": pos_mean - neg_mean,
            "wmf_separation": wmf_sep,
            "size_mb": round(size_mb, 1), "build_s": round(time.time() - t0, 1),
        })
        if verbose:
            extra = f" | whiten-MF sep={wmf_sep:+.2f}" if cfg.noise_model == "ir3d" else ""
            print(f"SNR={snr_db:+5.1f} dB | {lab.shape[0]} tubes {tuple(split['tubes'].shape[1:])} | "
                  f"plain-MF sep={pos_mean - neg_mean:+.2f} (pos {pos_mean:+.2f} / neg {neg_mean:+.2f})"
                  f"{extra} | {size_mb:.1f} MB | {time.time() - t0:.1f}s", flush=True)

    manifest = {
        "noise_model": cfg.noise_model,
        "config": {
            "snr_grid": list(cfg.snr_grid), "n_per_class": cfg.n_per_class, "n_frames": cfg.n_frames,
            "canvas_shape": list(cfg.canvas_shape), "target_sigma": cfg.target_sigma,
            "crop_size": cfg.crop_size, "channels": list(cfg.channels), "seed": cfg.seed,
        },
        "cells": cells,
        "total_build_s": round(time.time() - t_all, 1),
    }
    if cfg.noise_model == "ir3d":
        manifest["ir_noise"] = vars(cfg.ir_noise).copy()
        manifest["scintillation"] = vars(cfg.scintillation).copy()
        manifest["clutter_rms"] = cfg.clutter_rms
        manifest["note"] = ("Realistic IR; the PLAIN matched filter (cmf_scores) is suboptimal under "
                            "fixed-pattern/colored noise. Valid ceiling = whitening MF (wmf_scores), "
                            "computed via collapseprobe.whitening_mf.")
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    if verbose:
        print(f"\nManifest -> {out_dir / 'manifest.json'} | total {manifest['total_build_s']}s", flush=True)
    return manifest


def load_split(path: str | Path) -> dict:
    """Load one cached SNR cell into a plain dict of arrays."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the collapseprobe synthetic dataset.")
    ap.add_argument("--noise-model", type=str, default=None, choices=["awgn", "ir3d"])
    ap.add_argument("--snr", type=str, default=None, help="comma-separated SNR cells, e.g. '-6,-3'.")
    ap.add_argument("--n-per-class", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cfg = ProbeDataConfig()
    if args.noise_model is not None:
        cfg = replace(cfg, noise_model=args.noise_model)
    if args.snr is not None:
        cfg = replace(cfg, snr_grid=tuple(float(s) for s in args.snr.split(",")))
    if args.n_per_class is not None:
        cfg = replace(cfg, n_per_class=int(args.n_per_class))
    if args.out is not None:
        cfg = replace(cfg, out_dir=Path(args.out))

    print("=== collapseprobe dataset build ===", flush=True)
    print(f"noise_model={cfg.noise_model} | SNR grid: {cfg.snr_grid} | n_per_class: {cfg.n_per_class} | "
          f"T={cfg.n_frames} canvas={cfg.canvas_shape} | oracle-centered, paired", flush=True)
    build_probe_dataset(cfg, verbose=True)


if __name__ == "__main__":
    main()


__all__ = ["ProbeDataConfig", "build_probe_dataset", "build_split_for_snr", "load_split"]
