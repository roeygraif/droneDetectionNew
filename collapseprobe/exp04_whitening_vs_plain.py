"""Experiment 04: the right ceiling for IR3D — whitening vs. plain matched filter.

Under the IR 3-D noise model the noise is correlated (fixed pattern + flicker +
stripes), so the plain matched filter (`cmf_score`, what the dataset stores) is
*not* the optimum. The whitening matched filter (`whitening_mf.wmf_score`) is.
This experiment regenerates IR3D cells across the dynamic-range band and measures
detectability of both on the same paired pos/neg frames, plus an honest
(non-whitening) temporal integrator for reference.

Claim under test: d'(whiten) > d'(plain) under IR3D, with the gap set by how much
fixed-pattern energy the whitening removes. Sanity: with the fixed/flicker terms
zeroed (pure white floor) the two must coincide.

Run:  python -m collapseprobe.exp04_whitening_vs_plain
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collapseprobe.dataset import (  # noqa: E402
    ProbeDataConfig, _base_seq_cfg, _compose_observation,
)
from collapseprobe.ir_noise import IR3DNoiseConfig, ScintillationConfig, make_fixed_pattern  # noqa: E402
from collapseprobe.probing import detectability  # noqa: E402
from collapseprobe.whitening_mf import whitened_template  # noqa: E402
from synthetic.sequence import generate_sequence  # noqa: E402
from npceiling.theory.matched_filter import cmf_score  # noqa: E402

SNRS = (-9.0, -12.0, -15.0)
N_PER_CLASS = 150


def gen_cell(snr_db, cfg, fixed, rng):
    """Generate one IR3D cell; return per-detector paired scores."""
    base = _base_seq_cfg(cfg)
    out = {k: {"pos": [], "neg": []} for k in ("cmf", "wmf", "honest")}
    cg_iters = []
    for _ in range(cfg.n_per_class):
        sp = int(rng.integers(1, 2**31 - 1))
        sn = int(rng.integers(1, 2**31 - 1))
        pos = generate_sequence(replace(base, snr_db=snr_db, sequence_type="positive_uav", seed=sp))
        neg = generate_sequence(replace(base, snr_db=snr_db, sequence_type="empty_background", seed=sn))
        traj, vis = pos.positions, pos.target_visible
        xp = _compose_observation(pos, fixed, cfg, noise_seed=sp ^ 0x5151)
        xn = _compose_observation(neg, fixed, cfg, noise_seed=sn ^ 0x5151)

        # Plain matched filter (the dataset's stored ceiling).
        out["cmf"]["pos"].append(cmf_score(xp, traj, vis, cfg.target_sigma))
        out["cmf"]["neg"].append(cmf_score(xn, traj, vis, cfg.target_sigma))
        # Whitening matched filter — solve w once, apply to both members of the pair.
        w, s, info = whitened_template(cfg.canvas_shape, traj, vis, cfg.target_sigma, cfg.ir_noise)
        cg_iters.append(info["iters"])
        out["wmf"]["pos"].append(float((w * xp).sum()))
        out["wmf"]["neg"].append(float((w * xn).sum()))
        # Honest readout: temporal sum of the raw frame at the oracle center (no
        # template shape, no whitening) — a fair detector that knows only location.
        out["honest"]["pos"].append(_center_track_sum(xp, traj, vis))
        out["honest"]["neg"].append(_center_track_sum(xn, traj, vis))
    return out, float(np.mean(cg_iters))


def _center_track_sum(frames, positions, visible):
    """Sum the 3x3 patch at the (rounded) oracle center over visible frames."""
    T, H, W = frames.shape
    acc = 0.0
    for t in range(T):
        if not bool(visible[t]):
            continue
        r = int(round(float(positions[t, 0]))); c = int(round(float(positions[t, 1])))
        r0, r1 = max(0, r - 1), min(H, r + 2)
        c0, c1 = max(0, c - 1), min(W, c + 2)
        acc += float(frames[t, r0:r1, c0:c1].sum())
    return acc


def run(noise_cfg, scint_cfg, tag):
    print(f"\n=== {tag} ===")
    print(f"  noise: sigma_tvh={noise_cfg.sigma_tvh} sigma_vh(FPN)={noise_cfg.sigma_vh} "
          f"sigma_v={noise_cfg.sigma_v} sigma_h={noise_cfg.sigma_h} sigma_t={noise_cfg.sigma_t}")
    print(f"  {'SNR':>5} | {'plain MF d′/AUC/Pd':^24} | {'WHITEN MF d′/AUC/Pd':^24} | "
          f"{'honest d′/AUC/Pd':^22} | {'d′ gain':>7} | {'CGit':>5}")
    print("  " + "-" * 100)
    for snr in SNRS:
        cfg = ProbeDataConfig(noise_model="ir3d", snr_grid=(snr,), n_per_class=N_PER_CLASS,
                              ir_noise=noise_cfg, scintillation=scint_cfg, seed=20260530)
        rng = np.random.default_rng(cfg.seed)
        fixed = make_fixed_pattern(cfg.canvas_shape, cfg.ir_noise, np.random.default_rng(cfg.seed + 7))
        cell, cgit = gen_cell(snr, cfg, fixed, rng)
        m = {k: detectability(v["pos"], v["neg"]) for k, v in cell.items()}
        def f(d):
            return f"{d['dprime']:>5.2f}/{d['auc']:.3f}/{d['pd_at_far']:.2f}"
        gain = m["wmf"]["dprime"] / (m["cmf"]["dprime"] + 1e-9)
        print(f"  {snr:>5.0f} | {f(m['cmf']):^24} | {f(m['wmf']):^24} | {f(m['honest']):^22} | "
              f"{gain:>6.3f}x | {cgit:>5.0f}")


def main():
    t0 = time.time()
    # Sanity: pure white floor — whitening MF must coincide with the plain MF.
    white = IR3DNoiseConfig(sigma_tvh=1.0, sigma_t=0, sigma_tv=0, sigma_th=0, sigma_vh=0, sigma_v=0, sigma_h=0)
    run(white, ScintillationConfig(enabled=False), "SANITY: pure white floor (whiten should == plain)")
    # The real thing: default IR3D noise (FPN + stripes + flicker), scintillation on.
    run(IR3DNoiseConfig(), ScintillationConfig(), "IR3D default (FPN + stripes + flicker + scintillation)")
    # Stress: heavier fixed pattern — whitening should help more.
    run(IR3DNoiseConfig(sigma_vh=0.6, sigma_v=0.25, sigma_h=0.18), ScintillationConfig(),
        "IR3D heavy fixed-pattern (3x FPN)")
    print(f"\n(total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
