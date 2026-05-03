"""Write a JSON manifest describing the eval sweep.

Each cell is identified by ``(snr_idx, n_idx, run_idx)``; seeds are derived
deterministically by :class:`CachedEvalDataset`, so the manifest itself stays
tiny (a few KB) regardless of how many Monte Carlo runs are described.

    python -m scripts.generate_eval_manifest --out eval_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("eval_manifest.json"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "data_default.yaml",
    )
    parser.add_argument("--snr-min", type=float, default=-25.0)
    parser.add_argument("--snr-max", type=float, default=5.0)
    parser.add_argument("--snr-step", type=float, default=2.0)
    parser.add_argument("--n-grid", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--mc-runs", type=int, default=1000)
    parser.add_argument(
        "--maneuvers",
        action="store_true",
        help="Override config to enable target maneuvers across the whole sweep.",
    )
    args = parser.parse_args()

    snr_grid: list[float] = []
    snr = args.snr_min
    while snr <= args.snr_max + 1e-9:
        snr_grid.append(round(snr, 6))
        snr += args.snr_step

    overrides: dict = yaml.safe_load(args.config.read_text())
    for key in ("snr_db", "n_frames", "seed"):
        overrides.pop(key, None)
    if args.maneuvers:
        overrides["maneuvers"] = True

    manifest = {
        "snr_grid_db": snr_grid,
        "n_grid": list(args.n_grid),
        "monte_carlo_runs_per_cell": args.mc_runs,
        "seed_strategy": "blake2b(b'drone-eval', f'{snr_idx}-{n_idx}-{run_idx}')[:4]",
        "config_overrides": overrides,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)

    total = len(snr_grid) * len(args.n_grid) * args.mc_runs
    print(
        f"Wrote {args.out} | {len(snr_grid)} SNR x {len(args.n_grid)} N x "
        f"{args.mc_runs} MC = {total} sequences."
    )


if __name__ == "__main__":
    main()
