"""PyTorch dataset wrappers.

- :class:`SyntheticDroneDataset` -- on-the-fly random generation, for training.
- :class:`CachedEvalDataset`     -- deterministic regeneration from a seed
   manifest, for the waterfall sweep. Disk footprint: a few KB (the manifest
   is the only on-disk artifact; frames are reproduced on demand).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from synthetic.sequence import SequenceConfig, SequenceSample, generate_sequence


_TUPLE_FIELDS = ("canvas_shape",)
_ENTRY_FIELDS_FORBIDDEN_IN_OVERRIDES = frozenset({"snr_db", "n_frames", "seed"})


def _sample_to_torch(sample: SequenceSample) -> dict[str, torch.Tensor]:
    """Convert a :class:`SequenceSample` to a torch-friendly dict."""
    return {
        "frames": torch.from_numpy(sample.frames),
        "target_signal": torch.from_numpy(sample.target_signal),
        "positions": torch.from_numpy(sample.positions),
        "mask_soft": torch.from_numpy(sample.mask_soft),
        "mask_hard": torch.from_numpy(sample.mask_hard.astype(np.uint8)),
        "observed": torch.from_numpy(sample.observed.astype(np.uint8)),
        "snr_db_prescribed": torch.tensor(sample.snr_db_prescribed),
        "snr_db_measured": torch.tensor(sample.snr_db_measured),
    }


def derive_seed(snr_idx: int, n_idx: int, run_idx: int, salt: bytes = b"drone-eval") -> int:
    """Deterministic 32-bit seed for an eval-manifest entry."""
    h = hashlib.blake2b(digest_size=8, key=salt)
    h.update(f"{snr_idx}-{n_idx}-{run_idx}".encode())
    return int.from_bytes(h.digest()[:4], "big")


class SyntheticDroneDataset(IterableDataset):
    """On-the-fly random sampler over (SNR, N) for training.

    Yields a stream of :func:`_sample_to_torch` dicts. SNR is sampled uniformly
    from ``snr_range``; N is sampled uniformly from ``n_choices``.
    """

    def __init__(
        self,
        base_config: SequenceConfig,
        snr_range: tuple[float, float] = (-25.0, 5.0),
        n_choices: tuple[int, ...] = (1, 3, 5, 10, 20),
        seed: int = 0,
        n_samples: int | None = None,
    ):
        super().__init__()
        self.base_config = base_config
        self.snr_range = snr_range
        self.n_choices = tuple(n_choices)
        self.seed = seed
        self.n_samples = n_samples

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rng = np.random.default_rng(np.array([self.seed, worker_id], dtype=np.uint64))
        i = 0
        while self.n_samples is None or i < self.n_samples:
            snr_db = float(rng.uniform(*self.snr_range))
            n_frames = int(rng.choice(self.n_choices))
            seq_seed = int(rng.integers(0, 2**31 - 1))
            cfg = replace(self.base_config, snr_db=snr_db, n_frames=n_frames, seed=seq_seed)
            yield _sample_to_torch(generate_sequence(cfg))
            i += 1


class CachedEvalDataset(Dataset):
    """Deterministic eval set driven by a JSON manifest.

    The manifest enumerates the ``(snr_db, n_frames, run_idx)`` cells of the
    waterfall sweep; ``__getitem__`` regenerates each sequence from a derived
    seed. The manifest is the *only* on-disk artifact — full frames are
    reproduced on demand. Use ``--frame-cache`` in
    ``scripts/generate_eval_manifest.py`` if profiling shows regeneration is a
    bottleneck.
    """

    def __init__(self, manifest_path: str | Path):
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        overrides = dict(self.manifest.get("config_overrides", {}))
        forbidden = _ENTRY_FIELDS_FORBIDDEN_IN_OVERRIDES & overrides.keys()
        if forbidden:
            raise ValueError(
                f"config_overrides cannot contain {sorted(forbidden)} "
                f"(those vary per cell)"
            )
        for key in _TUPLE_FIELDS:
            if key in overrides:
                overrides[key] = tuple(overrides[key])
        self.config_overrides = overrides

        self.entries: list[tuple[int, int, int, float, int]] = [
            (snr_idx, n_idx, run_idx, float(snr_db), int(n_frames))
            for snr_idx, snr_db in enumerate(self.manifest["snr_grid_db"])
            for n_idx, n_frames in enumerate(self.manifest["n_grid"])
            for run_idx in range(int(self.manifest["monte_carlo_runs_per_cell"]))
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        snr_idx, n_idx, run_idx, snr_db, n_frames = self.entries[idx]
        seed = derive_seed(snr_idx, n_idx, run_idx)
        cfg = SequenceConfig(snr_db=snr_db, n_frames=n_frames, seed=seed, **self.config_overrides)
        return _sample_to_torch(generate_sequence(cfg))
