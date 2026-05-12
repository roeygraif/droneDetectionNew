"""PyTorch dataset wrappers.

- :class:`SyntheticDroneDataset` -- on-the-fly random generation for training,
   sampling SNR, N, and *sequence_type* per-item from configurable mixtures.
- :class:`CachedEvalDataset`     -- deterministic regeneration from a seed
   manifest, for the waterfall sweep. Disk footprint: a few KB. The manifest
   now enumerates a sequence_type axis alongside SNR / N / run, so eval can
   measure false-alarm rate (on empty / hard-negative cells) as well as
   probability of detection.

Note on torch collation: ``distractor_tracks`` and ``sequence_type`` are
non-tensor (Python list of dicts, string) — fine with ``batch_size=1``, but
training loops batching > 1 need a custom collate.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from synthetic.sequence import SEQUENCE_TYPES, SequenceConfig, SequenceSample, generate_sequence


_TUPLE_FIELDS = ("canvas_shape", "distractor_types")
_ENTRY_FIELDS_FORBIDDEN_IN_OVERRIDES = frozenset({"snr_db", "n_frames", "seed", "sequence_type"})

_DEFAULT_TYPE_MIX = {
    "positive_uav": 0.5,
    "empty_background": 0.2,
    "hard_negative": 0.2,
    "mixed_uav_and_distractors": 0.1,
}


def _track_to_serializable(track) -> dict:
    """Convert a :class:`DistractorTrack` to a plain dict with numpy arrays preserved."""
    return asdict(track)


def _sample_to_torch(sample: SequenceSample) -> dict:
    """Convert a :class:`SequenceSample` to a torch-friendly dict.

    Non-tensor fields (``sequence_type``, ``distractor_tracks``) are returned
    as Python objects; downstream code with ``batch_size > 1`` should supply a
    custom collate fn.
    """
    return {
        "frames": torch.from_numpy(sample.frames),
        "target_signal": torch.from_numpy(sample.target_signal),
        "target_signal_clean": torch.from_numpy(sample.target_signal_clean),
        "distractor_signal": torch.from_numpy(sample.distractor_signal),
        "clutter_signal": torch.from_numpy(sample.clutter_signal),
        "awgn_field": torch.from_numpy(sample.awgn_field),
        "positions": torch.from_numpy(sample.positions),
        "target_visible": torch.from_numpy(sample.target_visible.astype(np.uint8)),
        "mask_soft": torch.from_numpy(sample.mask_soft),
        "mask_hard": torch.from_numpy(sample.mask_hard.astype(np.uint8)),
        "observed": torch.from_numpy(sample.observed.astype(np.uint8)),
        "has_target": torch.tensor(bool(sample.has_target)),
        "sequence_type": sample.sequence_type,
        "distractor_tracks": [_track_to_serializable(t) for t in sample.distractor_tracks],
        "snr_db_prescribed": torch.tensor(sample.snr_db_prescribed),
        "snr_db_measured": torch.tensor(sample.snr_db_measured),
        "effective_scnr_db_per_frame": torch.from_numpy(sample.effective_scnr_db_per_frame),
        "effective_scnr_db_mean": torch.tensor(sample.effective_scnr_db_mean),
        "local_background_std_per_frame": torch.from_numpy(sample.local_background_std_per_frame),
    }


def derive_seed(snr_idx: int, n_idx: int, run_idx: int, type_idx: int = 0,
                salt: bytes = b"drone-eval") -> int:
    """Deterministic 32-bit seed for an eval-manifest entry.

    Backwards-compatible: when ``type_idx == 0`` (the only case in pre-existing
    manifests), the seed key matches the original three-coordinate format.
    """
    h = hashlib.blake2b(digest_size=8, key=salt)
    key = (f"{snr_idx}-{n_idx}-{run_idx}"
           if type_idx == 0
           else f"{snr_idx}-{n_idx}-{run_idx}-{type_idx}")
    h.update(key.encode())
    return int.from_bytes(h.digest()[:4], "big")


def _normalize_type_probs(probs: dict[str, float]) -> tuple[tuple[str, ...], np.ndarray]:
    if not probs:
        raise ValueError("sequence_type_probs cannot be empty")
    for k in probs:
        if k not in SEQUENCE_TYPES:
            raise ValueError(f"Unknown sequence_type {k!r} in sequence_type_probs; valid: {SEQUENCE_TYPES}")
    total = float(sum(probs.values()))
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"sequence_type_probs must sum to > 0; got {probs}")
    types = tuple(probs.keys())
    weights = np.array([probs[t] / total for t in types], dtype=np.float64)
    return types, weights


class SyntheticDroneDataset(IterableDataset):
    """On-the-fly random sampler over (SNR, N, sequence_type) for training.

    Yields a stream of :func:`_sample_to_torch` dicts. SNR is sampled uniformly
    from ``snr_range``, N from ``n_choices``, and sequence_type from
    ``sequence_type_probs`` (defaults to a 50/20/20/10 split).
    """

    def __init__(
        self,
        base_config: SequenceConfig,
        snr_range: tuple[float, float] = (-25.0, 5.0),
        n_choices: tuple[int, ...] = (1, 3, 5, 10, 20),
        seed: int = 0,
        n_samples: int | None = None,
        sequence_type_probs: dict[str, float] | None = None,
    ):
        super().__init__()
        self.base_config = base_config
        self.snr_range = snr_range
        self.n_choices = tuple(n_choices)
        self.seed = seed
        self.n_samples = n_samples
        self._types, self._weights = _normalize_type_probs(sequence_type_probs or _DEFAULT_TYPE_MIX)

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rng = np.random.default_rng(np.array([self.seed, worker_id], dtype=np.uint64))
        i = 0
        while self.n_samples is None or i < self.n_samples:
            snr_db = float(rng.uniform(*self.snr_range))
            n_frames = int(rng.choice(self.n_choices))
            seq_type = str(rng.choice(self._types, p=self._weights))
            seq_seed = int(rng.integers(0, 2**31 - 1))
            cfg = replace(
                self.base_config,
                snr_db=snr_db,
                n_frames=n_frames,
                seed=seq_seed,
                sequence_type=seq_type,
            )
            yield _sample_to_torch(generate_sequence(cfg))
            i += 1


class CachedEvalDataset(Dataset):
    """Deterministic eval set driven by a JSON manifest.

    The manifest enumerates the (snr_idx, n_idx, type_idx, run_idx) cells of
    the waterfall sweep; ``__getitem__`` regenerates each sequence from a
    derived seed. ``sequence_types`` is optional in the manifest; if absent it
    defaults to ``["positive_uav"]`` (backwards compatible with pre-existing
    manifests).
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

        sequence_types = list(self.manifest.get("sequence_types") or ["positive_uav"])
        for st in sequence_types:
            if st not in SEQUENCE_TYPES:
                raise ValueError(f"Unknown sequence_type {st!r} in manifest")
        self.sequence_types = sequence_types

        self.entries: list[tuple[int, int, int, int, float, int, str]] = [
            (snr_idx, n_idx, type_idx, run_idx, float(snr_db), int(n_frames), str(seq_type))
            for snr_idx, snr_db in enumerate(self.manifest["snr_grid_db"])
            for n_idx, n_frames in enumerate(self.manifest["n_grid"])
            for type_idx, seq_type in enumerate(sequence_types)
            for run_idx in range(int(self.manifest["monte_carlo_runs_per_cell"]))
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        snr_idx, n_idx, type_idx, run_idx, snr_db, n_frames, seq_type = self.entries[idx]
        seed = derive_seed(snr_idx, n_idx, run_idx, type_idx)
        cfg = SequenceConfig(
            snr_db=snr_db,
            n_frames=n_frames,
            seed=seed,
            sequence_type=seq_type,
            **self.config_overrides,
        )
        return _sample_to_torch(generate_sequence(cfg))
