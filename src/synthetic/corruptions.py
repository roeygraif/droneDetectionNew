"""Wildcard corruption: frame dropout.

Off by default (`dropout_prob=0`); turned on for the resilience study.

Camera jitter is *not* a post-hoc corruption — it has to be modeled at
sequence-construction time because it changes the apparent target position
(and shifts clutter, but not per-pixel readout noise). It lives in
``sequence.py`` for that reason.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def apply_frame_dropout(
    frames: np.ndarray,
    dropout_prob: float,
    mode: Literal["blank", "repeat_last"],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly drop frames at rate ``dropout_prob``.

    Returns ``(frames_out, observed_mask)`` where ``observed_mask`` is a
    ``(N,)`` boolean array indicating which frames were actually observed.
    The first frame is never dropped (so ``repeat_last`` always has a source).
    """
    if not 0.0 <= dropout_prob < 1.0:
        raise ValueError(f"dropout_prob must be in [0, 1), got {dropout_prob}")
    if mode not in ("blank", "repeat_last"):
        raise ValueError(f"mode must be 'blank' or 'repeat_last', got {mode!r}")

    n_frames = frames.shape[0]
    drop = rng.random(n_frames) < dropout_prob
    drop[0] = False
    observed = ~drop

    out = frames.copy()
    last = frames[0]
    for t in range(n_frames):
        if drop[t]:
            out[t] = 0.0 if mode == "blank" else last
        else:
            last = out[t]
    return out, observed
