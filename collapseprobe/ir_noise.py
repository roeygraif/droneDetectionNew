"""Physics-based IR/thermal noise + scintillation for collapseprobe.

We model a far-away drone in a staring thermal imager. The noise follows the
standard **NVESD 3-D noise model** (D'Agostino & Webb 1991), which decomposes
sensor noise into components along the temporal (t) and spatial (v=row, h=col)
axes. The decomposition matters for *this* thesis because it cleanly separates:

  RANDOM components (vary frame-to-frame) -> average out as 1/sqrt(T):
      sigma_tvh  random spatio-temporal noise (the "noise floor"; ~AWGN)
      sigma_t    frame-uniform flicker (per-frame DC level)
      sigma_tv   per-frame column noise   (vertical streaking, time-varying)
      sigma_th   per-frame row noise       (horizontal streaking, time-varying)

  FIXED components (identical every frame) -> do NOT average out, set a floor:
      sigma_vh   fixed per-pixel pattern  (FPN / NUC residual)  <-- the big one
      sigma_v    fixed column pattern      (fixed vertical stripes)
      sigma_h    fixed row pattern         (fixed horizontal stripes)

For a *moving* target the fixed pattern partially decorrelates along the track
(the target visits different fixed pixels each frame), so motion itself helps
reject FPN — a thesis-relevant effect this model reproduces for free.

Everything here is Gaussian with KNOWN second-order statistics, so the optimal
detector remains computable: it is the *whitening* matched filter (pre-whiten by
the inverse noise covariance, then correlate; Kay 1998, Vol. II §5). The
per-pixel temporal covariance implied by this model is, for pixel p and frames
t != t':

    Cov(n[t,p], n[t',p]) = sigma_vh^2                       (fixed pixel)
                         + sigma_v^2 [same column]          (fixed column)
                         + sigma_h^2 [same row]             (fixed row)
                         + sigma_t^2                        (flicker, all pixels)
                         + sigma_tvh^2 * rho^|t-t'|         (if temporal_1f_rho>0)
    Var(n[t,p]) = sigma_tvh^2 + sigma_vh^2 + sigma_v^2 + sigma_h^2 + sigma_t^2 + ...

The npceiling whitening upgrade (next step) consumes exactly these sigmas.

Scintillation: a far IR target twinkles (atmospheric turbulence + aspect change)
— the optical analog of radar Swerling fluctuation. We model it as a
multiplicative, temporally-correlated lognormal gain with unit mean, so the
prescribed peak SNR is preserved in expectation.

Pure numpy, deterministic given the rng. No torch, no new deps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Configs. Defaults are physics-based for a reasonably (but not perfectly)
# NUC-corrected thermal imager, expressed as multiples of the random floor
# sigma_tvh = 1.0. No real data was used to fit these — they are documented
# starting points (see module docstring) and meant to be tuned / swept.
# ---------------------------------------------------------------------------
@dataclass
class IR3DNoiseConfig:
    # Random (integrating) components.
    sigma_tvh: float = 1.0       # spatio-temporal random floor (reference unit)
    sigma_t: float = 0.05        # frame-uniform flicker
    sigma_tv: float = 0.0        # per-frame column noise (off by default)
    sigma_th: float = 0.0        # per-frame row noise (off by default)
    temporal_1f_rho: float = 0.0  # AR(1) temporal correlation of the random floor

    # Fixed (non-integrating) components — the detection-floor drivers.
    sigma_vh: float = 0.30       # fixed per-pixel FPN / NUC residual
    sigma_v: float = 0.12        # fixed column stripes
    sigma_h: float = 0.08        # fixed row stripes


@dataclass
class ScintillationConfig:
    enabled: bool = True
    scintillation_index: float = 0.15  # std/mean of multiplicative intensity
    temporal_rho: float = 0.6          # AR(1) correlation of log-amplitude


# ---------------------------------------------------------------------------
def make_fixed_pattern(shape_hw: tuple[int, int], cfg: IR3DNoiseConfig, rng: np.random.Generator) -> np.ndarray:
    """Generate the (H,W) FIXED noise pattern (same on every frame).

    This is a *sensor* property — generate it ONCE and share it across every
    sequence in the dataset (one imager). It is the structured nuisance the
    detector must learn to reject and that temporal integration cannot remove.
    """
    H, W = shape_hw
    fixed = np.zeros((H, W), dtype=np.float64)
    if cfg.sigma_vh > 0:
        fixed += rng.normal(0.0, cfg.sigma_vh, size=(H, W))
    if cfg.sigma_v > 0:
        fixed += rng.normal(0.0, cfg.sigma_v, size=(1, W))   # column pattern, broadcast over rows
    if cfg.sigma_h > 0:
        fixed += rng.normal(0.0, cfg.sigma_h, size=(H, 1))   # row pattern, broadcast over cols
    return fixed.astype(np.float32)


def sample_random_noise(shape_thw: tuple[int, int, int], cfg: IR3DNoiseConfig, rng: np.random.Generator) -> np.ndarray:
    """Generate the (T,H,W) RANDOM noise (independent per frame; integrates as 1/sqrt(T))."""
    T, H, W = shape_thw
    out = np.zeros((T, H, W), dtype=np.float64)

    # Spatio-temporal random floor, optionally AR(1)-correlated in time (1/f-ish).
    if cfg.sigma_tvh > 0:
        if cfg.temporal_1f_rho > 0:
            rho = float(cfg.temporal_1f_rho)
            e = rng.normal(0.0, 1.0, size=(T, H, W))
            w = np.empty_like(e)
            w[0] = e[0]
            scale = np.sqrt(1.0 - rho * rho)
            for t in range(1, T):
                w[t] = rho * w[t - 1] + scale * e[t]   # unit marginal variance
            out += cfg.sigma_tvh * w
        else:
            out += rng.normal(0.0, cfg.sigma_tvh, size=(T, H, W))

    if cfg.sigma_t > 0:  # frame-uniform flicker
        out += rng.normal(0.0, cfg.sigma_t, size=(T, 1, 1))
    if cfg.sigma_tv > 0:  # per-frame column noise
        out += rng.normal(0.0, cfg.sigma_tv, size=(T, 1, W))
    if cfg.sigma_th > 0:  # per-frame row noise
        out += rng.normal(0.0, cfg.sigma_th, size=(T, H, 1))

    return out.astype(np.float32)


def total_noise(shape_thw: tuple[int, int, int], fixed_pattern: np.ndarray, cfg: IR3DNoiseConfig, rng: np.random.Generator) -> np.ndarray:
    """Full (T,H,W) IR noise = shared fixed pattern (broadcast over T) + per-frame random."""
    T = shape_thw[0]
    rand = sample_random_noise(shape_thw, cfg, rng)
    return (fixed_pattern[None, :, :] + rand).astype(np.float32)


def total_noise_variance(cfg: IR3DNoiseConfig) -> float:
    """Per-pixel marginal noise variance implied by the config (for SNR bookkeeping)."""
    return float(
        cfg.sigma_tvh ** 2 + cfg.sigma_t ** 2 + cfg.sigma_tv ** 2 + cfg.sigma_th ** 2
        + cfg.sigma_vh ** 2 + cfg.sigma_v ** 2 + cfg.sigma_h ** 2
    )


# ---------------------------------------------------------------------------
def apply_scintillation(target_thw: np.ndarray, cfg: ScintillationConfig, rng: np.random.Generator) -> np.ndarray:
    """Multiply a (T,H,W) target by a unit-mean, temporally-correlated lognormal gain.

    The scintillation index S = std(I)/mean(I) is preserved; E[I] = 1 so the
    prescribed peak SNR holds in expectation (the target twinkles around it).
    """
    if not cfg.enabled or cfg.scintillation_index <= 0.0:
        return target_thw
    T = target_thw.shape[0]
    s = float(cfg.scintillation_index)
    sigma_g = np.sqrt(np.log(1.0 + s * s))   # lognormal: Var(I)/E[I]^2 = exp(sigma_g^2)-1 = s^2

    rho = float(cfg.temporal_rho)
    e = rng.normal(0.0, 1.0, size=T)
    g = np.empty(T, dtype=np.float64)
    g[0] = e[0]
    scale = np.sqrt(max(0.0, 1.0 - rho * rho))
    for t in range(1, T):
        g[t] = rho * g[t - 1] + scale * e[t]     # unit-variance AR(1)
    gain = np.exp(sigma_g * g - 0.5 * sigma_g * sigma_g)   # E[gain] = 1
    return (target_thw * gain[:, None, None]).astype(np.float32)


__all__ = [
    "IR3DNoiseConfig",
    "ScintillationConfig",
    "make_fixed_pattern",
    "sample_random_noise",
    "total_noise",
    "total_noise_variance",
    "apply_scintillation",
]
