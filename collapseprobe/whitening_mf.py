"""Whitening (generalized) matched filter — the optimum for the IR3D regime.

For a known signal in *correlated* Gaussian noise the Neyman-Pearson optimal
detector is the whitening matched filter (Kay 1998, Vol. II Sec. 5):

    w = C^{-1} s ,    statistic  T = w^T x ,    detectability  d' = sqrt(s^T C^{-1} s)

where ``s`` is the stacked known signal (the moving Gaussian target rendered on
the visible frames) and ``C`` is the known noise covariance of the stacked
(T,H,W) observation. The plain matched filter (``npceiling ... cmf_score``)
assumes ``C = sigma^2 I`` and is therefore *suboptimal* under the IR 3-D noise
model — that gap is what this module measures.

We never form ``C`` (it is ThW x ThW ~ 1.3e5 square). Instead we apply ``C`` as
a fast structured operator (``_cov_matvec``) built directly from the documented
covariance in ``ir_noise`` (its module docstring), and solve ``C w = s`` with
conjugate gradient. The signal template reuses ``render_unit_template`` from
``npceiling`` verbatim, so the target physics is not forked.

Covariance terms applied (all from ``IR3DNoiseConfig``), for entry (t,r,c):
    sigma_tvh^2  white floor               -> diagonal (or AR(1) along t if rho>0)
    sigma_t^2    flicker (per-frame DC)     -> couples all pixels within a frame
    sigma_tv^2   per-frame column noise     -> couples rows within (frame, col)
    sigma_th^2   per-frame row noise        -> couples cols within (frame, row)
    sigma_vh^2   fixed per-pixel FPN        -> couples all frames at a pixel
    sigma_v^2    fixed column stripes       -> couples all frames & rows at a col
    sigma_h^2    fixed row stripes          -> couples all frames & cols at a row

Intuition (per-pixel block): floor + FPN give C_p = sigma_tvh^2 I + sigma_vh^2 11^T
per pixel, whose inverse (Sherman-Morrison) *subtracts the per-pixel temporal
mean* — i.e. whitening rejects the fixed pattern that temporal averaging cannot.
The full operator adds the (smaller) cross-pixel stripe/flicker coupling exactly.

Pure numpy, no torch / scipy, matching ``ir_noise``'s style.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from npceiling.theory.matched_filter import render_unit_template  # noqa: E402

from collapseprobe.ir_noise import IR3DNoiseConfig  # noqa: E402


def _ar1_temporal_matmul(v: np.ndarray, rho: float) -> np.ndarray:
    """Apply the unit-marginal AR(1) temporal covariance Toeplitz (rho^|t-t'|)
    along axis 0 of v (T,H,W). T is small, so an explicit (T,T) matrix is fine."""
    T = v.shape[0]
    idx = np.arange(T)
    K = rho ** np.abs(idx[:, None] - idx[None, :])  # (T,T)
    return np.tensordot(K, v, axes=(1, 0))          # (T,H,W)


def _cov_matvec(v: np.ndarray, cfg: IR3DNoiseConfig) -> np.ndarray:
    """Apply the IR3D noise covariance C to a (T,H,W) array. Exact, structured."""
    c = cfg
    # White spatio-temporal floor (diagonal, or AR(1) in time if requested).
    if c.temporal_1f_rho > 0.0:
        out = c.sigma_tvh ** 2 * _ar1_temporal_matmul(v, float(c.temporal_1f_rho))
    else:
        out = (c.sigma_tvh ** 2) * v
    # Random per-frame components (within-frame coupling).
    if c.sigma_t > 0:    # flicker: sum over all pixels in the frame
        out = out + c.sigma_t ** 2 * v.sum(axis=(1, 2), keepdims=True)
    if c.sigma_tv > 0:   # per-frame column: sum over rows
        out = out + c.sigma_tv ** 2 * v.sum(axis=1, keepdims=True)
    if c.sigma_th > 0:   # per-frame row: sum over cols
        out = out + c.sigma_th ** 2 * v.sum(axis=2, keepdims=True)
    # Fixed (cross-frame) components.
    if c.sigma_vh > 0:   # fixed per-pixel: sum over frames
        out = out + c.sigma_vh ** 2 * v.sum(axis=0, keepdims=True)
    if c.sigma_v > 0:    # fixed column: sum over frames and rows
        out = out + c.sigma_v ** 2 * v.sum(axis=(0, 1), keepdims=True)
    if c.sigma_h > 0:    # fixed row: sum over frames and cols
        out = out + c.sigma_h ** 2 * v.sum(axis=(0, 2), keepdims=True)
    return out


def _conjugate_gradient(matvec, b: np.ndarray, tol: float = 1e-8, maxiter: int = 1000):
    """Solve A x = b for SPD A given a matvec, starting from 0. Returns (x, info)."""
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs_old = float(np.vdot(r, r).real)
    bnorm = float(np.sqrt(np.vdot(b, b).real)) + 1e-300
    iters = 0
    for iters in range(1, maxiter + 1):
        Ap = matvec(p)
        alpha = rs_old / (float(np.vdot(p, Ap).real) + 1e-300)
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(np.vdot(r, r).real)
        if np.sqrt(rs_new) / bnorm < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    rel_res = float(np.sqrt(np.vdot(r, r).real)) / bnorm
    return x, {"iters": iters, "rel_res": rel_res}


def build_signal_tube(
    canvas_shape: tuple[int, int],
    positions: np.ndarray,
    target_visible: np.ndarray,
    sigma_psf: float,
) -> np.ndarray:
    """Stacked known signal s as (T,H,W): unit-peak target template on each visible
    frame, zero elsewhere. Same template render as the plain matched filter."""
    T = positions.shape[0]
    H, W = canvas_shape
    s = np.zeros((T, H, W), dtype=np.float64)
    for t in range(T):
        if not bool(target_visible[t]):
            continue
        cy, cx = float(positions[t, 0]), float(positions[t, 1])
        if not (np.isfinite(cy) and np.isfinite(cx)):
            continue
        s[t] = render_unit_template((H, W), cy, cx, sigma_psf)
    return s


def whitened_template(
    canvas_shape: tuple[int, int],
    positions: np.ndarray,
    target_visible: np.ndarray,
    sigma_psf: float,
    noise_cfg: IR3DNoiseConfig,
    tol: float = 1e-8,
    maxiter: int = 1000,
):
    """Return (w, s, info): the whitened template w = C^{-1} s and the raw signal s.

    ``info`` carries CG diagnostics and the theoretical detectability
    d'_theory = sqrt(s . w) = sqrt(s^T C^{-1} s)."""
    s = build_signal_tube(canvas_shape, positions, target_visible, sigma_psf)
    w, cg = _conjugate_gradient(lambda v: _cov_matvec(v, noise_cfg), s, tol=tol, maxiter=maxiter)
    dprime_theory = float(np.sqrt(max(0.0, float(np.sum(s * w)))))
    return w, s, {**cg, "dprime_theory": dprime_theory}


def wmf_score(
    frames: np.ndarray,
    positions: np.ndarray,
    target_visible: np.ndarray,
    sigma_psf: float,
    noise_cfg: IR3DNoiseConfig,
    whitened: np.ndarray | None = None,
) -> float:
    """Whitening matched-filter statistic T = w^T x for one sequence.

    Pass a precomputed ``whitened`` template w (from ``whitened_template``) to
    score many frame-sets (e.g. the paired negative) against the same trajectory
    without re-solving — this mirrors how ``cmf_score`` is reused for the paired
    noise-only baseline.
    """
    if whitened is None:
        H, W = frames.shape[1], frames.shape[2]
        whitened, _, _ = whitened_template((H, W), positions, target_visible, sigma_psf, noise_cfg)
    return float(np.sum(whitened * frames))


__all__ = [
    "build_signal_tube",
    "whitened_template",
    "wmf_score",
    "_cov_matvec",
    "_conjugate_gradient",
]
