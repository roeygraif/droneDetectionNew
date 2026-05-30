"""Reusable detectability + linear-probe helpers for collapseprobe experiments.

Kept small and dependency-free (numpy + the project's existing pd_at_far) so every
experiment measures detectability the same way. Re-used by exp01 and later by the
layer-wise probing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training.metrics import pd_at_far  # noqa: E402


def roc_auc(pos, neg) -> float:
    """Area under the ROC curve = P(score_pos > score_neg), ties count half."""
    pos = np.asarray(pos, dtype=float)[:, None]
    neg = np.asarray(neg, dtype=float)[None, :]
    greater = (pos > neg).mean()
    equal = (pos == neg).mean()
    return float(greater + 0.5 * equal)


def dprime(pos, neg) -> float:
    """Detection-theory deflection: mean separation over pooled std."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    s = np.sqrt(0.5 * (pos.var() + neg.var())) + 1e-12
    return float((pos.mean() - neg.mean()) / s)


def detectability(pos_scores, neg_scores, far: float = 1e-2) -> dict:
    """Three numbers for one detector: ROC-AUC, d', and Pd at a fixed FAR."""
    pos = [float(v) for v in np.asarray(pos_scores).ravel()]
    neg = [float(v) for v in np.asarray(neg_scores).ravel()]
    pd = pd_at_far(pos, neg, target_far=far, n_normalization=max(1, len(neg)))
    return {
        "auc": roc_auc(pos, neg),
        "dprime": dprime(pos, neg),
        "pd_at_far": float(pd["pd"]),
    }


def fisher_lda_scores(X_train, y_train, X_test, ridge: float = 1e-2) -> np.ndarray:
    """Best linear detector (Fisher LDA): w = Sigma^-1 (mu1 - mu0), score = x . w.

    This is the linear matched filter on the feature vectors, so it is the natural
    "best linear readout" of a representation. Features are standardized by the
    train statistics first. Returns the projected scores for X_test.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train)

    mu = X_train.mean(0)
    sd = X_train.std(0) + 1e-8
    Xtr = (X_train - mu) / sd
    Xte = (X_test - mu) / sd

    mu1 = Xtr[y_train == 1].mean(0)
    mu0 = Xtr[y_train == 0].mean(0)
    centered = np.vstack([Xtr[y_train == 1] - mu1, Xtr[y_train == 0] - mu0])
    Sigma = np.cov(centered, rowvar=False)
    Sigma = np.atleast_2d(Sigma)
    Sigma = Sigma + ridge * (np.trace(Sigma) / Sigma.shape[0]) * np.eye(Sigma.shape[0])
    w = np.linalg.solve(Sigma, (mu1 - mu0))
    return Xte @ w


def stratified_half_split(labels, seed: int = 0):
    """Return (train_idx, test_idx) with each class split ~50/50, deterministically."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in (0, 1):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        h = len(idx) // 2
        tr.append(idx[:h])
        te.append(idx[h:])
    return np.concatenate(tr), np.concatenate(te)


__all__ = ["roc_auc", "dprime", "detectability", "fisher_lda_scores", "stratified_half_split"]
