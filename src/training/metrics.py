"""Lightweight metrics: ROC-AUC, PR-AUC, precision, recall, FPR at a threshold.

Pure numpy. We avoid scikit-learn so the project's dependency footprint stays
small.
"""

from __future__ import annotations

import numpy as np


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x).astype(np.float64).reshape(-1)


def roc_auc(scores, labels) -> float:
    """ROC-AUC via the rank-sum identity. Handles ties via average ranks."""
    s = _to_np(scores)
    y = _to_np(labels)
    if s.size == 0:
        return float("nan")
    n_pos = int((y > 0.5).sum())
    n_neg = int((y <= 0.5).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(s)
    # average ranks for ties
    s_sorted = s[order]
    i = 0
    n = s.size
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-indexed
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    rank_sum_pos = float(ranks[y > 0.5].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def pr_auc(scores, labels) -> float:
    """PR-AUC via the standard step-function integration."""
    s = _to_np(scores)
    y = _to_np(labels)
    if s.size == 0 or (y > 0.5).sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted > 0.5)
    fp = np.cumsum(y_sorted <= 0.5)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(1.0, float((y > 0.5).sum()))
    # AUC via right-Riemann
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    auc = float(np.sum(precision * (recall - recall_prev)))
    return auc


def precision_recall_fpr(scores, labels, threshold: float = 0.5) -> dict:
    s = _to_np(scores)
    y = _to_np(labels)
    pred = (s >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y > 0.5)).sum())
    fp = int(((pred == 1) & (y <= 0.5)).sum())
    fn = int(((pred == 0) & (y > 0.5)).sum())
    tn = int(((pred == 0) & (y <= 0.5)).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    return {"precision": float(p), "recall": float(r), "fpr": float(fpr),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def pd_at_far(scores_pos, scores_neg, target_far: float, n_normalization: int) -> dict:
    """Probability of detection at a target false alarm rate.

    ``scores_pos`` and ``scores_neg`` are the model's per-sequence (or
    per-tracklet) scores on the positive and negative pool respectively.
    ``n_normalization`` is the number of "trials" the FAR is normalized to —
    e.g. 100 for "1 false alarm per 100 sequences", or len(scores_neg).

    Returns the largest threshold whose false-alarm count is <= ``target_far``
    times that normalization, and the Pd at that threshold.
    """
    s_pos = _to_np(scores_pos)
    s_neg = _to_np(scores_neg)
    if s_pos.size == 0 or s_neg.size == 0:
        return {"threshold": float("nan"), "pd": float("nan"), "n_fa": 0}
    # We're looking for the threshold T s.t. (# neg >= T) / n_norm <= target_far.
    sn_sorted = np.sort(s_neg)[::-1]
    max_fa = int(np.floor(target_far * n_normalization))
    if max_fa >= sn_sorted.size:
        threshold = float(sn_sorted[-1] - 1e-9)
        n_fa = int(sn_sorted.size)
    elif max_fa <= 0:
        threshold = float(sn_sorted[0] + 1e-9)
        n_fa = 0
    else:
        threshold = float(sn_sorted[max_fa])
        n_fa = max_fa
    pd = float((s_pos >= threshold).mean())
    return {"threshold": threshold, "pd": pd, "n_fa": n_fa}


__all__ = ["roc_auc", "pr_auc", "precision_recall_fpr", "pd_at_far"]
