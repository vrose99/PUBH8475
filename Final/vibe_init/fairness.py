"""
Shared metric helpers used by fairness_timeseries.py.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def _safe_roc_auc(y_true, y_prob) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if len(np.unique(y_true)) < 2:
                return np.nan
            return roc_auc_score(y_true, y_prob)
        except Exception:
            return np.nan


def _fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return fp / (fp + tn) if (fp + tn) > 0 else np.nan


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    if len(y_true) == 0:
        return np.nan
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if in_bin.sum() == 0:
            continue
        conf = y_prob[in_bin].mean()
        acc  = y_true[in_bin].mean()
        ece += (in_bin.sum() / len(y_true)) * abs(conf - acc)
    return float(ece)


def clinical_utility_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    w_tp: float = 1.0,
    w_fp: float = -0.05,
    w_fn: float = -2.0,
    w_tn: float = 0.0,
) -> float:
    if len(y_true) == 0:
        return np.nan
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return (w_tp * tp + w_fp * fp + w_fn * fn + w_tn * tn) / len(y_true)


def _safe_brier(y_true, y_prob) -> float:
    try:
        if len(y_true) == 0:
            return np.nan
        return brier_score_loss(y_true, y_prob)
    except Exception:
        return np.nan
