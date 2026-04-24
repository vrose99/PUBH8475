"""
Fairness metric computation.

All public functions accept numpy arrays or pandas Series.
compute_fairness_report() is the primary entry point — it returns a flat dict
of metric_name → value that the evaluation loop collects into a results table.

Metric taxonomy
---------------
Per-group metrics (computed for female, male, and overall):
  n, prevalence, accuracy, auroc, tpr, fpr, precision (= PPV), f1,
  avg_precision (PR-AUC), brier (calibration), ece (expected calibration
  error), clinical_utility (sepsis-weighted score)

Gap metrics (signed = female - male):
  tpr_gap, fpr_gap, auroc_gap, precision_gap, f1_gap, accuracy_gap,
  ppv_gap (= precision_gap), brier_gap, ece_gap, clinical_utility_gap

Aggregate fairness metrics:
  demographic_parity_diff   — P(ŷ=1|female) - P(ŷ=1|male)
  equalized_odds_diff       — max(|TPR gap|, |FPR gap|)
  equal_opportunity_diff    — |TPR gap|
  sufficiency_diff          — |PPV gap|  (also called predictive parity gap)
  worst_group_auroc         — min(female_auroc, male_auroc)
      ^ guards against the "illusion of fairness": a small gap can coexist
        with terrible absolute performance for a minority group if the
        model has essentially given up on everyone.
  illusion_of_fairness      — Boolean flag: small gap (<0.05) but worst-
        group AUROC is much worse than overall (>0.10 drop).
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import Config


# ── Low-level metric helpers ──────────────────────────────────────────────────

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
    """
    Expected Calibration Error — weighted mean gap between predicted
    probability and empirical positive rate across probability bins.

    ECE = Σ_b (|B_b|/N) · |acc(B_b) - conf(B_b)|

    Lower is better.  A large group-level ECE gap means the model is
    systematically over/under-confident for one subgroup.
    """
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
    """
    Weighted-outcome clinical utility score (per-patient analog of the
    PhysioNet 2019 challenge scoring function).

    Default weights reflect the sepsis cost structure:
      +1.0 for a correctly detected septic patient (life-saving)
      -2.0 for a missed septic patient (most harmful outcome)
      -0.05 for a false alarm (alarm fatigue, but not catastrophic)
       0.0 for a correctly ruled-out non-septic patient (baseline)

    Normalised to [−1, ~1] by dividing by N so groups of different sizes
    are comparable.
    """
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


# ── Per-group summary ────────────────────────────────────────────────────────

def per_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    cfg: Optional[Config] = None,
) -> dict[str, float]:
    """Compute the full set of classification + calibration + utility metrics."""
    y_bin = (y_prob >= threshold).astype(int)

    # Clinical utility weights (configurable via cfg.fairness if desired)
    w_tp = getattr(getattr(cfg, "fairness", None), "utility_w_tp",  1.0) if cfg else 1.0
    w_fp = getattr(getattr(cfg, "fairness", None), "utility_w_fp", -0.05) if cfg else -0.05
    w_fn = getattr(getattr(cfg, "fairness", None), "utility_w_fn", -2.0) if cfg else -2.0
    w_tn = getattr(getattr(cfg, "fairness", None), "utility_w_tn",  0.0) if cfg else 0.0

    return {
        "n":                 len(y_true),
        "prevalence":        y_true.mean() if len(y_true) else np.nan,
        "accuracy":          accuracy_score(y_true, y_bin) if len(y_true) else np.nan,
        "auroc":             _safe_roc_auc(y_true, y_prob),
        "tpr":               recall_score(y_true, y_bin, zero_division=0),
        "fpr":               _fpr(y_true, y_bin),
        "precision":         precision_score(y_true, y_bin, zero_division=0),   # PPV
        "f1":                f1_score(y_true, y_bin, zero_division=0),
        "avg_precision":     (average_precision_score(y_true, y_prob)
                              if len(np.unique(y_true)) > 1 else np.nan),
        "brier":             _safe_brier(y_true, y_prob),
        "ece":               expected_calibration_error(y_true, y_prob),
        "clinical_utility":  clinical_utility_score(
                                 y_true, y_bin,
                                 w_tp=w_tp, w_fp=w_fp, w_fn=w_fn, w_tn=w_tn,
                             ),
    }


# ── Main fairness report ─────────────────────────────────────────────────────

def compute_fairness_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitive: np.ndarray,
    cfg: Config,
    label_prefix: str = "",
) -> dict[str, float]:
    """
    Return a flat dict of metric_name → value:
      female_*, male_*, overall_* for each per-group metric
      gap metrics (signed female − male)
      aggregate fairness scores
      worst_group_auroc and illusion_of_fairness guard
    """
    threshold = cfg.fairness.decision_threshold
    y_pred = (y_prob >= threshold).astype(int)

    f_mask = sensitive == cfg.fairness.female_value
    m_mask = sensitive == cfg.fairness.male_value

    overall = per_group_metrics(y_true, y_pred, y_prob, threshold, cfg=cfg)
    female  = per_group_metrics(
        y_true[f_mask], y_pred[f_mask], y_prob[f_mask], threshold, cfg=cfg
    )
    male    = per_group_metrics(
        y_true[m_mask], y_pred[m_mask], y_prob[m_mask], threshold, cfg=cfg
    )

    ppr_female = y_pred[f_mask].mean() if f_mask.any() else np.nan
    ppr_male   = y_pred[m_mask].mean() if m_mask.any() else np.nan

    def gap(metric: str) -> float:
        return female.get(metric, np.nan) - male.get(metric, np.nan)

    prefix = f"{label_prefix}_" if label_prefix else ""
    report: dict[str, float] = {}

    # Per-group
    for group, gname in [(female, "female"), (male, "male"), (overall, "overall")]:
        for k, v in group.items():
            report[f"{prefix}{gname}_{k}"] = v

    # Signed gap metrics (female − male)
    for m in ["tpr", "fpr", "auroc", "precision", "f1", "accuracy",
              "brier", "ece", "clinical_utility"]:
        report[f"{prefix}{m}_gap"] = gap(m)
    report[f"{prefix}ppv_gap"] = gap("precision")   # alias

    # Aggregate fairness metrics
    report[f"{prefix}demographic_parity_diff"]  = ppr_female - ppr_male
    report[f"{prefix}equal_opportunity_diff"]   = abs(gap("tpr"))
    report[f"{prefix}equalized_odds_diff"]      = max(
        abs(gap("tpr")), abs(gap("fpr"))
    )
    report[f"{prefix}sufficiency_diff"]         = abs(gap("precision"))

    # ── "Illusion of fairness" guard ──────────────────────────────────────────
    # A small AUROC gap is meaningless if both groups have collapsed to
    # near-chance predictions. Flag cells where the worst group's AUROC is
    # substantially worse than the overall AUROC despite a tiny gap.
    worst_auroc = min(
        female.get("auroc", np.nan),
        male.get("auroc", np.nan),
    )
    report[f"{prefix}worst_group_auroc"] = worst_auroc
    report[f"{prefix}auroc_drop_from_overall"] = (
        overall.get("auroc", np.nan) - worst_auroc
    )
    report[f"{prefix}illusion_of_fairness"] = int(
        (abs(gap("auroc")) < 0.05)
        and ((overall.get("auroc", 0) - worst_auroc) > 0.10)
    )

    return report


# ── Bootstrap CI wrapper ──────────────────────────────────────────────────────

def bootstrap_fairness_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitive: np.ndarray,
    cfg: Config,
    n_samples: int = 500,
    ci_level: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, float]:
    """
    Bootstrap the fairness report.

    Returns a dict containing the point estimate plus `<metric>_ci_low` and
    `<metric>_ci_high` columns at the requested confidence level.  Bootstrapping
    is done on the test-set predictions only — it does not retrain the model.

    Only numeric metrics are bootstrapped; integer flags (e.g.
    illusion_of_fairness) are kept from the point estimate.
    """
    rng = rng or np.random.default_rng(cfg.random_state)
    n = len(y_true)

    # Point estimate
    point = compute_fairness_report(y_true, y_prob, sensitive, cfg)

    # Bootstrap resamples
    samples: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        rep = compute_fairness_report(
            y_true[idx], y_prob[idx], sensitive[idx], cfg
        )
        for k, v in rep.items():
            samples[k].append(v)

    alpha = (1 - ci_level) / 2
    out = dict(point)
    for k, vals in samples.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            out[f"{k}_ci_low"]  = np.nan
            out[f"{k}_ci_high"] = np.nan
        else:
            out[f"{k}_ci_low"]  = float(np.quantile(arr, alpha))
            out[f"{k}_ci_high"] = float(np.quantile(arr, 1 - alpha))

    return out


# ── Summary helper ────────────────────────────────────────────────────────────

def fairness_summary_table(
    results: pd.DataFrame,
    cfg: Config,
    gap_metrics: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Pivot a raw results DataFrame into a compact summary table.
    """
    if gap_metrics is None:
        gap_metrics = [
            "auroc_gap", "tpr_gap", "fpr_gap", "ppv_gap",
            "demographic_parity_diff", "equalized_odds_diff",
            "sufficiency_diff", "worst_group_auroc",
            "clinical_utility_gap",
        ]

    keep = ["dataset_id", "model", "mitigation"] + [
        c for c in gap_metrics if c in results.columns
    ]
    return results[keep].set_index(["dataset_id", "model", "mitigation"])
