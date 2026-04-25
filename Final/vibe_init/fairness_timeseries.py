"""
Fairness metrics for early-detection / time-to-event prediction.

The central question is no longer "does the model predict sepsis correctly?"
but "does the model warn both groups equally early?"

Metric taxonomy
---------------
Detection timing metrics (per group):
  median_detection_lead_hours
      Median hours *before* sepsis onset at which the model first
      crosses the alarm threshold.  Positive = early warning given,
      0 = detected exactly at onset, negative = detected after onset (miss).

  pct_detected_before_onset
      % of septic patients in this group who received at least one
      warning before sepsis was recorded.

  pct_missed
      % of septic patients who never triggered an alarm at any hour.

  mean_detection_lead_hours  (mean counterpart, more sensitive to outliers)

Fairness gap metrics (female − male unless noted):
  detection_lead_gap_hours
      Median detection lead difference: positive means females warned earlier.

  missed_rate_gap
      Difference in missed-alarm rates: positive means males missed more.

  alarm_fatigue_rate_gap
      Difference in false-alarm rates (non-septic patients who triggered alarm).

Aggregate fairness:
  early_detection_parity
      |detection_lead_gap_hours| — lower is fairer.

  early_detection_equalized_odds
      max(|TPR gap|, |FPR gap|) evaluated at each threshold.
      Equivalent to equalized_odds_diff from static classifier.

  detection_time_auc
      Area under the detection-time ROC (C-index analog for time-to-event).

All public functions accept numpy arrays; pd.Series are also accepted.

Entry points
------------
  compute_detection_fairness_report(y_prob, y_true, sensitive,
                                    hours_until_sepsis, cfg)
      Returns a flat dict identical in structure to fairness.compute_fairness_report()
      so it slots directly into the existing evaluation loop.
"""

import warnings
from typing import Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from config import Config
from fairness import (
    _safe_roc_auc,
    _fpr,
    expected_calibration_error,
    clinical_utility_score,
    _safe_brier,
)


# ── Detection timing helpers ──────────────────────────────────────────────────

def detection_lead_hours(
    hours_until_sepsis: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    patient_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    For each (patient, hour) row, compute how early a detection was triggered
    relative to the actual sepsis onset.

    Definition
    ----------
    lead = −hours_until_sepsis  at the first hour where y_prob ≥ threshold.

    Positive lead → model alarmed BEFORE onset (desired early warning).
    Zero         → model alarmed exactly at onset.
    Negative     → model alarmed AFTER onset (too late for prevention).
    NaN          → either the patient never developed sepsis (censored) OR
                   the model never alarmed for this patient.

    If patient_ids is provided the function returns one value per patient
    (first alarm for each patient).  Without patient_ids it returns one
    value per row (used for aggregate statistics at threshold level).

    Parameters
    ----------
    hours_until_sepsis : array of shape (n_rows,); NaN for censored rows
    y_prob             : predicted positive-class probability (n_rows,)
    threshold          : alarm threshold
    patient_ids        : optional array of patient IDs (n_rows,) for
                         patient-level aggregation

    Returns
    -------
    np.ndarray of lead hours, one per patient (or per row if no patient_ids).
    """
    hours_until_sepsis = np.asarray(hours_until_sepsis, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    alarms = y_prob >= threshold

    if patient_ids is None:
        # Row-level: return lead for every row where an alarm fired and
        # the patient is septic.
        is_septic = np.isfinite(hours_until_sepsis)
        lead = np.full(len(y_prob), np.nan)
        mask = alarms & is_septic
        lead[mask] = -hours_until_sepsis[mask]   # positive = early
        return lead

    # Patient-level: for each patient find the FIRST alarm hour
    patient_ids = np.asarray(patient_ids)
    unique_patients = np.unique(patient_ids)
    leads = []

    for pid in unique_patients:
        pmask = patient_ids == pid
        p_hours = hours_until_sepsis[pmask]
        p_prob  = y_prob[pmask]
        p_alarm = p_prob >= threshold

        # Is this patient septic?
        is_septic = np.any(np.isfinite(p_hours))
        if not is_septic:
            leads.append(np.nan)   # censored
            continue

        # First hour where alarm fires
        alarm_indices = np.where(p_alarm)[0]
        if len(alarm_indices) == 0:
            leads.append(np.nan)   # model never alarmed → missed
        else:
            first_alarm = alarm_indices[0]
            lead = -p_hours[first_alarm]   # positive = alarmed early
            leads.append(float(lead))

    return np.array(leads)


def _group_detection_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    hours_until_sepsis: np.ndarray,
    threshold: float,
    patient_ids: Optional[np.ndarray],
    cfg: Optional["Config"] = None,
) -> dict:
    """
    Compute detection fairness metrics for one subgroup.
    """
    y_bin = (y_prob >= threshold).astype(int)
    is_septic_row = np.isfinite(hours_until_sepsis)

    # Patient-level lead times (one per patient)
    leads = detection_lead_hours(hours_until_sepsis, y_prob, threshold, patient_ids)
    valid_leads = leads[np.isfinite(leads)]

    # For septic patients only: were they detected?
    is_septic_patient = np.array([
        np.any(np.isfinite(hours_until_sepsis[patient_ids == pid]))
        if patient_ids is not None
        else False
        for pid in (np.unique(patient_ids) if patient_ids is not None else [])
    ]) if patient_ids is not None else np.isfinite(hours_until_sepsis)

    n_septic_patients = int(np.isfinite(leads).sum() + np.isnan(leads[leads != leads]).sum()) if patient_ids is not None else int(is_septic_row.sum())

    # Patients with finite lead were detected; NaN septic patients were missed
    if patient_ids is not None:
        n_septic_pids = sum(
            1 for pid in np.unique(patient_ids)
            if np.any(np.isfinite(hours_until_sepsis[patient_ids == pid]))
        )
        n_detected = int(np.isfinite(leads).sum())
        n_missed   = n_septic_pids - n_detected
    else:
        n_septic_pids = int(is_septic_row.sum())
        n_detected    = int((y_bin[is_septic_row] == 1).sum())
        n_missed      = n_septic_pids - n_detected

    missed_rate = n_missed / n_septic_pids if n_septic_pids > 0 else np.nan

    # False alarm rate (non-septic patients / rows that trigger alarm)
    non_septic_mask = ~is_septic_row
    if non_septic_mask.sum() > 0:
        alarm_fatigue_rate = float(y_bin[non_septic_mask].mean())
    else:
        alarm_fatigue_rate = np.nan

    # Classification metrics on rows (for consistency with static pipeline)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tpr = recall_score(y_true, y_bin, zero_division=0)
        fpr_val = _fpr(y_true, y_bin)
        prec = precision_score(y_true, y_bin, zero_division=0)
        f1  = f1_score(y_true, y_bin, zero_division=0)
        auroc = _safe_roc_auc(y_true, y_prob)
        avg_prec = (
            average_precision_score(y_true, y_prob)
            if len(np.unique(y_true)) > 1 else np.nan
        )
        brier = _safe_brier(y_true, y_prob)
        ece   = expected_calibration_error(y_true, y_prob)

    w_tp = getattr(getattr(cfg, "fairness", None), "utility_w_tp",  1.0) if cfg else 1.0
    w_fp = getattr(getattr(cfg, "fairness", None), "utility_w_fp", -0.05) if cfg else -0.05
    w_fn = getattr(getattr(cfg, "fairness", None), "utility_w_fn", -2.0) if cfg else -2.0
    w_tn = getattr(getattr(cfg, "fairness", None), "utility_w_tn",  0.0) if cfg else 0.0
    clin = clinical_utility_score(y_true, y_bin, w_tp=w_tp, w_fp=w_fp, w_fn=w_fn, w_tn=w_tn)

    return {
        # Patient counts
        "n":                            len(y_true),
        "n_septic":                     n_septic_pids,
        "prevalence":                   float(y_true.mean()) if len(y_true) else np.nan,
        # Standard classification (row-level)
        "tpr":                          float(tpr),
        "fpr":                          float(fpr_val),
        "precision":                    float(prec),
        "f1":                           float(f1),
        "auroc":                        float(auroc),
        "avg_precision":                float(avg_prec),
        "brier":                        float(brier),
        "ece":                          float(ece),
        "clinical_utility":             float(clin),
        # Early-detection specific
        "median_detection_lead_hours":  float(np.median(valid_leads)) if len(valid_leads) > 0 else np.nan,
        "mean_detection_lead_hours":    float(np.mean(valid_leads))   if len(valid_leads) > 0 else np.nan,
        "pct_detected_before_onset":    float((valid_leads > 0).mean() * 100) if len(valid_leads) > 0 else np.nan,
        "pct_detected_at_all":          float(n_detected / n_septic_pids * 100) if n_septic_pids > 0 else np.nan,
        "pct_missed":                   float(missed_rate * 100),
        "alarm_fatigue_rate":           float(alarm_fatigue_rate),
    }


# ── Main report ───────────────────────────────────────────────────────────────

def compute_detection_fairness_report(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    sensitive: np.ndarray,
    hours_until_sepsis: np.ndarray,
    cfg: "Config",
    patient_ids: Optional[np.ndarray] = None,
) -> dict:
    """
    Full detection-fairness report.  Drop-in replacement for
    fairness.compute_fairness_report() in the evaluation loop.

    Parameters
    ----------
    y_prob             : predicted probability of early sepsis (n_rows,)
    y_true             : binary early-detection label (n_rows,)
    sensitive          : Gender values (n_rows,)
    hours_until_sepsis : hours until onset; NaN for non-septic (n_rows,)
    cfg                : Config
    patient_ids        : optional patient ID array for patient-level timing
                         metrics.  Pass df['patient_id'].values for full
                         per-patient detection-lead calculation.

    Returns
    -------
    Flat dict: female_*, male_*, overall_* per metric,
               plus gap metrics and aggregate fairness scores.
    """
    threshold = cfg.fairness.decision_threshold

    f_mask = sensitive == cfg.fairness.female_value
    m_mask = sensitive == cfg.fairness.male_value

    pids_f = patient_ids[f_mask] if patient_ids is not None else None
    pids_m = patient_ids[m_mask] if patient_ids is not None else None
    pids_o = patient_ids if patient_ids is not None else None

    overall = _group_detection_metrics(
        y_true, y_prob, hours_until_sepsis, threshold, pids_o, cfg
    )
    female  = _group_detection_metrics(
        y_true[f_mask], y_prob[f_mask], hours_until_sepsis[f_mask],
        threshold, pids_f, cfg
    )
    male    = _group_detection_metrics(
        y_true[m_mask], y_prob[m_mask], hours_until_sepsis[m_mask],
        threshold, pids_m, cfg
    )

    def gap(metric: str) -> float:
        return female.get(metric, np.nan) - male.get(metric, np.nan)

    report: dict = {}

    # Per-group prefix
    for grp, gname in [(female, "female"), (male, "male"), (overall, "overall")]:
        for k, v in grp.items():
            report[f"{gname}_{k}"] = v

    # Gap metrics (female − male; positive = females better off)
    for m in ["tpr", "fpr", "auroc", "precision", "f1", "brier", "ece",
              "clinical_utility", "median_detection_lead_hours",
              "mean_detection_lead_hours", "pct_detected_before_onset",
              "pct_missed", "alarm_fatigue_rate"]:
        report[f"{m}_gap"] = gap(m)

    report["ppv_gap"] = gap("precision")   # alias

    # Aggregate fairness metrics (mirrors static pipeline naming)
    ppr_f = float((y_prob[f_mask] >= threshold).mean()) if f_mask.any() else np.nan
    ppr_m = float((y_prob[m_mask] >= threshold).mean()) if m_mask.any() else np.nan

    report["demographic_parity_diff"]      = ppr_f - ppr_m
    report["equal_opportunity_diff"]       = abs(gap("tpr"))
    report["equalized_odds_diff"]          = max(abs(gap("tpr")), abs(gap("fpr")))
    report["sufficiency_diff"]             = abs(gap("precision"))

    # Early-detection–specific aggregate fairness
    report["early_detection_parity"]       = abs(gap("median_detection_lead_hours"))
    report["missed_rate_gap"]              = gap("pct_missed")    # positive = females missed less
    report["alarm_fatigue_gap"]            = gap("alarm_fatigue_rate")

    # Illusion-of-fairness guard
    worst_auroc = min(female.get("auroc", np.nan), male.get("auroc", np.nan))
    report["worst_group_auroc"]          = worst_auroc
    report["auroc_drop_from_overall"]    = overall.get("auroc", np.nan) - worst_auroc
    report["illusion_of_fairness"]       = int(
        (abs(gap("auroc")) < 0.05) and
        ((overall.get("auroc", 0.0) - worst_auroc) > 0.10)
    )

    return report


# ── Summary helper ────────────────────────────────────────────────────────────

def detection_summary_table(
    results: "pd.DataFrame",
) -> "pd.DataFrame":
    """
    Pivot results to a human-readable summary focused on detection timing.

    Rows: (dataset_id, model, mitigation)
    Columns: key early-detection fairness metrics
    """
    import pandas as pd

    cols_of_interest = [
        "dataset_id", "model", "mitigation",
        "female_median_detection_lead_hours",
        "male_median_detection_lead_hours",
        "median_detection_lead_hours_gap",
        "female_pct_detected_before_onset",
        "male_pct_detected_before_onset",
        "female_pct_missed",
        "male_pct_missed",
        "equalized_odds_diff",
        "early_detection_parity",
    ]
    keep = [c for c in cols_of_interest if c in results.columns]
    return results[keep].round(3)
