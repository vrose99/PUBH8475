"""
PhysioNet 2019 Challenge utility function for sepsis prediction.

This implementation matches the official PhysioNet 2019 evaluate_sepsis_score.py.

Key parameters (from PhysioNet official code)
----------------------------------------------
dt_early   = -12  : earliest beneficial prediction time (12 hours before onset)
dt_optimal = -6   : optimal prediction time (6 hours before onset)
dt_late    = 3    : latest beneficial prediction time (3 hours after onset)

Utility function is piecewise linear:

  Septic patient, prediction = 1 (True Positive):
    - If -12 ≤ (t - t_sepsis) ≤ -6:  linear ramp from 0 to 1
    - If -6 < (t - t_sepsis) ≤ 3:     linear decay from 1 to 0
    - If (t - t_sepsis) > 3:          -2 (too late)

  Septic patient, prediction = 0 (False Negative):
    - If (t - t_sepsis) ≤ -6:  u = 0 (haven't reached optimal yet)
    - If -6 < (t - t_sepsis) ≤ 3:  linear decay from 0 to -2
    - If (t - t_sepsis) > 3:   u = -2 (missed sepsis)

  Non-septic patient, prediction = 1 (False Positive):
    u = -0.05 (alarm fatigue)

  Non-septic patient, prediction = 0 (True Negative):
    u = 0 (correct)

Normalization
-------------
Per-patient score = (observed - inaction) / (best - inaction)
Overall score = mean of per-patient scores

where:
  - observed: utility of actual predictions
  - inaction: utility if all predictions are 0
  - best: utility of oracle (perfect early detection)

Reference
---------
PhysioNet 2019 Challenge official code:
https://github.com/physionet/python-challenge/blob/master/evaluate_sepsis_score.py
"""

import logging
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# PhysioNet 2019 Challenge parameters
_DT_EARLY   = -12.0
_DT_OPTIMAL = -6.0
_DT_LATE    = 3.0
_MAX_U_TP   = 1.0
_MIN_U_FN   = -2.0
_U_FP       = -0.05
_U_TN       = 0.0


def _compute_prediction_utility(
    labels: np.ndarray,
    predictions: np.ndarray,
    dt_early: float = _DT_EARLY,
    dt_optimal: float = _DT_OPTIMAL,
    dt_late: float = _DT_LATE,
    max_u_tp: float = _MAX_U_TP,
    min_u_fn: float = _MIN_U_FN,
    u_fp: float = _U_FP,
    u_tn: float = _U_TN,
) -> float:
    """
    Compute total utility for a single patient.

    Parameters
    ----------
    labels : (n,) int array
        Binary sepsis labels (0 or 1) for each time step
    predictions : (n,) int array
        Binary sepsis predictions (0 or 1) for each time step
    dt_early, dt_optimal, dt_late : float
        Time windows (hours relative to sepsis onset)
    max_u_tp, min_u_fn, u_fp, u_tn : float
        Utility values for TP, FN, FP, TN

    Returns
    -------
    float : total utility for this patient
    """
    # Determine if patient is septic and when
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal  # adjust label position to true onset
    else:
        is_septic = False
        t_sepsis = float('inf')

    n = len(labels)

    # Compute slopes and intercepts for piecewise linear utility function
    # Segment 1: dt_early to dt_optimal (e.g., -12 to -6 hours)
    m_1 = max_u_tp / (dt_optimal - dt_early)
    b_1 = -m_1 * dt_early

    # Segment 2: dt_optimal to dt_late (e.g., -6 to +3 hours)
    m_2 = -max_u_tp / (dt_late - dt_optimal)
    b_2 = -m_2 * dt_late

    # Segment 3: for FN after dt_optimal (penalty phase)
    m_3 = min_u_fn / (dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    # Compute utility for each time step
    u = np.zeros(n)

    for t in range(n):
        if t <= t_sepsis + dt_late:
            # TP: septic patient + positive prediction
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    # Early phase: linear ramp from 0 to max
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                else:
                    # Decay phase: linear decrease
                    u[t] = m_2 * (t - t_sepsis) + b_2

            # FP: non-septic patient + positive prediction
            elif not is_septic and predictions[t]:
                u[t] = u_fp

            # FN: septic patient + negative prediction
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                else:
                    # Penalty phase: decay from 0 to min
                    u[t] = m_3 * (t - t_sepsis) + b_3

            # TN: non-septic patient + negative prediction
            elif not is_septic and not predictions[t]:
                u[t] = u_tn

    return float(np.sum(u))


def physionet_utility(
    labels: np.ndarray,
    y_pred_binary: np.ndarray,
    patient_ids: Optional[np.ndarray] = None,
) -> float:
    """
    Compute normalised PhysioNet 2019 utility score.

    Parameters
    ----------
    labels : (n,) int array
        Binary sepsis labels (0 or 1) — SepsisLabel column from PSV data
    y_pred_binary : (n,) int array
        Binary alarm predictions (0 or 1)
    patient_ids : (n,) optional array
        If provided, normalisation is done per patient then averaged.
        Preferred because it prevents long-stay patients from dominating.
        If None, all rows are treated as a single block.

    Returns
    -------
    float : normalised utility score (typically in [-1, 1] range)
    """
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(y_pred_binary, dtype=int)

    if patient_ids is None:
        # Single-patient (or all-rows) computation
        obs = _compute_prediction_utility(labels, predictions)
        inaction = _compute_prediction_utility(
            labels, np.zeros_like(predictions)
        )
        best = _compute_prediction_utility(
            labels, _oracle_predictions(labels)
        )
        denom = best - inaction
        return 0.0 if denom == 0.0 else float((obs - inaction) / denom)

    # Per-patient normalisation
    pids = np.asarray(patient_ids)
    patient_scores = []

    for pid in np.unique(pids):
        mask = pids == pid
        labels_p = labels[mask]
        pred_p = predictions[mask]

        obs_p = _compute_prediction_utility(labels_p, pred_p)
        inaction_p = _compute_prediction_utility(
            labels_p, np.zeros_like(pred_p)
        )
        best_p = _compute_prediction_utility(
            labels_p, _oracle_predictions(labels_p)
        )

        denom_p = best_p - inaction_p
        if denom_p == 0.0:
            continue  # non-septic patient contributes nothing

        patient_scores.append((obs_p - inaction_p) / denom_p)

    if not patient_scores:
        return 0.0

    return float(np.mean(patient_scores))


def _oracle_predictions(labels: np.ndarray) -> np.ndarray:
    """
    Compute the oracle (optimal) predictions for a patient.

    The oracle alarms in the window [t_sepsis + dt_early, t_sepsis + dt_late]
    and never alarms for non-septic patients.
    """
    predictions = np.zeros_like(labels)

    if np.any(labels):
        t_sepsis = np.argmax(labels) - _DT_OPTIMAL
        start_idx = max(0, int(t_sepsis + _DT_EARLY))
        end_idx = min(len(labels), int(t_sepsis + _DT_LATE + 1))
        predictions[start_idx:end_idx] = 1

    return predictions


def evaluate_utility_at_threshold(
    y_proba: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    patient_ids: Optional[np.ndarray] = None,
) -> float:
    """
    Compute utility when y_proba >= threshold triggers alarm.

    Parameters
    ----------
    y_proba : (n,) array
        Predicted probability of sepsis
    labels : (n,) int array
        Binary sepsis labels
    threshold : float
        Decision threshold
    patient_ids : (n,) optional array
        Patient IDs for per-patient normalisation

    Returns
    -------
    float : normalised utility score
    """
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return physionet_utility(labels, y_pred, patient_ids=patient_ids)


def find_optimal_threshold(
    y_proba: np.ndarray,
    labels: np.ndarray,
    patient_ids: Optional[np.ndarray] = None,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Search for the decision threshold that maximises utility.

    Returns
    -------
    (optimal_threshold, max_utility)
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.5
    best_utility = float("-inf")

    for thresh in thresholds:
        u = evaluate_utility_at_threshold(
            y_proba, labels, thresh, patient_ids=patient_ids
        )
        if u > best_utility:
            best_utility = u
            best_threshold = thresh

    return best_threshold, best_utility


def utility_curve(
    y_proba: np.ndarray,
    labels: np.ndarray,
    patient_ids: Optional[np.ndarray] = None,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute utility score at every threshold. Returns a tidy DataFrame."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    for thresh in thresholds:
        u = evaluate_utility_at_threshold(
            y_proba, labels, thresh, patient_ids=patient_ids
        )
        rows.append({"threshold": float(thresh), "utility": float(u)})

    return pd.DataFrame(rows)


def evaluate_utility_per_group(
    y_proba: np.ndarray,
    labels: np.ndarray,
    group_column: np.ndarray,
    patient_ids: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict:
    """
    Compute normalised utility separately for each group value.

    Returns
    -------
    dict : {group_value: utility_score}
    """
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    group_column = np.asarray(group_column)

    results = {}
    for gval in np.unique(group_column):
        mask = group_column == gval
        pids = patient_ids[mask] if patient_ids is not None else None
        score = physionet_utility(labels[mask], y_pred[mask], patient_ids=pids)
        results[gval] = float(score)

    return results


class UtilityEvaluator:
    """Convenience wrapper: holds predictions + metadata, exposes evaluation API."""

    def __init__(
        self,
        y_proba: np.ndarray,
        labels: np.ndarray,
        patient_ids: Optional[np.ndarray] = None,
        group_column: Optional[np.ndarray] = None,
    ):
        self.y_proba = np.asarray(y_proba, dtype=float)
        self.labels = np.asarray(labels, dtype=int)
        self.patient_ids = patient_ids
        self.group_column = group_column

    def at_threshold(self, threshold: float = 0.5) -> float:
        return evaluate_utility_at_threshold(
            self.y_proba, self.labels, threshold, self.patient_ids
        )

    def find_optimal_threshold(self, thresholds: Optional[np.ndarray] = None):
        return find_optimal_threshold(
            self.y_proba, self.labels, self.patient_ids, thresholds
        )

    def per_group(self, threshold: float = 0.5) -> Dict:
        if self.group_column is None:
            raise ValueError("group_column not provided at init")
        return evaluate_utility_per_group(
            self.y_proba, self.labels, self.group_column,
            self.patient_ids, threshold
        )

    def curve(self, thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
        return utility_curve(
            self.y_proba, self.labels, self.patient_ids, thresholds
        )

    def summary(self) -> Dict:
        u_half = self.at_threshold(0.5)
        opt_thresh, u_max = self.find_optimal_threshold()
        return {
            "utility_at_0.5": u_half,
            "optimal_threshold": opt_thresh,
            "max_utility": u_max,
        }
