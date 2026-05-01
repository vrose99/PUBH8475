"""
Fairness metrics aligned with main pipeline.py evaluation.

Computes three fairness metrics per gender group:
  1. overall_physionet_utility — PhysioNet 2019 utility score (ideal = 1.0)
  2. disparate_impact — P(alarm|female) / P(alarm|male) (ideal = 1.0)
  3. equal_opportunity — TPR(female) − TPR(male) (ideal = 0)

Gender groups: 'F' (female) and 'M' (male)
"""

import numpy as np
from sklearn.metrics import recall_score
from utility import physionet_utility


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    patient_ids: np.ndarray,
    hours_until_sepsis: np.ndarray = None,
    threshold: float = 0.4,
) -> dict:
    """
    Compute fairness metrics per gender group.

    Parameters
    ----------
    y_true : (n,) binary array
        Binary sepsis labels
    y_prob : (n,) float array
        Predicted probabilities [0, 1]
    y_pred : (n,) binary array
        Binary predictions at given threshold
    sensitive : (n,) array
        Gender values; 'F' or 'M'
    patient_ids : (n,) array
        Patient IDs for utility computation
    hours_until_sepsis : (n,) float array, optional
        Hours until sepsis onset (for PhysioNet utility)
    threshold : float
        Decision threshold for predictions

    Returns
    -------
    dict with keys:
        - 'overall_physionet_utility': float
        - 'disparate_impact': float (PPR_F / PPR_M)
        - 'equal_opportunity': float (TPR_F - TPR_M)
        - 'female_tpr': float
        - 'male_tpr': float
        - 'female_ppr': float
        - 'male_ppr': float
    """
    # Separate by gender
    f_mask = (sensitive == 'F') | (sensitive == 1) | (sensitive == 'Female')
    m_mask = (sensitive == 'M') | (sensitive == 0) | (sensitive == 'Male')

    # 1. Overall PhysioNet Utility
    if hours_until_sepsis is not None:
        overall_utility = physionet_utility(y_true, y_pred, patient_ids=patient_ids)
    else:
        overall_utility = np.nan

    # 2. True Positive Rates by gender
    try:
        female_tpr = recall_score(y_true[f_mask], y_pred[f_mask], zero_division=0) if f_mask.sum() > 0 else np.nan
    except:
        female_tpr = np.nan

    try:
        male_tpr = recall_score(y_true[m_mask], y_pred[m_mask], zero_division=0) if m_mask.sum() > 0 else np.nan
    except:
        male_tpr = np.nan

    equal_opp = female_tpr - male_tpr if not np.isnan(female_tpr) and not np.isnan(male_tpr) else np.nan

    # 3. Positive Prediction Rates (alarming rates) by gender
    female_ppr = float(y_pred[f_mask].mean()) if f_mask.sum() > 0 else np.nan
    male_ppr = float(y_pred[m_mask].mean()) if m_mask.sum() > 0 else np.nan

    # Disparate Impact = PPR(F) / PPR(M)
    if not np.isnan(male_ppr) and male_ppr > 0:
        disparate_impact = female_ppr / male_ppr
    else:
        disparate_impact = np.nan

    return {
        'overall_physionet_utility': float(overall_utility),
        'disparate_impact': float(disparate_impact),
        'equal_opportunity': float(equal_opp),
        'female_tpr': float(female_tpr),
        'male_tpr': float(male_tpr),
        'female_ppr': float(female_ppr),
        'male_ppr': float(male_ppr),
    }
