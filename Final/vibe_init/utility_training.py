"""
Utility-aware threshold search for PhysioNet 2019 scoring.

Instead of using a fixed 0.5 decision threshold, grid-search for the
threshold that maximises the normalised PhysioNet utility on the
(training) data provided.

Entry points
------------
find_utility_threshold(y_prob, hours_until_sepsis, patient_ids)
    Returns (best_threshold, best_utility).
"""

import logging
from typing import Optional, Tuple

import numpy as np

from fairness_timeseries import physionet_2019_utility

logger = logging.getLogger(__name__)


def find_utility_threshold(
    y_prob: np.ndarray,
    hours_until_sepsis: np.ndarray,
    patient_ids: Optional[np.ndarray] = None,
    step: float = 0.05,
) -> Tuple[float, float]:
    """
    Grid-search the decision threshold that maximises PhysioNet 2019
    normalised utility on the supplied data (typically the training set).

    Parameters
    ----------
    y_prob             : predicted probabilities (n_rows,)
    hours_until_sepsis : hours until first SepsisLabel=1; negative for
                         post-onset rows; NaN for non-septic patients
    patient_ids        : optional patient ID array for patient-level
                         normalisation (recommended)
    step               : threshold grid step (default 0.05)

    Returns
    -------
    (best_threshold, best_utility)
        best_threshold : float in (0, 1) that maximises utility
        best_utility   : the utility score at that threshold
    """
    y_prob = np.asarray(y_prob, dtype=float)
    hours_until_sepsis = np.asarray(hours_until_sepsis, dtype=float)

    best_threshold = 0.5
    best_utility = float("-inf")

    thresholds = np.arange(step, 1.0, step)
    utilities = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        u = physionet_2019_utility(
            hours_until_sepsis, y_pred, patient_ids=patient_ids
        )
        utilities.append(float(u))
        if u > best_utility:
            best_utility = u
            best_threshold = float(t)

    logger.info(
        "Threshold search: best=%.2f (utility=%.4f) | range=[%.4f, %.4f]",
        best_threshold, best_utility, min(utilities), max(utilities),
    )
    return best_threshold, best_utility
