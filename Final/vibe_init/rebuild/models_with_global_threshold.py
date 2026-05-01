"""
Model wrappers that use a GLOBAL optimal threshold (not per-model tuning).

This avoids overfitting by:
1. Training models on training data
2. Finding optimal threshold on VALIDATION data
3. Applying same threshold to all models on TEST data

Key difference from models_utility_tuned.py:
- No automatic tuning during fit()
- Threshold must be set explicitly via set_threshold()
- All models share the same threshold for fair comparison
"""

import logging
from typing import Optional
import numpy as np
from models import LogisticGLM, XGBoostModel, GRUModel
from utility import find_optimal_threshold

logger = logging.getLogger(__name__)


class GlobalThresholdModel:
    """
    Wrapper that applies a global decision threshold (found on validation data).

    Workflow:
    1. model.fit(X_train, y_train) — train on training data
    2. threshold = find_global_threshold(model1, model2, model3, X_val, y_val, ...)
    3. model.set_threshold(threshold) — use same threshold for all models
    4. predictions = model.predict(X_test) — predict using that threshold
    """

    def __init__(self, base_model, threshold=0.5):
        """
        Args:
            base_model: One of LogisticGLM, XGBoostModel, or GRUModel
            threshold: Decision threshold (default 0.5)
        """
        self.base_model = base_model
        self.threshold = threshold
        self._is_fit = False

    def fit(self, X, y):
        """Fit base model on training data."""
        self.base_model.fit(X, y)
        self._is_fit = True
        logger.info(
            f"Fitted {self.base_model.__class__.__name__} "
            f"(will use threshold={self.threshold:.4f})"
        )
        return self

    def set_threshold(self, threshold: float):
        """Set the decision threshold."""
        self.threshold = threshold
        logger.info(f"Threshold set to {threshold:.4f}")
        return self

    def predict_proba(self, X):
        """Return probabilities from base model."""
        if not self._is_fit:
            raise ValueError("Model not fit. Call fit() first.")
        return self.base_model.predict_proba(X)

    def predict(self, X):
        """Predict using the set threshold."""
        if not self._is_fit:
            raise ValueError("Model not fit. Call fit() first.")
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)


def find_global_threshold(
    models: list,
    X_val: np.ndarray,
    y_val: np.ndarray,
    patient_ids_val: Optional[np.ndarray] = None,
    thresholds: Optional[np.ndarray] = None,
) -> tuple:
    """
    Find a single optimal threshold that maximizes MEAN utility across all models.

    Args:
        models: List of fitted models (all must be fit)
        X_val: Validation features
        y_val: Validation labels
        patient_ids_val: Optional patient IDs for per-patient utility
        thresholds: Array of thresholds to search (default: 0.01 to 0.99 in 100 steps)

    Returns:
        (optimal_threshold, utilities_per_model, mean_utility)
        where utilities_per_model is a dict {model_name: utility}
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)

    best_threshold = 0.5
    best_mean_utility = float("-inf")
    best_utilities = {}

    logger.info(f"Searching {len(thresholds)} thresholds for global optimum...")
    logger.info(f"Evaluating {len(models)} models on {len(X_val)} validation samples...")

    for thresh in thresholds:
        utilities = {}
        mean_utility = 0

        for model in models:
            # Get probabilities
            y_proba = model.predict_proba(X_val)[:, 1]

            # Convert to predictions at this threshold
            y_pred = (y_proba >= thresh).astype(int)

            # Compute utility
            from utility import physionet_utility

            utility = physionet_utility(y_val, y_pred, patient_ids=patient_ids_val)
            model_name = model.base_model.__class__.__name__
            utilities[model_name] = utility

        mean_utility = np.mean(list(utilities.values()))

        if mean_utility > best_mean_utility:
            best_mean_utility = mean_utility
            best_threshold = thresh
            best_utilities = utilities

    logger.info(f"\nOptimal threshold found: {best_threshold:.4f}")
    logger.info(f"Mean utility across models: {best_mean_utility:.4f}")
    for model_name, utility in best_utilities.items():
        logger.info(f"  {model_name}: {utility:.4f}")

    return best_threshold, best_utilities, best_mean_utility


def get_global_threshold_model(model_name: str, threshold=0.5, **kwargs):
    """
    Factory function to get a global-threshold model.

    Args:
        model_name: "glm", "xgboost", or "gru"
        threshold: Decision threshold (default 0.5)
        **kwargs: Additional arguments passed to model constructor

    Returns:
        GlobalThresholdModel wrapper
    """
    if model_name == "glm":
        base = LogisticGLM(**kwargs)
    elif model_name == "xgboost":
        base = XGBoostModel(**kwargs)
    elif model_name == "gru":
        base = GRUModel(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return GlobalThresholdModel(base, threshold=threshold)
