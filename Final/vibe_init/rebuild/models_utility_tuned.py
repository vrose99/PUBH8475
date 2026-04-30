"""
Utility-optimized model wrapper for sepsis risk prediction.

Extends base models with:
- Optimal threshold selection (maximize PhysioNet utility, not accuracy)
- Class weight adjustment for the PhysioNet cost structure
- Utility-focused hyperparameter tuning
"""

import logging
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from models import LogisticGLM, XGBoostModel, GRUModel
from utility import find_optimal_threshold, evaluate_utility_at_threshold

logger = logging.getLogger(__name__)


class UtilityTunedModel:
    """
    Wrapper around any base model that:
    1. Fits the model on training data
    2. Finds the optimal decision threshold for utility
    3. Uses that threshold for predictions

    The key insight: PhysioNet utility is non-convex in threshold space.
    The default 0.5 threshold is often suboptimal.
    """

    def __init__(self, base_model, threshold_search_range=(0.01, 0.99, 100)):
        """
        Args:
            base_model: One of LogisticGLM, XGBoostModel, or GRUModel
            threshold_search_range: (min_threshold, max_threshold, n_points)
        """
        self.base_model = base_model
        self.threshold_search_range = threshold_search_range
        self.optimal_threshold = 0.5  # Will be set during tuning
        self._is_fit = False

    def fit(self, X, y, patient_ids: Optional[np.ndarray] = None):
        """
        Fit base model and find optimal threshold.

        Args:
            X: Training features
            y: Training labels
            patient_ids: Optional patient IDs for per-patient utility calculation
        """
        # Fit the base model
        self.base_model.fit(X, y)

        # Find optimal threshold using validation data (in practice, use CV or separate val set)
        # For now, we'll find threshold that maximizes training utility
        y_proba = self.base_model.predict_proba(X)[:, 1]

        min_t, max_t, n_points = self.threshold_search_range
        thresholds = np.linspace(min_t, max_t, n_points)

        logger.info(f"Searching {n_points} thresholds for optimal utility...")

        self.optimal_threshold, max_utility = find_optimal_threshold(
            y_proba, y, patient_ids=patient_ids, thresholds=thresholds
        )

        logger.info(
            f"Optimal threshold found: {self.optimal_threshold:.4f} "
            f"(utility: {max_utility:.4f})"
        )

        self._is_fit = True
        return self

    def predict_proba(self, X):
        """Return probabilities from base model."""
        if not self._is_fit:
            raise ValueError("Model not fit. Call fit() first.")
        return self.base_model.predict_proba(X)

    def predict(self, X):
        """Predict using optimal threshold."""
        if not self._is_fit:
            raise ValueError("Model not fit. Call fit() first.")
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.optimal_threshold).astype(int)


class LogisticGLMUtilityTuned(UtilityTunedModel):
    """LogisticGLM with utility-optimized threshold tuning."""

    def __init__(self, C: float = 0.1, random_state: int = 42, **kwargs):
        """
        Args:
            C: Regularization strength (default 0.1 is less regularized than original 0.01)
                Higher C → less regularization → can detect more positives
            random_state: Random seed
        """
        base_model = LogisticGLM(C=C, random_state=random_state)
        super().__init__(base_model, **kwargs)


class XGBoostUtilityTuned(UtilityTunedModel):
    """XGBoost with utility-optimized threshold tuning and hyperparameters."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        scale_pos_weight: float = 1.5,
        random_state: int = 42,
        **kwargs
    ):
        """
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Max tree depth (shallow trees for early stopping)
            learning_rate: Higher learning rate to adapt to cost structure
            scale_pos_weight: Weight positive class more (default 1.0 = no weight)
                Use > 1.0 to encourage more positive predictions
            random_state: Random seed
        """
        # Create base XGBoost model with utility-aware hyperparameters
        base_model = XGBoostModel(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )

        # Override scale_pos_weight in the internal XGBoost model
        base_model.model.set_params(scale_pos_weight=scale_pos_weight)

        super().__init__(base_model, **kwargs)


class GRUUtilityTuned(UtilityTunedModel):
    """GRU with utility-optimized threshold tuning."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        random_state: int = 42,
        **kwargs
    ):
        """
        Args:
            hidden_size: GRU hidden dimension
            num_layers: Number of GRU layers
            dropout: Dropout rate (higher → more regularization)
            epochs: Training epochs (more epochs to allow convergence)
            batch_size: Batch size
            learning_rate: Adam learning rate (higher for faster adaptation)
            random_state: Random seed
        """
        base_model = GRUModel(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        super().__init__(base_model, **kwargs)


def get_utility_tuned_model(model_name: str, **kwargs):
    """
    Factory function to get a utility-tuned model.

    Args:
        model_name: "glm", "xgboost", or "gru"
        **kwargs: Additional arguments passed to model constructor

    Returns:
        UtilityTuned wrapper model
    """
    if model_name == "glm":
        return LogisticGLMUtilityTuned(**kwargs)
    elif model_name == "xgboost":
        return XGBoostUtilityTuned(**kwargs)
    elif model_name == "gru":
        return GRUUtilityTuned(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
