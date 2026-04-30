"""
Training and evaluation utilities for the bootstrap pipeline.

Handles:
- Extracting features and labels from DataFrames
- Training models on training data
- Evaluating models on bootstrap samples
- Collecting and aggregating metrics
"""

import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from utility import physionet_utility, evaluate_utility_per_group

logger = logging.getLogger(__name__)

# Metadata columns that should not be used as features
_META_COLS = {
    "patient_id", "hour", "SepsisLabel", "target",
    "hours_until_sepsis", "is_censored",
    "Gender", "Unit1", "Unit2", "Age", "HospAdmTime",
}


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Extract feature column names (exclude metadata)."""
    return [c for c in df.columns if c not in _META_COLS]


def extract_Xy(
    df: pd.DataFrame,
    label_column: str = "target",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features X and binary labels y from a DataFrame.

    Args:
        df: DataFrame with features and label column
        label_column: Name of target column

    Returns:
        (X, y) where X is (n_samples, n_features) and y is (n_samples,)
    """
    if label_column not in df.columns:
        # Fallback to SepsisLabel if target not present
        if "SepsisLabel" in df.columns:
            label_column = "SepsisLabel"
        else:
            raise ValueError(
                f"Neither '{label_column}' nor 'SepsisLabel' found in DataFrame"
            )

    feat_cols = get_feature_columns(df)
    X = df[feat_cols].values.astype(float)
    y = df[label_column].values.astype(int)

    return X, y


def train_model(
    model,
    train_df: pd.DataFrame,
    label_column: str = "target",
) -> Tuple:
    """
    Train a model on training data.

    Args:
        model: Unfitted model object with fit() method
        train_df: Training DataFrame (patient-hours with features and label)
        label_column: Name of target column

    Returns:
        (fitted_model, feature_columns)
    """
    X_train, y_train = extract_Xy(train_df, label_column=label_column)
    feat_cols = get_feature_columns(train_df)

    logger.info(
        "Training on %d samples, %d features, %d positive class",
        len(X_train),
        X_train.shape[1],
        (y_train == 1).sum(),
    )

    model.fit(X_train, y_train)

    return model, feat_cols


def evaluate_model(
    model,
    eval_df: pd.DataFrame,
    label_column: str = "SepsisLabel",
    patient_id_column: Optional[str] = "patient_id",
) -> Dict[str, float]:
    """
    Evaluate a fitted model on a test/evaluation set.

    Computes:
    - AUROC, Accuracy, Precision, Recall, F1 (at threshold 0.5)
    - PhysioNet 2019 utility score (uses label_column for computation)

    Args:
        model: Fitted model with predict_proba() and predict() methods
        eval_df: Evaluation DataFrame
        label_column: Name of target column
        hours_until_sepsis_column: Column with hours until sepsis (optional)
        patient_id_column: Column with patient IDs (optional)

    Returns:
        Dictionary of metric names to values
    """
    X_eval, y_eval = extract_Xy(eval_df, label_column=label_column)

    # Probabilities and predictions
    y_proba = model.predict_proba(X_eval)[:, 1]
    y_pred = model.predict(X_eval)

    # Compute metrics
    metrics = {
        "n_samples": len(X_eval),
        "n_positive": int((y_eval == 1).sum()),
        "prevalence": float((y_eval == 1).mean()),
    }

    if len(np.unique(y_eval)) > 1:
        # Only compute AUROC if both classes present
        metrics["auroc"] = float(roc_auc_score(y_eval, y_proba))
    else:
        metrics["auroc"] = np.nan

    # Threshold-based metrics
    metrics["accuracy"] = float(accuracy_score(y_eval, y_pred))
    metrics["precision"] = float(precision_score(y_eval, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_eval, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_eval, y_pred, zero_division=0))

    # PhysioNet utility (official implementation uses SepsisLabel, not hours_until_sepsis)
    if label_column and label_column in eval_df.columns:
        labels = eval_df[label_column].values
        pids = (
            eval_df[patient_id_column].values
            if patient_id_column and patient_id_column in eval_df.columns
            else None
        )
        try:
            utility = physionet_utility(labels, y_pred, patient_ids=pids)
            metrics["utility"] = float(utility)
        except Exception as e:
            logger.debug(f"Could not compute utility: {e}")
            metrics["utility"] = np.nan
    else:
        # label_column not available
        metrics["utility"] = np.nan

    return metrics


def evaluate_model_per_group(
    model,
    eval_df: pd.DataFrame,
    group_column: str = "Gender",
    label_column: str = "SepsisLabel",
    patient_id_column: Optional[str] = "patient_id",
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model separately for each group (e.g., by Gender).

    Args:
        model: Fitted model
        eval_df: Evaluation DataFrame
        group_column: Column defining groups
        label_column: Target column (SepsisLabel)
        patient_id_column: Column with patient IDs (optional)

    Returns:
        Nested dict: {group_value: {metric: value}}
    """
    results = {}

    for group_val in sorted(eval_df[group_column].unique()):
        group_df = eval_df[eval_df[group_column] == group_val]
        group_metrics = evaluate_model(
            model,
            group_df,
            label_column=label_column,
            patient_id_column=patient_id_column,
        )
        results[group_val] = group_metrics

    return results


class BootstrapEvaluator:
    """
    Evaluates a model across multiple bootstrap samples.

    Stores per-iteration metrics and can aggregate them.
    Supports standard metrics (AUROC, recall, etc.) and PhysioNet utility.
    """

    def __init__(
        self,
        model,
        train_df: pd.DataFrame,
        label_column: str = "SepsisLabel",
        patient_id_column: Optional[str] = "patient_id",
    ):
        """
        Args:
            model: Unfitted model
            train_df: Training data (will be used to fit model once)
            label_column: Target column name (SepsisLabel)
            patient_id_column: Column with patient IDs (for utility aggregation)
        """
        self.model = model
        self.train_df = train_df
        self.label_column = label_column
        self.patient_id_column = patient_id_column

        # Fit the model once on training data
        self.model, self.feat_cols = train_model(
            self.model, self.train_df, label_column=label_column
        )

        self.bootstrap_metrics = []
        self.bootstrap_per_group = []

    def evaluate_iteration(
        self,
        bootstrap_df: pd.DataFrame,
        iteration_idx: int,
        compute_per_group: bool = False,
        group_column: str = "Gender",
    ) -> Dict:
        """
        Evaluate model on one bootstrap sample.

        Args:
            bootstrap_df: Bootstrap sample DataFrame
            iteration_idx: Iteration number (for logging)
            compute_per_group: If True, also compute per-group metrics
            group_column: Column for grouping (if compute_per_group=True)

        Returns:
            Dictionary of metrics for this iteration
        """
        metrics = evaluate_model(
            self.model,
            bootstrap_df,
            label_column=self.label_column,
            patient_id_column=self.patient_id_column,
        )
        metrics["iteration"] = iteration_idx

        # Per-group metrics (optional)
        if compute_per_group:
            per_group = evaluate_model_per_group(
                self.model,
                bootstrap_df,
                group_column=group_column,
                label_column=self.label_column,
                patient_id_column=self.patient_id_column,
            )
            metrics["per_group"] = per_group

        self.bootstrap_metrics.append(metrics)

        logger.debug(
            "Iteration %d: AUROC=%.3f, Accuracy=%.3f, Recall=%.3f, Utility=%.3f",
            iteration_idx,
            metrics.get("auroc", np.nan),
            metrics.get("accuracy", np.nan),
            metrics.get("recall", np.nan),
            metrics.get("utility", np.nan),
        )

        return metrics

    def aggregate_bootstrap_metrics(self) -> Dict[str, Dict]:
        """
        Aggregate bootstrap results across all iterations.

        Returns dict with keys like:
          auroc: {mean, std, median}
          accuracy: {mean, std, median}
          utility: {mean, std, median}
          ... for each metric
        """
        if not self.bootstrap_metrics:
            raise ValueError("No bootstrap metrics collected. Run evaluate_iteration() first.")

        aggregated = {}

        # List of numeric metrics to aggregate
        numeric_metrics = ["auroc", "accuracy", "precision", "recall", "f1", "utility"]

        for metric_name in numeric_metrics:
            values = [
                m[metric_name]
                for m in self.bootstrap_metrics
                if metric_name in m and not np.isnan(m[metric_name])
            ]

            if values:
                aggregated[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n_iterations": len(values),
                }

        return aggregated

    def summary(self) -> pd.DataFrame:
        """
        Return bootstrap metrics as a DataFrame (one row per iteration).
        """
        return pd.DataFrame(self.bootstrap_metrics)
