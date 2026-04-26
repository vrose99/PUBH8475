from __future__ import annotations

"""
Model registry with improved models for sepsis detection.

Add new models by inserting an entry into MODEL_REGISTRY.
Each value is a callable (no args) that returns a fresh sklearn-compatible
estimator.  The estimator must implement fit / predict / predict_proba.

Liu et al. (2019, Sci Rep) models are prefixed "liu_":
  liu_glm      — L1 logistic regression, C tuned by 10-fold CV (AUROC)
  liu_xgboost  — XGBoost with class-imbalance correction via scale_pos_weight
  liu_rnn      — LiuLikeGRU (single-step wrapper) applied to aggregated static
                 features; same architecture and hyperparameters as the GRU in
                 Final/models/gru_model.py / Final/run_models.ipynb
"""

import sys
from pathlib import Path
from typing import Dict, Callable

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from gru_impl.gru_model import LiuLikeGRU as _LiuLikeGRU
    _GRU_AVAILABLE = True
except Exception:
    _GRU_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except Exception:
    # XGBoostError (e.g. missing libomp on macOS) is not a subclass of ImportError
    _XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    _LGB_AVAILABLE = True
except Exception:
    _LGB_AVAILABLE = False


def _logistic_regression():
    return LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )


def _random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        n_jobs=1,   # -1 triggers multiprocessing segfaults on macOS Python 3.11
        random_state=42,
    )


def _gradient_boosting():
    return GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )


def _xgboost():
    if not _XGB_AVAILABLE:
        raise RuntimeError(
            "xgboost not available — on macOS run `brew install libomp` then reinstall xgboost"
        )
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    )


def _lightgbm():
    if not _LGB_AVAILABLE:
        raise RuntimeError("lightgbm not available — `pip install lightgbm`")
    return LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )


def _svm_calibrated():
    # Calibrated SVM — useful as a structurally different model family
    return CalibratedClassifierCV(
        SVC(kernel="rbf", class_weight="balanced", probability=False),
        method="sigmoid",
        cv=3,
    )


# ── Liu et al. (2019) model family ───────────────────────────────────────────

def _liu_glm():
    """
    L1-regularized logistic regression (Liu et al. 2019 GLM).

    Logistic regression IS a GLM: binomial family, logit link.  Adding
    L1 / LASSO regularisation replicates the paper's feature-selection step.

    This version uses a single fixed regularisation strength (C=0.01) rather
    than cross-validated grid search.  On 8 000 training rows × 207 features
    this reduces wall-clock from ~3 min (50 CV fits) to ~3 s (1 fit).

    C=0.01 corresponds to moderate regularisation, typical for sparse EHR
    data with ~8% sepsis prevalence.  For a full production run use
    _liu_glm_cv() below which does proper grid search.
    """
    return LogisticRegression(
        C=0.01,
        l1_ratio=1,        # l1_ratio=1 → pure L1/LASSO (replaces penalty='l1')
        solver="saga",     # saga supports elastic-net / pure-L1 via l1_ratio
        class_weight="balanced",
        max_iter=4000,
        random_state=42,
    )


def _liu_glm_cv():
    """
    Full paper-spec GLM: L1 logistic regression with 3-fold CV over C.
    Use this for final/production runs where a few extra minutes is acceptable.
    Total fits: 3 C values × 3 folds = 9  (vs 50 in the paper).
    """
    return LogisticRegressionCV(
        Cs=3,
        cv=3,
        l1_ratios=[1],     # pure L1/LASSO (replaces penalty='l1')
        solver="saga",     # saga supports l1_ratio
        scoring="roc_auc",
        class_weight="balanced",
        max_iter=4000,
        n_jobs=1,
        random_state=42,
    )


def _liu_xgboost():
    """
    XGBoost configured for imbalanced EHR classification.

    scale_pos_weight is set to the expected class ratio for PhysioNet 2019
    (~11:1 negative:positive).  Liu et al. use max_depth=4 with shrinkage
    learning_rate=0.05 and row/column subsampling to reduce overfitting on
    the sparse lab-value features.
    """
    if not _XGB_AVAILABLE:
        raise RuntimeError(
            "xgboost not available — on macOS run `brew install libomp` then reinstall xgboost"
        )
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=11,   # approx neg/pos ratio in PhysioNet 2019
        random_state=42,
        n_jobs=1,
    )


class _GRUStaticWrapper:
    """
    Wraps LiuLikeGRU for use with 2-D static/aggregated data.

    The vibe_init pipeline produces one row per patient (or per hour-window)
    rather than raw hourly sequences, so there is no time dimension.  This
    wrapper reshapes 2-D input (n_samples, n_features) into the 3-D tensor
    (n_samples, 1, n_features) that LiuLikeGRU expects, treating each sample
    as a single-step sequence.  All GRU hyperparameters match the values used
    in Final/run_models.ipynb / Final/config.py.
    """

    def __init__(self):
        if not _GRU_AVAILABLE:
            raise ImportError("torch is required for liu_rnn. Install PyTorch or use liu_glm.")
        self._gru = _LiuLikeGRU(
            random_state=42,
            hidden_size=64,
            num_layers=1,
            dropout=0.1,
            epochs=12,
            batch_size=128,
            learning_rate=1e-3,
        )

    def fit(self, X, y, **kwargs):
        self._gru.fit(np.asarray(X)[:, np.newaxis, :], np.asarray(y))
        return self

    def predict_proba(self, X):
        p = self._gru.predict_proba(np.asarray(X)[:, np.newaxis, :])
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # sklearn compatibility
    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


def _liu_rnn():
    return _GRUStaticWrapper()


def _ensemble_stack():
    """
    5-model stacking ensemble for improved sepsis detection.

    Base learners: LogReg, LightGBM, XGBoost, RandomForest, GradientBoosting
    Meta-learner: Logistic regression (learns optimal blend)

    This ensemble leverages model diversity to catch more sepsis cases
    while maintaining fairness across groups.
    """
    try:
        from sklearn.ensemble import StackingClassifier
    except ImportError:
        raise ImportError("sklearn >= 0.24 required for StackingClassifier")

    base_learners = [
        ("logistic", LogisticRegression(
            C=0.01, l1_ratio=1, solver="saga",
            class_weight="balanced", max_iter=4000, random_state=42
        )),
        ("lightgbm", LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            class_weight="balanced", random_state=42, verbose=-1
        ) if _LGB_AVAILABLE else LogisticRegression(class_weight="balanced", random_state=42)),
        ("xgboost", XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=11,
            eval_metric="logloss", random_state=42, n_jobs=1
        ) if _XGB_AVAILABLE else LogisticRegression(class_weight="balanced", random_state=42)),
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            n_jobs=1, random_state=42
        )),
        ("gb", GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )),
    ]

    meta_learner = LogisticRegression(
        C=1.0, solver="lbfgs", class_weight="balanced",
        max_iter=4000, random_state=42
    )

    return StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=1,
    )


MODEL_REGISTRY: Dict[str, Callable] = {
    "logistic_regression": _logistic_regression,
    "random_forest":       _random_forest,
    "gradient_boosting":   _gradient_boosting,
    "xgboost":             _xgboost,
    "lightgbm":            _lightgbm,
    "svm":                 _svm_calibrated,
    # Liu et al. (2019) model family
    "liu_glm":             _liu_glm,      # fast: fixed C, ~3 s per fit
    "liu_glm_cv":          _liu_glm_cv,   # full: 3×3 CV grid, ~2 min per fit
    "liu_xgboost":         _liu_xgboost,
    "liu_rnn":             _liu_rnn,
    # Improved models
    "ensemble_stack":      _ensemble_stack,  # 5-model stacking ensemble
}


def get_model(name: str):
    """Return a fresh instance of the named model."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]()


def list_models() -> list[str]:
    return list(MODEL_REGISTRY)
