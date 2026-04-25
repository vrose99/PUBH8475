"""
Model registry.

Add new models by inserting an entry into MODEL_REGISTRY.
Each value is a callable (no args) that returns a fresh sklearn-compatible
estimator.  The estimator must implement fit / predict / predict_proba.

Liu et al. (2019, Sci Rep) models are prefixed "liu_":
  liu_glm      — L1 logistic regression, C tuned by 10-fold CV (AUROC)
  liu_xgboost  — XGBoost with class-imbalance correction via scale_pos_weight
  liu_rnn      — MLP approximating the RNN family on aggregated static features
                 (the static pipeline has no raw time-steps; for true sequence
                  modelling see Final/models/gru_model.py)
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
    L1-regularized logistic regression with 10-fold CV C selection.

    Liu et al. report LASSO-based feature selection inside a GLM, tuned by
    10-fold cross-validation optimising AUROC, with balanced class weights to
    handle the ~8% sepsis prevalence in PhysioNet 2019.
    """
    return LogisticRegressionCV(
        Cs=10,
        cv=10,
        penalty="l1",
        solver="liblinear",
        scoring="roc_auc",
        class_weight="balanced",
        max_iter=4000,
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


def _liu_rnn():
    """
    MLP approximating the RNN model family on aggregated static features.

    Liu et al. use a recurrent network on hourly time-series.  The vibe_init
    pipeline aggregates patient records into one row before model fitting, so
    raw sequences are unavailable.  An MLP with two hidden layers and dropout
    regularisation is the closest sklearn-native analogue; it learns non-linear
    interactions among the aggregated vital/lab statistics in the same way an
    RNN would over the sequence.  For true sequence modelling see
    Final/models/gru_model.py used by Final/run_experiment.py.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-3,          # L2 regularisation (dropout analogue)
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=42,
        )),
    ])


MODEL_REGISTRY: dict[str, callable] = {
    "logistic_regression": _logistic_regression,
    "random_forest":       _random_forest,
    "gradient_boosting":   _gradient_boosting,
    "xgboost":             _xgboost,
    "lightgbm":            _lightgbm,
    "svm":                 _svm_calibrated,
    # Liu et al. (2019) model family
    "liu_glm":             _liu_glm,
    "liu_xgboost":         _liu_xgboost,
    "liu_rnn":             _liu_rnn,
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
