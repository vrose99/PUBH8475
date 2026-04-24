"""
Model registry.

Add new models by inserting an entry into MODEL_REGISTRY.
Each value is a callable (no args) that returns a fresh sklearn-compatible
estimator.  The estimator must implement fit / predict / predict_proba.
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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


MODEL_REGISTRY: dict[str, callable] = {
    "logistic_regression": _logistic_regression,
    "random_forest":       _random_forest,
    "gradient_boosting":   _gradient_boosting,
    "xgboost":             _xgboost,
    "lightgbm":            _lightgbm,
    "svm":                 _svm_calibrated,
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
