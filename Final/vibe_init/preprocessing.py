from __future__ import annotations

"""
Preprocessing pipeline: imputation → scaling.

Kept separate from mitigation.py because it must run on both training and test
splits independently (fit on train, transform both) and is applied regardless
of which mitigation strategy is chosen.
"""

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import Config


def build_preprocessor(cfg: Config) -> Pipeline:
    """
    Return a sklearn Pipeline that imputes then (optionally) scales.
    Call .fit_transform(X_train) then .transform(X_test).
    """
    strategy = cfg.model.imputation_strategy

    if strategy == "knn":
        imputer = KNNImputer(n_neighbors=5)
    else:
        imputer = SimpleImputer(strategy=strategy)  # "mean" or "median"

    steps = [("imputer", imputer)]
    if cfg.model.scale_features:
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps)
