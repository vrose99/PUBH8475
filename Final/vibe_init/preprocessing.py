"""
Preprocessing pipeline: imputation → scaling.

Kept separate from mitigation.py because it must run on both training and test
splits independently (fit on train, transform both) and is applied regardless
of which mitigation strategy is chosen.
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import Config
from data_loader import LABEL_COL


def build_preprocessor(cfg: Config) -> Pipeline:
    """
    Return a sklearn Pipeline that imputes then (optionally) scales.

    The pipeline is unfitted — call .fit_transform(X_train) then
    .transform(X_test).
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


def split_features_label(
    df: pd.DataFrame,
    cfg: Config,
    feature_cols: list[str],
):
    """
    Split a DataFrame into (X, y, sensitive) arrays.

    Returns
    -------
    X          : np.ndarray  shape (n, n_features)
    y          : np.ndarray  shape (n,)
    sensitive  : np.ndarray  shape (n,)  — sensitive attribute values
    """
    sensitive_col = cfg.fairness.sensitive_column

    # Drop dataset_id tag and label from features; keep sensitive for eval
    drop_cols = {LABEL_COL, "dataset_id"}
    feat_cols = [c for c in feature_cols if c not in drop_cols]

    X = df[feat_cols].values.astype(float)
    y = df[LABEL_COL].values.astype(int)
    sensitive = df[sensitive_col].values.astype(float)

    return X, y, sensitive, feat_cols


def train_test_split_stratified(
    df: pd.DataFrame,
    cfg: Config,
    rng: np.random.Generator,
):
    """
    Stratified split on SepsisLabel so class balance is preserved.
    Returns (train_df, test_df).
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=cfg.model.test_size,
        random_state=cfg.random_state,
        stratify=df[LABEL_COL],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
