"""
Full evaluation loop.

For each (dataset × model × mitigation) combination:
  1. Split train / test (stratified)
  2. Fit preprocessor on train, transform both
  3. Apply mitigation strategy to training data
  4. Fit model (or use pre-fitted fairlearn estimator)
  5. Compute per-group fairness metrics on the test set
  6. Collect results into a long-format DataFrame

Entry point: run_evaluation(datasets, cfg) -> pd.DataFrame
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config
from data_loader import LABEL_COL
from fairness import compute_fairness_report
from mitigation import get_mitigation
from models import get_model
from preprocessing import (
    build_preprocessor,
    split_features_label,
    train_test_split_stratified,
)

logger = logging.getLogger(__name__)


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {LABEL_COL, "dataset_id"}
    return [c for c in df.columns if c not in drop]


def _predict_proba(estimator, X: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities, handling both sklearn and fairlearn."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    # fairlearn ExponentiatedGradient wraps predict but not predict_proba
    # fall back to _pmf_predict which returns probabilities
    if hasattr(estimator, "_pmf_predict"):
        return estimator._pmf_predict(X)[:, 1]
    raise AttributeError(f"{type(estimator)} has no predict_proba or _pmf_predict")


def evaluate_single(
    df: pd.DataFrame,
    dataset_id: str,
    model_name: str,
    mitigation_name: str,
    cfg: Config,
    rng: np.random.Generator,
) -> dict:
    """
    Run one (dataset, model, mitigation) cell.

    Returns a flat dict with scalar metric values.
    """
    feature_cols = _get_feature_cols(df)

    train_df, test_df = train_test_split_stratified(df, cfg, rng)

    X_train, y_train, s_train, feat_names = split_features_label(
        train_df, cfg, feature_cols
    )
    X_test, y_test, s_test, _ = split_features_label(test_df, cfg, feature_cols)

    # Preprocess: fit on train, apply to both
    preprocessor = build_preprocessor(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_train = preprocessor.fit_transform(X_train)
        X_test  = preprocessor.transform(X_test)

    # Apply mitigation
    mitigate_fn = get_mitigation(mitigation_name)
    base_model   = get_model(model_name)

    # Each mitigation returns (X_tr, y_tr, s_tr, sample_weight, prefitted_model).
    # prefitted_model is non-None only for fairness_penalty (ExponentiatedGradient
    # fits itself internally). All other strategies return None and we fit below.
    X_tr, y_tr, s_tr, sample_weight, prefitted_model = mitigate_fn(
        X_train, y_train, s_train, base_model, cfg
    )

    if prefitted_model is not None:
        # fairness_penalty: already fitted inside the mitigation function
        fitted_model = prefitted_model
    elif mitigation_name == "robust_model":
        # Swap base model for a calibrated balanced LR regardless of what
        # model_name is — this is the "is a simpler model more fair?" check.
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        fitted_model = CalibratedClassifierCV(
            LogisticRegression(
                max_iter=2000, solver="saga",
                class_weight="balanced",
                random_state=cfg.random_state,
            ),
            method="isotonic", cv=3, n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_model.fit(X_tr, y_tr)
    else:
        fitted_model = base_model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if sample_weight is not None:
                try:
                    fitted_model.fit(X_tr, y_tr, sample_weight=sample_weight)
                except TypeError:
                    logger.warning(
                        "%s does not support sample_weight — fitting without weights",
                        model_name,
                    )
                    fitted_model.fit(X_tr, y_tr)
            else:
                fitted_model.fit(X_tr, y_tr)

    # Predict
    y_prob = _predict_proba(fitted_model, X_test)

    # Compute fairness report
    report = compute_fairness_report(y_prob=y_prob, y_true=y_test, sensitive=s_test, cfg=cfg)

    report["dataset_id"]      = dataset_id
    report["model"]           = model_name
    report["mitigation"]      = mitigation_name
    report["n_train"]         = len(y_train)
    report["n_test"]          = len(y_test)
    report["train_prevalence"] = y_train.mean()
    report["test_prevalence"]  = y_test.mean()

    return report


def run_evaluation(
    datasets: dict[str, pd.DataFrame],
    cfg: Config,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Run the full evaluation grid and return results as a long-format DataFrame.

    Parameters
    ----------
    datasets : dict mapping dataset_id → perturbed DataFrame
    cfg      : project Config

    Returns
    -------
    pd.DataFrame with one row per (dataset × model × mitigation).
    """
    rng = rng or np.random.default_rng(cfg.random_state)

    combos = [
        (did, mdl, mit)
        for did in datasets
        for mdl in cfg.model.models
        for mit in cfg.mitigation.strategies
    ]

    logger.info(
        "Running evaluation: %d datasets × %d models × %d mitigations = %d cells",
        len(datasets), len(cfg.model.models),
        len(cfg.mitigation.strategies), len(combos),
    )

    records = []
    for dataset_id, model_name, mitigation_name in tqdm(combos, desc="Evaluating"):
        try:
            result = evaluate_single(
                df=datasets[dataset_id],
                dataset_id=dataset_id,
                model_name=model_name,
                mitigation_name=mitigation_name,
                cfg=cfg,
                rng=rng,
            )
            records.append(result)
        except Exception as exc:
            logger.error(
                "FAILED  dataset=%s  model=%s  mitigation=%s — %s",
                dataset_id, model_name, mitigation_name, exc,
                exc_info=True,
            )

    results = pd.DataFrame(records)

    # Reorder leading columns
    id_cols = ["dataset_id", "model", "mitigation"]
    other_cols = [c for c in results.columns if c not in id_cols]
    results = results[id_cols + other_cols]

    return results


def run_parameter_sweep(
    df_clean: pd.DataFrame,
    cfg: Config,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Appendix: sweep perturbation parameters and record how fairness metrics change.

    Returns a DataFrame with additional columns: sweep_param, sweep_value.
    """
    from perturbations import dataset_mar, dataset_row_removal

    rng = rng or np.random.default_rng(cfg.random_state)
    all_records: list[pd.DataFrame] = []

    sweep = cfg.sweep

    # Sweep 1: row_removal_fraction
    for frac in sweep.row_removal_fractions:
        cfg_copy = Config()
        cfg_copy.perturbation.row_removal_fraction = frac
        d = dataset_row_removal(df_clean, cfg_copy, "female", rng)
        res = run_evaluation({"D1A_sweep": d}, cfg_copy, rng)
        res["sweep_param"]  = "row_removal_fraction"
        res["sweep_value"]  = frac
        all_records.append(res)

    # Sweep 2: mar_missing_fraction
    for frac in sweep.mar_missing_fractions:
        cfg_copy = Config()
        cfg_copy.perturbation.mar_missing_fraction = frac
        d = dataset_mar(df_clean, cfg_copy, "female", rng)
        res = run_evaluation({"D2A_sweep": d}, cfg_copy, rng)
        res["sweep_param"]  = "mar_missing_fraction"
        res["sweep_value"]  = frac
        all_records.append(res)

    # Sweep 3: mar_n_columns
    for n_cols in sweep.mar_n_columns_values:
        cfg_copy = Config()
        cfg_copy.perturbation.mar_n_columns = n_cols
        d = dataset_mar(df_clean, cfg_copy, "female", rng)
        res = run_evaluation({"D2A_sweep": d}, cfg_copy, rng)
        res["sweep_param"]  = "mar_n_columns"
        res["sweep_value"]  = n_cols
        all_records.append(res)

    return pd.concat(all_records, ignore_index=True)
