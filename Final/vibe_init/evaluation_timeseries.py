from __future__ import annotations

"""
Early-detection evaluation loop.

Entry points
------------
  run_timeseries_evaluation(df_ts, cfg, rng)  → pd.DataFrame
      Full grid: models × mitigations on the time-series dataset.

  evaluate_ts_single(df_ts, model_name, mitigation_name, cfg, rng) → dict
      One (model, mitigation) cell.
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config
from data_loader_timeseries import (
    LABEL_COL_TS,
    patient_level_split,
    split_Xy_sensitive,
)
from fairness_timeseries import compute_detection_fairness_report
from mitigation import get_mitigation
from models import get_model
from preprocessing import build_preprocessor
from utility_training import find_utility_threshold

logger = logging.getLogger(__name__)


def evaluate_ts_single(
    df_ts: pd.DataFrame,
    model_name: str,
    mitigation_name: str,
    cfg: Config,
    rng: np.random.Generator,
) -> dict:
    """
    Train and evaluate one (model, mitigation) combination on the time-series
    early-detection dataset.
    """
    train_df, test_df = patient_level_split(df_ts, cfg, rng)

    X_train, y_train, s_train, times_train, feat_names = split_Xy_sensitive(train_df, cfg)
    X_test,  y_test,  s_test,  times_test,  _          = split_Xy_sensitive(test_df,  cfg)

    patient_ids_test = test_df["patient_id"].values

    preprocessor = build_preprocessor(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_train = preprocessor.fit_transform(X_train)
        X_test  = preprocessor.transform(X_test)

    mitigate_fn  = get_mitigation(mitigation_name)
    base_model   = get_model(model_name)

    X_tr, y_tr, s_tr, sample_weight, prefitted_model = mitigate_fn(
        X_train, y_train, s_train, base_model, cfg
    )

    if prefitted_model is not None:
        fitted_model = prefitted_model
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

    if hasattr(fitted_model, "predict_proba"):
        y_prob = fitted_model.predict_proba(X_test)[:, 1]
    elif hasattr(fitted_model, "_pmf_predict"):
        y_prob = fitted_model._pmf_predict(X_test)[:, 1]
    else:
        raise AttributeError(f"{type(fitted_model)} has no predict_proba or _pmf_predict")

    # ── Utility-aware threshold search on training data ───────────────────────
    # Search for the threshold that maximises PhysioNet utility on the training
    # set, then evaluate the test set at that threshold instead of the fixed 0.5.
    # We predict on the original (pre-SMOTE) X_train so that hours_until_sepsis
    # alignment is guaranteed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if hasattr(fitted_model, "predict_proba"):
            y_prob_train = fitted_model.predict_proba(X_train)[:, 1]
        else:
            y_prob_train = fitted_model._pmf_predict(X_train)[:, 1]

    train_pids = train_df["patient_id"].values
    best_threshold, best_train_utility = find_utility_threshold(
        y_prob_train, times_train, patient_ids=train_pids
    )

    # Evaluate test set using the utility-optimised threshold.
    original_threshold = cfg.fairness.decision_threshold
    cfg.fairness.decision_threshold = best_threshold

    report = compute_detection_fairness_report(
        y_prob=y_prob,
        y_true=y_test,
        sensitive=s_test,
        hours_until_sepsis=times_test,
        cfg=cfg,
        patient_ids=patient_ids_test,
    )

    cfg.fairness.decision_threshold = original_threshold  # restore

    report["utility_threshold"] = best_threshold
    report["train_utility_at_threshold"] = best_train_utility

    report["model"]            = model_name
    report["mitigation"]       = mitigation_name
    report["n_train_rows"]     = len(y_train)
    report["n_test_rows"]      = len(y_test)
    report["n_train_patients"] = train_df["patient_id"].nunique()
    report["n_test_patients"]  = test_df["patient_id"].nunique()
    report["train_prevalence"] = float(y_train.mean())
    report["test_prevalence"]  = float(y_test.mean())

    f_mask = s_test == cfg.fairness.female_value
    m_mask = s_test == cfg.fairness.male_value
    logger.info(
        "Predictions [%s/%s]: %.1f%% alarm @ utility-threshold=%.2f, "
        "mean prob=%.3f, max prob=%.3f | "
        "Female: %.1f%% alarm | Male: %.1f%% alarm",
        model_name, mitigation_name,
        (y_prob >= best_threshold).mean() * 100, best_threshold,
        y_prob.mean(), y_prob.max(),
        (y_prob[f_mask] >= best_threshold).mean() * 100 if f_mask.any() else float("nan"),
        (y_prob[m_mask] >= best_threshold).mean() * 100 if m_mask.any() else float("nan"),
    )

    return report


def run_timeseries_evaluation(
    df_ts: pd.DataFrame,
    cfg: Config,
    rng: Optional[np.random.Generator] = None,
    dataset_id: str = "D0_ts",
) -> pd.DataFrame:
    """
    Run the full (model × mitigation) evaluation grid on the time-series dataset.
    Returns pd.DataFrame with one row per (model × mitigation).
    """
    rng = rng or np.random.default_rng(cfg.random_state)

    combos = [
        (mdl, mit)
        for mdl in cfg.model.models
        for mit in cfg.mitigation.strategies
    ]

    logger.info(
        "Time-series evaluation: %d models × %d mitigations = %d cells",
        len(cfg.model.models),
        len(cfg.mitigation.strategies),
        len(combos),
    )

    records = []
    for model_name, mitigation_name in tqdm(combos, desc="Evaluating"):
        try:
            result = evaluate_ts_single(
                df_ts=df_ts,
                model_name=model_name,
                mitigation_name=mitigation_name,
                cfg=cfg,
                rng=rng,
            )
            result["dataset_id"] = dataset_id
            records.append(result)
        except Exception as exc:
            logger.error(
                "FAILED  model=%s  mitigation=%s — %s",
                model_name, mitigation_name, exc,
                exc_info=True,
            )

    results = pd.DataFrame(records)

    if results.empty:
        return results

    id_cols = ["dataset_id", "model", "mitigation"]
    other   = [c for c in results.columns if c not in id_cols]
    return results[id_cols + other]
