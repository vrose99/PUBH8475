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
    return_predictions: bool = False,
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Train and evaluate one (model, mitigation) combination on the time-series
    early-detection dataset. Includes advanced feature engineering.

    If train_df/test_df are provided they are used directly (bootstrap path).
    Otherwise a standard patient-level split is performed on df_ts.
    """
    from feature_engineering import engineer_features

    # Split FIRST to prevent data leakage
    if train_df is None or test_df is None:
        train_df, test_df = patient_level_split(df_ts, cfg, rng)

    # Apply feature engineering separately to each split
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

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

    # ── Use conservative fixed threshold instead of training-optimized search ─
    # Training-optimized thresholds overfit and fail on test.
    # Use a fixed low threshold (0.3) to catch more sepsis cases.
    # This is more conservative than the default 0.5.
    best_threshold = 0.3
    best_train_utility = np.nan  # not applicable with fixed threshold

    # ── Per-group threshold rescaling (threshold_optimization mitigation) ──────
    # When the mitigation returned a _PerGroupThresholdWrapper the model stores
    # per_group_thresholds_ = {female_val: t_f, male_val: t_m}.
    # We apply a logit shift so that (adjusted_prob >= best_threshold) is
    # exactly equivalent to (raw_prob >= per_group_threshold) for each group.
    # This preserves the existing single-threshold evaluation path while
    # honouring the group-specific operating points found during calibration.
    per_group_thresholds = getattr(fitted_model, "per_group_thresholds_", None)
    if per_group_thresholds is not None:
        t_global = best_threshold
        logit_global = np.log(t_global / (1.0 - t_global))
        y_prob = y_prob.copy()
        for g_val, t_g in per_group_thresholds.items():
            g_mask = s_test == g_val
            if not g_mask.any() or not (0.0 < t_g < 1.0):
                continue
            logit_g = np.log(t_g / (1.0 - t_g))
            eps = 1e-9
            p_clip = np.clip(y_prob[g_mask], eps, 1.0 - eps)
            logit_p = np.log(p_clip / (1.0 - p_clip))
            y_prob[g_mask] = 1.0 / (1.0 + np.exp(-(logit_p - logit_g + logit_global)))
        logger.info(
            "Per-group thresholds applied: %s",
            {k: f"{v:.3f}" for k, v in per_group_thresholds.items()},
        )

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

    if return_predictions:
        report["_y_prob"] = y_prob
        report["_y_test"] = y_test
        report["_s_test"] = s_test
        report["_times_test"] = times_test
        report["_patient_ids_test"] = patient_ids_test

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
    results = results[id_cols + other]

    # Apply standard bootstrap if enabled
    if cfg.bootstrap.enabled:
        results = apply_bootstrap_to_results(
            results, df_ts, cfg, rng, dataset_id
        )

    return results


# ── Bootstrap utilities ────────────────────────────────────────────────────

def _bootstrap_resample_test_patients(
    test_df: pd.DataFrame,
    rng: np.random.Generator,
    cfg: "Config" = None,
    n_per_group: Optional[int] = None,
) -> pd.DataFrame:
    """
    Sample n_per_group female and n_per_group male patients WITH REPLACEMENT
    from the test pool. Defaults to all available patients per gender.
    """
    from config import Config
    if cfg is None:
        cfg = Config()

    sensitive_col = cfg.fairness.sensitive_column
    f_val = cfg.fairness.female_value
    m_val = cfg.fairness.male_value

    f_pids = test_df[test_df[sensitive_col] == f_val]["patient_id"].unique()
    m_pids = test_df[test_df[sensitive_col] == m_val]["patient_id"].unique()

    n_f = n_per_group if n_per_group is not None else len(f_pids)
    n_m = n_per_group if n_per_group is not None else len(m_pids)

    f_sampled = rng.choice(f_pids, size=n_f, replace=True)
    m_sampled = rng.choice(m_pids, size=n_m, replace=True)

    all_sampled = np.concatenate([f_sampled, m_sampled])

    # Build the resampled DataFrame, handling repeated patients correctly
    frames = [test_df[test_df["patient_id"] == pid] for pid in all_sampled]
    return pd.concat(frames, ignore_index=True)


def apply_bootstrap_to_results(
    results: pd.DataFrame,
    df_ts: pd.DataFrame,
    cfg: Config,
    rng: np.random.Generator,
    dataset_id: str,
) -> pd.DataFrame:
    """
    Bootstrap: fixed training set, resampled test patients per iteration.

    Protocol:
      1. Split all patients into train pool + test pool
      2. Cap train pool to cfg.bootstrap.max_patients_train
      3. For each bootstrap iteration:
         - Sample cfg.bootstrap.max_patients_test patients WITH REPLACEMENT
           from the test pool (stratified by gender)
         - Evaluate fixed trained model on that sample
         - Collect metrics
      4. Aggregate: median and 2.5/97.5 percentile CIs
    """
    from data_loader_timeseries import patient_level_split

    # ── Build fixed train set and test pool ──────────────────────────────────
    train_df_full, test_pool_df = patient_level_split(df_ts, cfg, rng)

    # Cap training patients if requested
    max_train = cfg.bootstrap.max_patients_train
    if max_train and train_df_full["patient_id"].nunique() > max_train:
        sensitive_col = cfg.fairness.sensitive_column
        f_val = cfg.fairness.female_value
        m_val = cfg.fairness.male_value

        f_pids = train_df_full[train_df_full[sensitive_col] == f_val]["patient_id"].unique()
        m_pids = train_df_full[train_df_full[sensitive_col] == m_val]["patient_id"].unique()

        # Keep equal numbers per gender up to max_train total
        n_per_group = max_train // 2
        f_keep = rng.choice(f_pids, size=min(n_per_group, len(f_pids)), replace=False)
        m_keep = rng.choice(m_pids, size=min(n_per_group, len(m_pids)), replace=False)
        keep_pids = np.concatenate([f_keep, m_keep])
        train_df = train_df_full[train_df_full["patient_id"].isin(keep_pids)].reset_index(drop=True)
    else:
        train_df = train_df_full

    logger.info(
        "Bootstrap setup: train=%d patients (%d rows) | test pool=%d patients (%d rows) | "
        "sampling %d per iteration",
        train_df["patient_id"].nunique(), len(train_df),
        test_pool_df["patient_id"].nunique(), len(test_pool_df),
        cfg.bootstrap.max_patients_test,
    )

    bootstrap_results = []

    for idx, row in results.iterrows():
        model_name = row["model"]
        mitigation_name = row["mitigation"]

        logger.info(
            "Bootstrap %s / %s (%d iterations)",
            model_name, mitigation_name, cfg.bootstrap.n_iterations,
        )

        baseline_metrics = {col: row[col] for col in row.index
                            if col not in ["dataset_id", "model", "mitigation"]}
        bootstrap_metrics = {col: [] for col in baseline_metrics.keys()}

        for boot_iter in tqdm(
            range(cfg.bootstrap.n_iterations),
            desc=f"Bootstrap {model_name}/{mitigation_name}",
            leave=False,
        ):
            try:
                # Sample max_patients_test patients from test pool with replacement
                test_df_boot = _bootstrap_resample_test_patients(
                    test_pool_df, rng, cfg,
                    n_per_group=cfg.bootstrap.max_patients_test // 2,
                )

                boot_result = evaluate_ts_single(
                    df_ts=df_ts,
                    model_name=model_name,
                    mitigation_name=mitigation_name,
                    cfg=cfg,
                    rng=rng,
                    train_df=train_df,
                    test_df=test_df_boot,
                )

                for col in baseline_metrics.keys():
                    if col in boot_result:
                        bootstrap_metrics[col].append(boot_result[col])

            except Exception as exc:
                logger.warning(
                    "Bootstrap iteration %d failed for %s/%s: %s",
                    boot_iter, model_name, mitigation_name, exc,
                )

        agg_row = {
            "dataset_id": dataset_id,
            "model": model_name,
            "mitigation": mitigation_name,
        }

        for col, baseline_val in baseline_metrics.items():
            if col in bootstrap_metrics and bootstrap_metrics[col]:
                vals = np.array(bootstrap_metrics[col])
                agg_row[col] = float(np.median(vals))
                agg_row[f"{col}_ci_lower"] = float(np.percentile(vals, 2.5))
                agg_row[f"{col}_ci_upper"] = float(np.percentile(vals, 97.5))
            else:
                agg_row[col] = baseline_val

        bootstrap_results.append(agg_row)

    return pd.DataFrame(bootstrap_results)
