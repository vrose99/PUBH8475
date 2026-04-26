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
) -> dict:
    """
    Train and evaluate one (model, mitigation) combination on the time-series
    early-detection dataset. Includes advanced feature engineering.
    """
    from feature_engineering import engineer_features

    # Split FIRST to prevent data leakage
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

    # Apply bootstrap if enabled
    if cfg.bootstrap.enabled:
        results = apply_bootstrap_to_results(
            results, df_ts, cfg, rng, dataset_id
        )

    return results


# ── Bootstrap utilities ────────────────────────────────────────────────────

def _bootstrap_resample_patients(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    sensitive: np.ndarray,
    hours_until_sepsis: np.ndarray,
    patient_ids: np.ndarray,
    rng: np.random.Generator,
    cfg: "Config" = None,
) -> tuple:
    """
    Resample test set by patient (with replacement) for bootstrap.
    Stratified by gender: resample all unique female and male patients with replacement.
    Returns resampled arrays maintaining the same structure.
    """
    from config import Config
    if cfg is None:
        cfg = Config()

    f_val = cfg.fairness.female_value
    m_val = cfg.fairness.male_value

    # Split by gender
    f_mask = sensitive == f_val
    m_mask = sensitive == m_val

    f_pids = np.unique(patient_ids[f_mask])
    m_pids = np.unique(patient_ids[m_mask])

    # Create mapping from pid to indices
    pid_to_idx = {pid: np.where(patient_ids == pid)[0] for pid in np.unique(patient_ids)}

    # Resample with replacement: all female and male patients
    f_resampled_pids = rng.choice(f_pids, size=len(f_pids), replace=True)
    m_resampled_pids = rng.choice(m_pids, size=len(m_pids), replace=True)

    # Collect indices for resampled patients
    boot_indices = []
    for pid in f_resampled_pids:
        boot_indices.extend(pid_to_idx[pid])
    for pid in m_resampled_pids:
        boot_indices.extend(pid_to_idx[pid])

    boot_indices = np.array(boot_indices)
    return (
        y_prob[boot_indices],
        y_true[boot_indices],
        sensitive[boot_indices],
        hours_until_sepsis[boot_indices],
    )


def _compute_bootstrap_metrics(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    sensitive: np.ndarray,
    hours_until_sepsis: np.ndarray,
    cfg: Config,
) -> dict:
    """Compute fairness metrics for a single bootstrap sample."""
    report = compute_detection_fairness_report(
        y_prob=y_prob,
        y_true=y_true,
        sensitive=sensitive,
        hours_until_sepsis=hours_until_sepsis,
        cfg=cfg,
    )
    return report


def apply_bootstrap_to_results(
    results: pd.DataFrame,
    df_ts: pd.DataFrame,
    cfg: Config,
    rng: np.random.Generator,
    dataset_id: str,
) -> pd.DataFrame:
    """
    Apply bootstrap resampling to test metrics.
    Trains model once, then bootstraps the test set predictions.

    Bootstrap protocol:
      For each iteration, resample all unique female patients with replacement
      and all unique male patients with replacement from the test set.
      This maintains gender balance and scales with max_patients.
      Compute metrics on each bootstrap sample.
    """
    bootstrap_results = []

    for idx, row in results.iterrows():
        model_name = row["model"]
        mitigation_name = row["mitigation"]

        logger.info(
            "Bootstrapping %s / %s (%d iterations)",
            model_name, mitigation_name, cfg.bootstrap.n_iterations,
        )

        # Train model once and get predictions
        boot_result = evaluate_ts_single(
            df_ts=df_ts,
            model_name=model_name,
            mitigation_name=mitigation_name,
            cfg=cfg,
            rng=rng,
            return_predictions=True,
        )

        # Extract predictions
        y_prob = boot_result.pop("_y_prob")
        y_test = boot_result.pop("_y_test")
        s_test = boot_result.pop("_s_test")
        times_test = boot_result.pop("_times_test")
        patient_ids_test = boot_result.pop("_patient_ids_test")

        # Store baseline metrics
        baseline_metrics = {col: boot_result[col] for col in boot_result.keys()}

        # Bootstrap resample test set
        bootstrap_metrics = {col: [] for col in baseline_metrics.keys()}

        for boot_iter in tqdm(
            range(cfg.bootstrap.n_iterations),
            desc=f"Bootstrap {model_name}/{mitigation_name}",
            leave=False,
        ):
            try:
                # Resample test set by patient (stratified by gender: all female + all male)
                y_prob_boot, y_test_boot, s_test_boot, times_test_boot = (
                    _bootstrap_resample_patients(
                        y_prob, y_test, s_test, times_test,
                        patient_ids_test, rng, cfg=cfg,
                    )
                )

                # Compute metrics on bootstrap sample
                boot_metrics = _compute_bootstrap_metrics(
                    y_prob_boot, y_test_boot, s_test_boot, times_test_boot, cfg
                )

                # Store metrics
                for col in baseline_metrics.keys():
                    if col in boot_metrics:
                        bootstrap_metrics[col].append(boot_metrics[col])

            except Exception as exc:
                logger.warning(
                    "Bootstrap iteration %d failed for %s/%s: %s",
                    boot_iter, model_name, mitigation_name, exc,
                )

        # Aggregate bootstrap metrics
        agg_row = {
            "dataset_id": dataset_id,
            "model": model_name,
            "mitigation": mitigation_name,
        }

        # Add baseline values and bootstrap CIs
        for col, baseline_val in baseline_metrics.items():
            agg_row[col] = baseline_val

            if col in bootstrap_metrics and bootstrap_metrics[col]:
                vals = np.array(bootstrap_metrics[col])
                agg_row[f"{col}_ci_lower"] = float(np.percentile(vals, 2.5))
                agg_row[f"{col}_ci_upper"] = float(np.percentile(vals, 97.5))

        bootstrap_results.append(agg_row)

    return pd.DataFrame(bootstrap_results)
