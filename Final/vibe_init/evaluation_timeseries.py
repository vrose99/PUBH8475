"""
Early-detection evaluation loop.

Mirrors evaluation.py but uses:
  - data_loader_timeseries  for patient-hour datasets
  - fairness_timeseries     for detection-timing fairness metrics
  - patient_level_split     to prevent data leakage

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

logger = logging.getLogger(__name__)


# ── Single-cell evaluation ────────────────────────────────────────────────────

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
    early-detection dataset.

    Parameters
    ----------
    df_ts             : output of data_loader_timeseries.load_timeseries_dataset()
    model_name        : key in models.MODEL_REGISTRY
    mitigation_name   : key in mitigation.MITIGATION_REGISTRY
    cfg               : project Config
    rng               : numpy random generator
    return_predictions: if True, attach y_prob/y_true/sensitive to report dict

    Returns
    -------
    Flat dict with all detection-fairness metrics plus meta columns.
    """
    # ── Patient-level split (no leakage) ─────────────────────────────────────
    train_df, test_df = patient_level_split(df_ts, cfg, rng)

    X_train, y_train, s_train, times_train, feat_names = split_Xy_sensitive(train_df, cfg)
    X_test,  y_test,  s_test,  times_test,  _          = split_Xy_sensitive(test_df,  cfg)

    patient_ids_test = test_df["patient_id"].values

    # ── Preprocess ────────────────────────────────────────────────────────────
    preprocessor = build_preprocessor(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_train = preprocessor.fit_transform(X_train)
        X_test  = preprocessor.transform(X_test)

    # ── Mitigation ────────────────────────────────────────────────────────────
    mitigate_fn  = get_mitigation(mitigation_name)
    base_model   = get_model(model_name)

    X_tr, y_tr, s_tr, sample_weight, prefitted_model = mitigate_fn(
        X_train, y_train, s_train, base_model, cfg
    )

    # ── Fit model ─────────────────────────────────────────────────────────────
    if prefitted_model is not None:
        fitted_model = prefitted_model

    elif mitigation_name == "robust_model":
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

    # ── Predict ───────────────────────────────────────────────────────────────
    if hasattr(fitted_model, "predict_proba"):
        y_prob = fitted_model.predict_proba(X_test)[:, 1]
    elif hasattr(fitted_model, "_pmf_predict"):
        y_prob = fitted_model._pmf_predict(X_test)[:, 1]
    else:
        raise AttributeError(f"{type(fitted_model)} has no predict_proba or _pmf_predict")

    # ── Compute detection-fairness report ─────────────────────────────────────
    report = compute_detection_fairness_report(
        y_prob=y_prob,
        y_true=y_test,
        sensitive=s_test,
        hours_until_sepsis=times_test,
        cfg=cfg,
        patient_ids=patient_ids_test,
    )

    # ── Meta ──────────────────────────────────────────────────────────────────
    report["model"]            = model_name
    report["mitigation"]       = mitigation_name
    report["n_train_rows"]     = len(y_train)
    report["n_test_rows"]      = len(y_test)
    report["n_train_patients"] = train_df["patient_id"].nunique()
    report["n_test_patients"]  = test_df["patient_id"].nunique()
    report["train_prevalence"] = float(y_train.mean())
    report["test_prevalence"]  = float(y_test.mean())

    if return_predictions:
        report["_y_prob"]      = y_prob
        report["_y_true"]      = y_test
        report["_sensitive"]   = s_test
        report["_hours"]       = times_test
        report["_patient_ids"] = patient_ids_test

    return report


# ── Full grid evaluation ──────────────────────────────────────────────────────

def run_timeseries_evaluation(
    df_ts: pd.DataFrame,
    cfg: Config,
    rng: Optional[np.random.Generator] = None,
    dataset_id: str = "D0_ts",
) -> pd.DataFrame:
    """
    Run the full (model × mitigation) evaluation grid on a time-series dataset.

    Parameters
    ----------
    df_ts      : output of load_timeseries_dataset()
    cfg        : project Config
    rng        : random generator
    dataset_id : label attached to every row of results (e.g. "D0_ts")

    Returns
    -------
    pd.DataFrame with one row per (model × mitigation).
    """
    rng = rng or np.random.default_rng(cfg.random_state)

    combos = [
        (mdl, mit)
        for mdl in cfg.model.models
        for mit in cfg.mitigation.strategies
    ]

    logger.info(
        "Time-series evaluation [%s]: %d models × %d mitigations = %d cells",
        dataset_id,
        len(cfg.model.models),
        len(cfg.mitigation.strategies),
        len(combos),
    )

    records = []
    for model_name, mitigation_name in tqdm(combos, desc=f"Evaluating {dataset_id}"):
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

    # Reorder leading columns
    id_cols  = ["dataset_id", "model", "mitigation"]
    other    = [c for c in results.columns if c not in id_cols]
    results  = results[id_cols + other]

    return results


# ── Multi-dataset variant evaluation ─────────────────────────────────────────

def run_timeseries_all_variants(
    df_ts_variants: dict[str, pd.DataFrame],
    cfg: Config,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Run the grid across multiple dataset variants (D0, D1A, D1B, …) and
    concatenate results into one long-format DataFrame.

    Parameters
    ----------
    df_ts_variants : {dataset_id: df_ts} from build_timeseries_variants()
    cfg            : project Config
    rng            : random generator

    Returns
    -------
    pd.DataFrame — one row per (dataset_id × model × mitigation).
    """
    rng = rng or np.random.default_rng(cfg.random_state)
    all_results = []

    for dataset_id, df_ts in df_ts_variants.items():
        res = run_timeseries_evaluation(df_ts, cfg, rng, dataset_id=dataset_id)
        all_results.append(res)

    return pd.concat(all_results, ignore_index=True)


# ── Dataset variant builder for time-series data ──────────────────────────────

def build_timeseries_variants(
    df_ts: pd.DataFrame,
    cfg: Config,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """
    Apply the same perturbation logic as perturbations.build_all_datasets()
    but on patient-hour DataFrames.

    Returns {dataset_id: perturbed_df_ts}.
    """
    variants = {"D0_ts": df_ts.copy()}

    # Numeric clinical feature columns only (not meta/demo)
    from data_loader_timeseries import NUMERIC_COLS, _META_COLS_TS, LABEL_COL_TS
    raw_feat_cols = [f"{c}_raw" for c in NUMERIC_COLS if f"{c}_raw" in df_ts.columns]
    roll_feat_cols = [
        c for c in df_ts.columns
        if any(c.startswith(f"{nc}_roll") or c.startswith(f"{nc}_trend")
               for nc in NUMERIC_COLS)
    ]
    all_clinical_cols = raw_feat_cols + roll_feat_cols

    sensitive_col = cfg.fairness.sensitive_column

    def _row_removal(df, target_val, frac):
        target_mask = df[sensitive_col] == target_val
        # Remove at patient level so we don't create partial patients
        target_pids = df[target_mask]["patient_id"].unique()
        n_remove = int(len(target_pids) * frac)
        remove_pids = rng.choice(target_pids, size=n_remove, replace=False)
        return df[~df["patient_id"].isin(remove_pids)].copy()

    def _mar(df, target_val, frac, n_cols):
        df = df.copy()
        target_mask = (df[sensitive_col] == target_val).values
        cols = all_clinical_cols[:n_cols] if len(all_clinical_cols) >= n_cols else all_clinical_cols
        n_rows = target_mask.sum()
        n_blank = int(n_rows * frac)
        blank_idx = rng.choice(np.where(target_mask)[0], size=n_blank, replace=False)
        df.iloc[blank_idx, df.columns.get_indexer(cols)] = np.nan
        return df

    def _noise(df, target_val, n_cols, std_mult):
        df = df.copy()
        target_mask = (df[sensitive_col] == target_val).values
        cols = raw_feat_cols[:n_cols] if len(raw_feat_cols) >= n_cols else raw_feat_cols
        for col in cols:
            std = float(df[col].std(skipna=True))
            noise = rng.normal(0, std * std_mult, target_mask.sum())
            df.loc[target_mask, col] = df.loc[target_mask, col].values + noise
        return df

    f_val = cfg.fairness.female_value
    m_val = cfg.fairness.male_value
    frac  = cfg.perturbation.row_removal_fraction
    mar_f = cfg.perturbation.mar_missing_fraction
    mar_n = cfg.perturbation.mar_n_columns
    ns_n  = cfg.perturbation.noise_n_columns
    ns_m  = cfg.perturbation.noise_std_multiplier

    if cfg.run_dataset_1a:
        variants["D1A_ts"] = _row_removal(df_ts, f_val, frac)
    if cfg.run_dataset_1b:
        variants["D1B_ts"] = _row_removal(df_ts, m_val, frac)
    if cfg.run_dataset_2a:
        variants["D2A_ts"] = _mar(df_ts, f_val, mar_f, mar_n)
    if cfg.run_dataset_2b:
        variants["D2B_ts"] = _mar(df_ts, m_val, mar_f, mar_n)
    if cfg.run_dataset_3a:
        variants["D3A_ts"] = _noise(df_ts, f_val, ns_n, ns_m)
    if cfg.run_dataset_3b:
        variants["D3B_ts"] = _noise(df_ts, m_val, ns_n, ns_m)

    logger.info(
        "Built %d time-series dataset variants: %s",
        len(variants), list(variants.keys()),
    )
    return variants
