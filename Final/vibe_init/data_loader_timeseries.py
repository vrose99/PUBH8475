from __future__ import annotations

"""
Time-series data loader for early-detection fairness analysis.

PhysioNet/CinC 2019 Challenge — PSV format (one file per patient,
one row per ICU hour).

This module converts the raw hourly time series into a supervised
learning dataset where:

  Observation unit : one (patient, ICU-hour) row
  Input features   : raw values at this hour PLUS rolling-window
                     statistics over the past `window_hours` hours
                     (mean, std, min, max, trend/slope)
  Target label     : 1  if sepsis is first recorded within the next
                     `lookahead_hours` hours, else 0
  Extra columns    : patient_id, hour, hours_until_sepsis, is_censored,
                     Gender (sensitive attribute)

Key design decisions
--------------------
* Patient-level train/test split — prevents any hourly row from the
  same patient appearing in both splits (data-leakage prevention).
* Only hours *before* sepsis onset are included for positive patients.
  Including post-onset hours would allow the model to "cheat" by
  detecting already-active sepsis rather than predicting it early.
* Rows with no valid vital signs at all are dropped (fully blank hours
  often indicate charting gaps rather than stable readings).

Entry points
------------
  load_timeseries_dataset(cfg, ...)  → pd.DataFrame  (one row per patient-hour)
  patient_level_split(df, cfg, rng) → (train_df, test_df)  (no leakage)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)

# ── Column definitions (same as data_loader.py) ───────────────────────────────

VITAL_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
]

LAB_COLS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]

NUMERIC_COLS = VITAL_COLS + LAB_COLS

DEMO_COLS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]

LABEL_COL_RAW = "SepsisLabel"
LABEL_COL_TS  = "target"       # early-detection label (this module's output)


# ── Per-patient feature engineering ──────────────────────────────────────────

def _rolling_features(
    patient_df: pd.DataFrame,
    window_hours: int = 6,
) -> pd.DataFrame:
    """
    For each numeric column build rolling statistics over the past
    `window_hours` rows (hours).

    Columns added per signal X:
      X_raw        — value at this hour (NaN if not measured)
      X_roll_mean  — rolling mean over past window_hours
      X_roll_std   — rolling std
      X_roll_min   — rolling min
      X_roll_max   — rolling max
      X_trend      — linear slope over past window (positive = rising)

    All statistics use min_periods=1 so early hours are still usable.
    """
    out_frames = []

    for col in NUMERIC_COLS:
        if col not in patient_df.columns:
            # Column absent from this file — fill with NaN
            n = len(patient_df)
            out_frames.append(pd.DataFrame({
                f"{col}_raw":       np.nan,
                f"{col}_roll_mean": np.nan,
                f"{col}_roll_std":  np.nan,
                f"{col}_roll_min":  np.nan,
                f"{col}_roll_max":  np.nan,
                f"{col}_trend":     np.nan,
            }, index=patient_df.index))
            continue

        s = patient_df[col].astype(float)
        roll = s.rolling(window=window_hours, min_periods=1)

        # Trend: slope of a linear regression over the rolling window.
        # Approximation: (last − first) / (n − 1) within window; fast and
        # equivalent to the sign of the OLS slope.
        def _slope(x: pd.Series) -> float:
            vals = x.dropna().values
            n = len(vals)
            if n < 2:
                return 0.0
            t = np.arange(n)
            return float(np.polyfit(t, vals, 1)[0])

        trend = s.rolling(window=window_hours, min_periods=2).apply(_slope, raw=False)

        out_frames.append(pd.DataFrame({
            f"{col}_raw":       s,
            f"{col}_roll_mean": roll.mean(),
            f"{col}_roll_std":  roll.std(),
            f"{col}_roll_min":  roll.min(),
            f"{col}_roll_max":  roll.max(),
            f"{col}_trend":     trend,
        }, index=patient_df.index))

    return pd.concat(out_frames, axis=1)


def _process_patient(
    patient_df: pd.DataFrame,
    patient_id: str,
    lookahead_hours: int,
    window_hours: int,
) -> pd.DataFrame:
    """
    Convert one patient's hourly DataFrame into a set of (hour, features, label)
    rows ready for model training.

    Rules
    -----
    * Positive patients (sepsis_hour exists):
        - All hours included (pre-onset and post-onset) to match the
          official PhysioNet 2019 challenge evaluation setup.
        - hours_until_sepsis = sepsis_hour − current_hour
          Positive = hours before first SepsisLabel=1.
          Negative = hours after first SepsisLabel=1 (post-onset).
        - target = 1 for hours where hours_until_sepsis ≤ lookahead_hours
          (includes the lookahead window and all post-onset rows).
    * Negative patients (never septic):
        - All hours included.
        - target = 0, hours_until_sepsis = NaN.
    """
    patient_df = patient_df.reset_index(drop=True)

    # Find sepsis onset
    if LABEL_COL_RAW not in patient_df.columns:
        return pd.DataFrame()

    sepsis_rows = patient_df.index[patient_df[LABEL_COL_RAW] == 1].tolist()
    sepsis_hour = sepsis_rows[0] if sepsis_rows else None

    # Keep all hours for all patients (pre-onset and post-onset).
    # Post-onset rows (hours_until_sepsis < 0) contribute FN penalties
    # under the PhysioNet utility function when the model misses them,
    # which matches the official challenge normalisation.
    if sepsis_hour is not None:
        patient_df = patient_df.copy()

    # ── Feature engineering ────────────────────────────────────────────────
    feat_df = _rolling_features(patient_df, window_hours=window_hours)

    # Demographics (constant per patient — take first row)
    for col in DEMO_COLS:
        if col in patient_df.columns:
            feat_df[col] = patient_df[col].iloc[0]
        else:
            feat_df[col] = np.nan

    # ICULOS (time since hospital admission) is informative as a feature
    if "ICULOS" in patient_df.columns:
        feat_df["ICULOS"] = patient_df["ICULOS"].values
    else:
        feat_df["ICULOS"] = np.arange(len(patient_df))

    feat_df["patient_id"] = patient_id
    feat_df["hour"]        = np.arange(len(feat_df))

    # ── Label construction ─────────────────────────────────────────────────
    if sepsis_hour is not None:
        feat_df["hours_until_sepsis"] = sepsis_hour - feat_df["hour"]
        feat_df[LABEL_COL_TS] = (
            feat_df["hours_until_sepsis"] <= lookahead_hours
        ).astype(int)
        feat_df["is_censored"] = 0
    else:
        feat_df["hours_until_sepsis"] = np.nan
        feat_df[LABEL_COL_TS] = 0
        feat_df["is_censored"] = 1

    return feat_df.reset_index(drop=True)


# ── Public API ────────────────────────────────────────────────────────────────

def load_timeseries_dataset(
    cfg: Config,
    lookahead_hours: int = 6,
    window_hours: int = 6,
    max_patients: Optional[int] = None,
    cache: bool = True,
    drop_blank_hours: bool = True,
    psv_files: Optional[list[Path]] = None
) -> pd.DataFrame:
    """
    Load all PSV files and produce a patient-hour supervised dataset.

    Parameters
    ----------
    cfg             : project Config (cfg.data_dir must point to PSV files)
    lookahead_hours : prediction horizon — label=1 means "sepsis within this
                      many hours" (default 6, aligns with clinical urgency)
    window_hours    : rolling-window size for temporal features (default 6)
    max_patients    : cap for smoke-testing (None = all)
    cache           : save/load a parquet cache keyed by lookahead/window
    drop_blank_hours: drop rows where ALL vital signs are NaN (charting gaps)

    Returns
    -------
    pd.DataFrame with one row per (patient, ICU hour):
      - Feature columns: <signal>_raw, <signal>_roll_mean/std/min/max/trend
      - Demographic cols: Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS
      - Meta cols: patient_id, hour, hours_until_sepsis, is_censored
      - Label: ``target``  (1 = sepsis imminent, 0 = not)
    """
    # "_full" suffix = full timeline (pre + post-onset rows).
    cache_name = f"_ts_cache_la{lookahead_hours}_w{window_hours}_full.csv.gz"
    cache_path = cfg.data_dir / cache_name
    # Also handle old parquet caches gracefully
    old_cache = cfg.data_dir / f"_ts_cache_la{lookahead_hours}_w{window_hours}.parquet"

    if cache and cache_path.exists():
        logger.info("Loading cached time-series dataset from %s", cache_path)
        df = pd.read_csv(cache_path, compression="gzip")
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].astype(float).fillna(0).astype("int64")
        return df

    if psv_files is None:
        psv_files = sorted(cfg.data_dir.rglob("*.psv"))
    if not psv_files:
        raise FileNotFoundError(
            f"No PSV files found under {cfg.data_dir}.\n"
            "Download the PhysioNet 2019 challenge data and set cfg.data_dir."
        )

    if max_patients:
        psv_files = psv_files[:max_patients]

    logger.info(
        "Building time-series dataset: %d patients, lookahead=%dh, window=%dh",
        len(psv_files), lookahead_hours, window_hours,
    )

    all_records: list[pd.DataFrame] = []
    skipped = 0

    for psv_file in tqdm(psv_files, desc="Processing patients"):
        try:
            patient_df = pd.read_csv(psv_file, sep="|")
            patient_id = psv_file.stem

            rows = _process_patient(
                patient_df, patient_id, lookahead_hours, window_hours
            )
            if len(rows) > 0:
                all_records.append(rows)
        except Exception as exc:
            logger.warning("Skipping %s — %s", psv_file.name, exc)
            skipped += 1

    if skipped:
        logger.warning("Skipped %d patient files due to errors", skipped)

    df = pd.concat(all_records, ignore_index=True)

    # Drop rows where ALL vitals are NaN (complete charting gaps)
    if drop_blank_hours:
        vital_raw_cols = [f"{c}_raw" for c in VITAL_COLS if f"{c}_raw" in df.columns]
        if vital_raw_cols:
            all_nan_mask = df[vital_raw_cols].isnull().all(axis=1)
            n_before = len(df)
            df = df[~all_nan_mask].reset_index(drop=True)
            logger.info(
                "Dropped %d blank hours (%.1f%% of rows)",
                n_before - len(df), 100 * (n_before - len(df)) / n_before,
            )

    # Ensure Gender is integer (use numpy int64, NOT pandas Int64/Arrow)
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(float).fillna(0).astype("int64")

    _log_dataset_summary(df, lookahead_hours)

    if cache:
        df.to_csv(cache_path, index=False, compression="gzip")
        logger.info("Cached time-series dataset to %s", cache_path)

    return df


def _log_dataset_summary(df: pd.DataFrame, lookahead_hours: int):
    n_patients = df["patient_id"].nunique()
    n_rows     = len(df)
    n_pos      = df[LABEL_COL_TS].sum()
    n_female   = (df[df["Gender"] == 0]["patient_id"].nunique())
    n_male     = (df[df["Gender"] == 1]["patient_id"].nunique())
    prev       = df[LABEL_COL_TS].mean()

    # Per-group label prevalence (at row level)
    f_prev = df[df["Gender"] == 0][LABEL_COL_TS].mean()
    m_prev = df[df["Gender"] == 1][LABEL_COL_TS].mean()

    logger.info(
        "Time-series dataset: %d patients (%dF/%dM), %d patient-hours, "
        "%.1f%% positive (sepsis within %dh). Female positive rate: %.1f%%, Male: %.1f%%",
        n_patients, n_female, n_male, n_rows,
        100 * prev, lookahead_hours,
        100 * f_prev, 100 * m_prev,
    )


# ── Patient-level train/test split ────────────────────────────────────────────

def patient_level_split(
    df: pd.DataFrame,
    cfg: Config,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split patients (not rows) into train and test sets.

    This is critical: splitting rows would leak future hours of a patient
    into the training set.  We split unique patient_ids stratified by
    whether they ever had sepsis.

    Returns (train_df, test_df) — both are row-level DataFrames.
    """
    from sklearn.model_selection import train_test_split

    # Per-patient sepsis label for stratification
    patient_labels = (
        df.groupby("patient_id")[LABEL_COL_TS]
        .max()
        .reset_index()
        .rename(columns={LABEL_COL_TS: "ever_septic"})
    )

    train_pids, test_pids = train_test_split(
        patient_labels["patient_id"].to_numpy(),
        test_size=cfg.model.test_size,
        random_state=cfg.random_state,
        stratify=patient_labels["ever_septic"].to_numpy(),
    )

    train_df = df[df["patient_id"].isin(train_pids)].copy().reset_index(drop=True)
    test_df  = df[df["patient_id"].isin(test_pids)].copy().reset_index(drop=True)

    logger.info(
        "Patient-level split: %d train patients (%d rows) / %d test patients (%d rows)",
        len(train_pids), len(train_df), len(test_pids), len(test_df),
    )
    return train_df, test_df


# ── Feature / label extraction helpers ───────────────────────────────────────

# Columns that are metadata — not model features
_META_COLS_TS = {
    LABEL_COL_TS, "patient_id", "hour",
    "hours_until_sepsis", "is_censored",
    "Gender", "Unit1", "Unit2",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of model-input feature columns (no meta or label)."""
    return [c for c in df.columns if c not in _META_COLS_TS]


def split_Xy_sensitive(
    df: pd.DataFrame,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Extract (X, y, sensitive, hours_until_sepsis, feature_names) arrays.

    Returns
    -------
    X                  : (n_rows, n_features)
    y                  : (n_rows,)  binary early-detection label
    sensitive          : (n_rows,)  Gender values
    hours_until_sepsis : (n_rows,)  float, NaN for non-septic
    feature_names      : list[str]
    """
    feat_cols = get_feature_columns(df)
    X = df[feat_cols].values.astype(float)
    y = df[LABEL_COL_TS].values.astype(int)
    sensitive = df[cfg.fairness.sensitive_column].values.astype(float)
    hours_until_sepsis = df["hours_until_sepsis"].values.astype(float)

    return X, y, sensitive, hours_until_sepsis, feat_cols
