"""
Advanced feature engineering for time-series sepsis detection.

Implements temporal features, forward-fill imputation, and log transformations
based on reference models (Sepsyd, Separatrix).

All functions are fully vectorized (groupby + shift/diff) — no Python loops
over rows. Every function always produces the same column set regardless of
the input split so train/test feature counts always match.
"""

import numpy as np
import pandas as pd

_NUMERIC_BASE = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]

_LOG_FEATURES = {
    "Lactate", "AST", "BUN", "Creatinine", "Glucose",
    "Bilirubin_direct", "Bilirubin_total", "TroponinI",
    "Alkalinephos", "BaseExcess", "HCO3", "PaCO2",
}

_LOOKBACK_STEPS = 5


def _resolve_raw_cols(df: pd.DataFrame) -> "list[str]":
    """Return the clinical columns actually present, preferring _raw suffix."""
    raw = [f"{c}_raw" for c in _NUMERIC_BASE if f"{c}_raw" in df.columns]
    if raw:
        return raw
    return [c for c in _NUMERIC_BASE if c in df.columns]


def forward_fill_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill missing values within each patient (last-observation-carried-forward).
    Fully vectorized via groupby + ffill.
    """
    raw_cols = _resolve_raw_cols(df)
    if not raw_cols:
        return df

    filled = df.groupby("patient_id", sort=False)[raw_cols].ffill()
    df = df.copy()
    df[raw_cols] = filled
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add delta (rate of change) and lookback (lag) features.

    Uses groupby + diff/shift so the operation is fully vectorized.
    Columns are always created for all lags, filled with 0 when unavailable,
    ensuring train and test always have identical feature sets.
    """
    raw_cols = _resolve_raw_cols(df)
    if not raw_cols:
        return df

    new_cols: dict[str, pd.Series] = {}
    grouped = df.groupby("patient_id", sort=False)

    for col in raw_cols:
        # Delta: change from previous hour within same patient
        new_cols[f"{col}_delta"] = grouped[col].diff()

        # Lookback: values from previous N hours (0-filled at patient boundary)
        for lag in range(1, _LOOKBACK_STEPS + 1):
            new_cols[f"{col}_lag{lag}"] = grouped[col].shift(lag).fillna(0)

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log-transform right-skewed clinical features.

    Always creates the log column for each eligible feature (NaN where values
    are non-positive), so train and test always have the same columns.
    """
    new_cols: dict[str, pd.Series] = {}

    for base in _NUMERIC_BASE:
        if base not in _LOG_FEATURES:
            continue
        for suffix in ("_raw", ""):
            col = base + suffix
            if col not in df.columns:
                continue
            vals = df[col].where(df[col] > 0)   # NaN where non-positive
            new_cols[f"{col}_log"] = np.log(vals)

    if new_cols:
        new_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, new_df], axis=1)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in order:
      1. Forward-fill imputation  (within each patient)
      2. Delta + lookback features
      3. Log transforms
    """
    df = forward_fill_imputation(df)
    df = add_temporal_features(df)
    df = apply_log_transforms(df)
    return df
