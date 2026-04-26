"""
Advanced feature engineering for time-series sepsis detection.

Implements temporal features, forward-fill imputation, and log transformations
based on reference models (Sepsyd, Separatrix).
"""

import numpy as np
import pandas as pd

_LOG_TRANSFORM_FEATURES = {
    "Lactate", "AST", "BUN", "Creatinine", "Glucose",
    "Bilirubin_direct", "Bilirubin_total", "TroponinI",
    "Alkalinephos", "BaseExcess", "HCO3", "PaCO2",
}

_LOOKBACK_STEPS = 5


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add delta features (rates of change) and lookback features to time-series data.

    For each patient-hour row, adds:
    - Delta features: change from previous hour (X_t - X_{t-1})
    - Lookback features: values from past 5 hours (zero-padded if unavailable)

    Processes at patient level to ensure deltas don't cross patient boundaries.
    """
    numeric_cols = [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
        "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
        "Fibrinogen", "Platelets",
    ]
    raw_cols = [c for c in [f"{col}_raw" for col in numeric_cols] if c in df.columns]
    if not raw_cols:
        raw_cols = [c for c in numeric_cols if c in df.columns]

    if not raw_cols:
        return df

    new_features = {}

    for pid in df["patient_id"].unique():
        mask = df["patient_id"] == pid
        patient_idx = np.where(mask)[0]

        for col in raw_cols:
            if col not in df.columns:
                continue

            values = df.loc[mask, col].values.astype(float)

            # ── Delta features (rate of change) ────────────────────────────
            deltas = np.full_like(values, np.nan)
            for i in range(1, len(values)):
                if not np.isnan(values[i]) and not np.isnan(values[i - 1]):
                    deltas[i] = values[i] - values[i - 1]

            delta_col = f"{col}_delta"
            if delta_col not in new_features:
                new_features[delta_col] = np.full(len(df), np.nan)
            new_features[delta_col][patient_idx] = deltas

            # ── Lookback features (past 5 hours) ──────────────────────────
            for lag in range(1, _LOOKBACK_STEPS + 1):
                lookback = np.full_like(values, 0.0, dtype=float)
                for i in range(lag, len(values)):
                    if not np.isnan(values[i - lag]):
                        lookback[i] = values[i - lag]
                lag_col = f"{col}_lag{lag}"
                if lag_col not in new_features:
                    new_features[lag_col] = np.zeros(len(df))
                new_features[lag_col][patient_idx] = lookback

    # Concatenate all new features at once (much faster)
    if new_features:
        new_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_df], axis=1)

    return df


def forward_fill_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill missing values at patient level (last-observation-carried-forward).

    More appropriate than mean/median for clinical time-series where values
    change slowly and missingness indicates lack of measurement, not absence of value.
    """
    df = df.copy()

    numeric_cols = [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
        "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
        "Fibrinogen", "Platelets",
    ]
    raw_cols = [c for c in [f"{col}_raw" for col in numeric_cols] if c in df.columns]
    if not raw_cols:
        raw_cols = [c for c in numeric_cols if c in df.columns]

    for pid in df["patient_id"].unique():
        mask = df["patient_id"] == pid
        patient_idx = np.where(mask)[0]

        for col in raw_cols:
            if col not in df.columns:
                continue

            # Get values for this patient
            values = df.loc[mask, col].values.copy()

            # Forward fill
            last_valid = np.nan
            for i in range(len(values)):
                if not np.isnan(values[i]):
                    last_valid = values[i]
                elif not np.isnan(last_valid):
                    values[i] = last_valid

            df.loc[patient_idx, col] = values

    return df


def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log transformation to right-skewed features.

    Log scale helps tree-based models handle skewed distributions
    and reduces the influence of outliers.
    """
    numeric_cols = [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
        "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
        "Fibrinogen", "Platelets",
    ]

    new_features = {}

    for col in numeric_cols:
        for suffix in ["_raw", ""]:
            col_name = col + suffix
            if col_name not in df.columns:
                continue

            if col in _LOG_TRANSFORM_FEATURES:
                # Avoid log(0) and log(negative) — add small offset
                values = df[col_name].values.astype(float)
                # Only transform positive values
                mask = values > 0
                if mask.any():
                    transformed = np.full_like(values, np.nan, dtype=float)
                    transformed[mask] = np.log(values[mask])
                    new_features[f"{col_name}_log"] = transformed

    # Concatenate all new features at once
    if new_features:
        new_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_df], axis=1)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in order.

    1. Forward-fill imputation (before creating deltas)
    2. Add delta features (rates of change)
    3. Add lookback features (past 5 hours)
    4. Apply log transformations
    """
    df = forward_fill_imputation(df)
    df = add_temporal_features(df)
    df = apply_log_transforms(df)
    return df
