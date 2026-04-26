"""
Clinical feature selection and engineering inspired by PhysioNet 2019 top solutions.

Key insight from competition: 1st place (Morrill et al.) won with ~40-50 hand-picked
clinical features, not 400+. Sepsyd (2nd) used ALL variables but minimal engineering.
Separatrix (3rd) had 407 features but ranked lower due to overfitting.

Strategy: Select core clinical variables + targeted engineering (variance, missingness)
"""

import numpy as np
import pandas as pd

# Core clinical variables selected based on top 3 solutions
# 1st place: HR, MAP, SBP, BUN, Creatinine, Bilirubin, measurement frequency
# 2nd place: All 40 variables, but variance features were most helpful
# 3rd place: 13 core covariates
_CORE_VITALS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"
]

_CORE_LABS = [
    "BUN", "Creatinine", "Bilirubin_total", "Lactate", "Glucose"
]

_DEMOGRAPHICS = ["Age", "Gender"]

SELECTED_FEATURES = _CORE_VITALS + _CORE_LABS + _DEMOGRAPHICS


def select_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only core clinical variables (inspired by 1st place winner).

    Reduces from 33 base variables → 15 core variables.
    Keeps ONLY raw values, drops pre-aggregated rolling stats (mean/std/min/max/trend).
    """
    # Keep: metadata/label columns + only _raw versions of selected variables
    keep_cols = set()

    # Always keep metadata
    metadata = {"patient_id", "hour", "target", "SepsisLabel",
                "Gender", "Age", "Unit1", "Unit2", "HospAdmTime",
                "hours_until_sepsis", "is_censored"}
    keep_cols.update(c for c in df.columns if c in metadata)

    # Keep ONLY _raw versions of selected base variables, drop all pre-aggregated stats
    for var in SELECTED_FEATURES:
        raw_col = f"{var}_raw"
        if raw_col in df.columns:
            keep_cols.add(raw_col)
        # Also keep base column if it exists (demographics like Age, Gender)
        elif var in df.columns:
            keep_cols.add(var)

    df = df[[c for c in df.columns if c in keep_cols]]
    return df


def add_variance_features(df: pd.DataFrame, window_hours: int = 6) -> pd.DataFrame:
    """
    Add variance features (6-hour window) instead of excessive lags.

    Sepsyd found variance features helpful; delta features were not.
    Applied only to vitals, not all variables.
    """
    raw_cols = [
        "HR_raw", "O2Sat_raw", "Temp_raw", "SBP_raw", "MAP_raw", "DBP_raw",
        "Resp_raw", "EtCO2_raw", "BUN_raw", "Creatinine_raw", "Lactate_raw"
    ]
    raw_cols = [c for c in raw_cols if c in df.columns]

    if not raw_cols:
        return df

    new_cols: dict[str, pd.Series] = {}
    grouped = df.groupby("patient_id", sort=False)

    for col in raw_cols:
        # 6-hour variance (rolling)
        new_cols[f"{col}_var6h"] = grouped[col].rolling(window=window_hours, min_periods=1).var().reset_index(drop=True)

        # Min/max over window (also helpful per Sepsyd)
        new_cols[f"{col}_min6h"] = grouped[col].rolling(window=window_hours, min_periods=1).min().reset_index(drop=True)
        new_cols[f"{col}_max6h"] = grouped[col].rolling(window=window_hours, min_periods=1).max().reset_index(drop=True)

    if new_cols:
        new_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, new_df], axis=1)

    return df


def add_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary missingness indicators (present/absent per variable).

    Sepsyd used 34 missing value flags and found them important.
    Captures measurement patterns without requiring data imputation.
    """
    raw_cols = [
        "HR_raw", "O2Sat_raw", "Temp_raw", "SBP_raw", "MAP_raw", "DBP_raw",
        "Resp_raw", "EtCO2_raw", "BUN_raw", "Creatinine_raw", "Lactate_raw"
    ]
    raw_cols = [c for c in raw_cols if c in df.columns]

    for col in raw_cols:
        df[f"{col}_missing"] = np.isnan(df[col]).astype(int)

    return df


def add_clinical_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add clinically-meaningful ratios (used by 1st place winner).

    ShockIndex = HR / SBP
    BUN/Creatinine ratio
    """
    # ShockIndex (indicator of organ perfusion)
    if "HR_raw" in df.columns and "SBP_raw" in df.columns:
        df["ShockIndex"] = df["HR_raw"] / (df["SBP_raw"] + 1e-6)

    # BUN/Creatinine ratio (kidney function indicator)
    if "BUN_raw" in df.columns and "Creatinine_raw" in df.columns:
        df["BUN_Creatinine_ratio"] = df["BUN_raw"] / (df["Creatinine_raw"] + 1e-6)

    return df


def select_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature selection and engineering at D0 level (preprocessing step).

    This affects all downstream datasets (D0-D3).

    Strategy:
    1. Select only core clinical variables (15, not 40)
    2. Add variance features (proven helpful)
    3. Add missingness indicators
    4. Add clinical ratios
    5. Result: ~60-80 features (vs 431 before)
    """
    df = select_core_features(df)
    df = df.copy()  # Ensure we have a copy to avoid SettingWithCopyWarning

    # Forward-fill imputation (minimal, only for variance calculation)
    raw_cols = [c for c in df.columns if c.endswith("_raw")]
    for col in raw_cols:
        df[col] = df.groupby("patient_id", sort=False)[col].ffill()

    df = add_variance_features(df, window_hours=6)
    df = add_missingness_features(df)
    df = add_clinical_ratios(df)

    return df
