"""
Dataset perturbations for robustness evaluation.

Variants (following main pipeline):
  D0 — Original: input data as-is
  D1A — Female patient removal: 50% of female patient IDs removed entirely (imbalanced gender)
  D2A — Asymmetric missingness-at-random (simulates data collection bias):
        Females: 50% of rows have 50% of measurements missing (high missingness)
        Males: 10% of rows have 10% of measurements missing (low missingness)

All perturbations preserve time-series structure and all sepsis cases to maintain case balance.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard numeric clinical measurement columns in PhysioNet data
_BASE_NUMERIC_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]


def _get_numeric_cols(df: pd.DataFrame) -> list:
    """
    Detect numeric clinical columns in the DataFrame.
    Looks for columns in _BASE_NUMERIC_COLS that exist in the data.
    """
    return [c for c in _BASE_NUMERIC_COLS if c in df.columns]


def build_all_datasets(
    df_train: pd.DataFrame,
    random_state: int = 42,
    female_val: str = 'F',
) -> Dict[str, pd.DataFrame]:
    """
    Build three perturbation variants from training data.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataset with columns: patient_id, SepsisLabel, Gender, numeric measurements
    random_state : int
        Random seed for reproducibility
    female_val : str
        Value representing female gender in the Gender column

    Returns
    -------
    dict
        {
            'D0': original dataset,
            'D1A': dataset with 50% of female patient IDs removed (gender imbalance),
            'D2A': dataset with gender-asymmetric missingness:
                   - Females: 50% of rows have 50% of measurements missing (high missingness)
                   - Males: 10% of rows have 10% of measurements missing (low missingness)
        }
    """
    rng = np.random.default_rng(random_state)

    # ── D0: Original (use as-is) ─────────────────────────────────────────────
    df_d0 = df_train.copy()

    # ── D1A: Row removal (females only) ──────────────────────────────────────
    df_d1a = _dataset_row_removal(df_d0.copy(), female_val, rng)

    # ── D2A: MAR (females only) ──────────────────────────────────────────────
    df_d2a = _dataset_mar(df_d0.copy(), female_val, rng)

    variants = {
        "D0": df_d0,
        "D1A": df_d1a,
        "D2A": df_d2a,
    }

    for did, dff in variants.items():
        n_patients = dff["patient_id"].nunique()
        n_rows = len(dff)
        f_rows = len(dff[dff["Gender"] == female_val]) if "Gender" in dff.columns else 0
        m_rows = len(dff[dff["Gender"] != female_val]) if "Gender" in dff.columns else 0
        n_missing = dff.isna().sum().sum()
        logger.info(
            "%s: %d patients, %d rows | Female: %d rows | Male: %d rows | Missing values: %d",
            did, n_patients, n_rows, f_rows, m_rows, n_missing,
        )

    return variants


def _dataset_row_removal(
    df_train: pd.DataFrame,
    female_val: str = 'F',
    rng: np.random.Generator = None,
    removal_fraction: float = 0.5,
) -> pd.DataFrame:
    """
    Remove 50% of female PATIENT IDs entirely (all their rows).
    Creates an imbalanced gender distribution in training data.
    Preserves all male patients and all sepsis patients.

    Parameters
    ----------
    df_train : pd.DataFrame
        Dataset to perturb
    female_val : str
        Value representing female gender
    rng : np.random.Generator, optional
        Random number generator
    removal_fraction : float
        Fraction of female patient IDs to remove (default 0.5 = 50%)

    Returns
    -------
    pd.DataFrame
        Perturbed dataset with ~50% fewer female patients
    """
    if rng is None:
        rng = np.random.default_rng(42)

    df = df_train.copy()

    # Get all female patient IDs
    female_mask = (df["Gender"] == female_val).values
    female_pids = df[female_mask]["patient_id"].unique()

    # Randomly select half of female patient IDs to remove
    n_female_to_remove = max(1, int(len(female_pids) * removal_fraction))
    pids_to_remove = rng.choice(
        female_pids,
        size=min(n_female_to_remove, len(female_pids)),
        replace=False
    )

    # Remove all rows for those patient IDs
    df = df[~df["patient_id"].isin(pids_to_remove)]

    return df.reset_index(drop=True)


def _dataset_mar(
    df_train: pd.DataFrame,
    female_val: str = 'F',
    rng: np.random.Generator = None,
    female_missing_row_fraction: float = 0.5,
    female_missing_col_fraction: float = 0.5,
    male_missing_row_fraction: float = 0.1,
    male_missing_col_fraction: float = 0.1,
) -> pd.DataFrame:
    """
    Missingness-at-random with gender-based asymmetry.
    Applies differential missingness rates to females vs males.
    Simulates realistic data collection bias where one demographic group
    has systematically lower quality records.

    Parameters
    ----------
    df_train : pd.DataFrame
        Dataset to perturb
    female_val : str
        Value representing female gender
    rng : np.random.Generator, optional
        Random number generator
    female_missing_row_fraction : float
        Fraction of female non-sepsis rows to perturb (default 0.5 = 50%)
    female_missing_col_fraction : float
        Fraction of numeric columns to set NaN in female rows (default 0.5 = 50%)
    male_missing_row_fraction : float
        Fraction of male non-sepsis rows to perturb (default 0.1 = 10%)
    male_missing_col_fraction : float
        Fraction of numeric columns to set NaN in male rows (default 0.1 = 10%)

    Returns
    -------
    pd.DataFrame
        Perturbed dataset with asymmetric missingness by gender
    """
    if rng is None:
        rng = np.random.default_rng(42)

    df = df_train.copy()
    numeric_cols = _get_numeric_cols(df)

    # Process females: high missingness
    female_non_sepsis_mask = (df["Gender"] == female_val) & (df["SepsisLabel"] == 0)
    female_indices = np.where(female_non_sepsis_mask)[0]
    if len(female_indices) > 0:
        n_female_perturb = max(1, int(len(female_indices) * female_missing_row_fraction))
        female_perturb_indices = rng.choice(
            female_indices,
            size=min(n_female_perturb, len(female_indices)),
            replace=False
        )
        n_cols_blank = max(1, int(len(numeric_cols) * female_missing_col_fraction))
        for idx in female_perturb_indices:
            cols_to_blank = rng.choice(numeric_cols, size=n_cols_blank, replace=False)
            df.iloc[idx, df.columns.get_indexer(cols_to_blank)] = np.nan

    # Process males: low missingness
    male_non_sepsis_mask = (df["Gender"] != female_val) & (df["SepsisLabel"] == 0)
    male_indices = np.where(male_non_sepsis_mask)[0]
    if len(male_indices) > 0:
        n_male_perturb = max(1, int(len(male_indices) * male_missing_row_fraction))
        male_perturb_indices = rng.choice(
            male_indices,
            size=min(n_male_perturb, len(male_indices)),
            replace=False
        )
        n_cols_blank = max(1, int(len(numeric_cols) * male_missing_col_fraction))
        for idx in male_perturb_indices:
            cols_to_blank = rng.choice(numeric_cols, size=n_cols_blank, replace=False)
            df.iloc[idx, df.columns.get_indexer(cols_to_blank)] = np.nan

    return df.reset_index(drop=True)