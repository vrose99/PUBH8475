"""
Dataset perturbations for robustness evaluation.

Variants (following main pipeline):
  D0 — Original: input data as-is
  D1A — Row removal: 50% of non-sepsis rows removed (both genders)
  D2A — Missingness-at-random: 25% of non-sepsis rows have 25% of measurements set to NaN (both genders)

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
            'D1A': dataset with 50% of non-sepsis female rows removed,
            'D2A': dataset with 25% of non-sepsis female rows having 25% measurements as NaN
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
    Remove 50% of non-sepsis ROWS, stratified by gender.
    Removes at the row level (patient-hour level), not patient level.
    Preserves all sepsis rows (SepsisLabel==1).
    Applies removal to both genders to maximize dataset differentiation.

    Parameters
    ----------
    df_train : pd.DataFrame
        Dataset to perturb
    female_val : str
        Value representing female gender
    rng : np.random.Generator, optional
        Random number generator
    removal_fraction : float
        Fraction of non-sepsis rows to remove (default 0.5 = 50%)

    Returns
    -------
    pd.DataFrame
        Perturbed dataset
    """
    if rng is None:
        rng = np.random.default_rng(42)

    df = df_train.copy()

    # Select all non-sepsis rows (both genders)
    non_sepsis_mask = (df["SepsisLabel"] == 0).values

    n_removable = non_sepsis_mask.sum()
    n_remove = max(1, int(n_removable * removal_fraction))

    if n_removable > 0:
        # Get indices of rows to remove
        removable_indices = np.where(non_sepsis_mask)[0]
        remove_indices = rng.choice(
            removable_indices,
            size=min(n_remove, len(removable_indices)),
            replace=False
        )

        # Remove those rows
        df = df.drop(df.index[remove_indices])

    return df.reset_index(drop=True)


def _dataset_mar(
    df_train: pd.DataFrame,
    female_val: str = 'F',
    rng: np.random.Generator = None,
    missing_row_fraction: float = 0.25,
    missing_col_fraction: float = 0.25,
) -> pd.DataFrame:
    """
    Missingness-at-random: randomly select fraction of non-sepsis rows
    and set fraction of their numeric measurements to NaN.
    Simulates differential data collection quality.
    Applies to both genders to maximize dataset differentiation.

    Parameters
    ----------
    df_train : pd.DataFrame
        Dataset to perturb
    female_val : str
        Value representing female gender (unused, kept for API compatibility)
    rng : np.random.Generator, optional
        Random number generator
    missing_row_fraction : float
        Fraction of non-sepsis rows to perturb (default 0.25 = 25%)
    missing_col_fraction : float
        Fraction of numeric columns to set NaN in perturbed rows (default 0.25 = 25%)

    Returns
    -------
    pd.DataFrame
        Perturbed dataset
    """
    if rng is None:
        rng = np.random.default_rng(42)

    df = df_train.copy()
    non_sepsis_mask = (df["SepsisLabel"] == 0).values

    n_available = non_sepsis_mask.sum()

    # Select fraction of non-sepsis rows to perturb (both genders)
    n_perturb = max(1, int(n_available * missing_row_fraction))
    if n_available > 0:
        perturb_indices = rng.choice(
            np.where(non_sepsis_mask)[0],
            size=min(n_perturb, n_available),
            replace=False
        )
    else:
        perturb_indices = []

    # For each perturbed row, set fraction of numeric columns to NaN
    numeric_cols = _get_numeric_cols(df)
    n_cols_blank = max(1, int(len(numeric_cols) * missing_col_fraction))

    for idx in perturb_indices:
        cols_to_blank = rng.choice(numeric_cols, size=n_cols_blank, replace=False)
        df.iloc[idx, df.columns.get_indexer(cols_to_blank)] = np.nan

    return df.reset_index(drop=True)