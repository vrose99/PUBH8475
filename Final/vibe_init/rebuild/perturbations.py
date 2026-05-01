"""
Dataset perturbations for robustness evaluation.

Variants (following main pipeline):
  D0 — Original: gender-balanced with all sepsis cases preserved
  D1A — Row removal: 50% of non-sepsis female rows removed
  D2A — Missingness-at-random: 25% of non-sepsis female rows have 25% of measurements set to NaN

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
            'D0': original balanced dataset,
            'D1A': dataset with 50% of non-sepsis female rows removed,
            'D2A': dataset with 25% of non-sepsis female rows having 25% measurements as NaN
        }
    """
    rng = np.random.default_rng(random_state)

    # ── D0: Parent — forced gender parity ────────────────────────────────────
    df_d0 = _dataset_parent_parity(df_train.copy(), female_val, rng)

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
        logger.info(
            "%s: %d patients, %d rows | Female: %d rows | Male: %d rows",
            did, n_patients, n_rows, f_rows, m_rows,
        )

    return variants


def _dataset_parent_parity(
    df_train: pd.DataFrame,
    female_val: str = 'F',
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Create balanced gender cohort while preserving ALL sepsis cases.

    Strategy:
      1. Keep all sepsis patients (SepsisLabel==1) from both genders
      2. Calculate target count per gender (minimum of both groups)
      3. Randomly subsample non-sepsis patients to reach balanced target

    This ensures zero sepsis cases are removed and male/female counts are equal.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    df = df_train.copy()

    f_mask = df["Gender"] == female_val
    m_mask = df["Gender"] != female_val

    f_pids = df[f_mask]["patient_id"].unique()
    m_pids = df[m_mask]["patient_id"].unique()

    # All sepsis patients (will be preserved)
    f_sepsis_pids = set(df[f_mask & (df["SepsisLabel"] == 1)]["patient_id"].unique())
    m_sepsis_pids = set(df[m_mask & (df["SepsisLabel"] == 1)]["patient_id"].unique())

    # All non-sepsis patients (available for subsampling)
    f_non_sepsis_pids = np.array([pid for pid in f_pids if pid not in f_sepsis_pids])
    m_non_sepsis_pids = np.array([pid for pid in m_pids if pid not in m_sepsis_pids])

    # Balanced target: min of (sepsis + available non-sepsis) per gender
    f_max = len(f_sepsis_pids) + len(f_non_sepsis_pids)
    m_max = len(m_sepsis_pids) + len(m_non_sepsis_pids)
    target_n = min(f_max, m_max)

    logger.info(
        "D0: balanced cohort (preserving all sepsis) — target %d per gender | "
        "Female: %d sepsis + up to %d non-sepsis | Male: %d sepsis + up to %d non-sepsis",
        target_n,
        len(f_sepsis_pids), target_n - len(f_sepsis_pids),
        len(m_sepsis_pids), target_n - len(m_sepsis_pids),
    )

    # Keep all sepsis patients
    keep_pids = set(f_sepsis_pids) | set(m_sepsis_pids)

    # Subsample non-sepsis patients to reach target
    n_f_need = max(0, target_n - len(f_sepsis_pids))
    if n_f_need > 0 and len(f_non_sepsis_pids) > 0:
        f_sampled = rng.choice(
            f_non_sepsis_pids,
            size=min(n_f_need, len(f_non_sepsis_pids)),
            replace=False
        )
        keep_pids.update(f_sampled)

    n_m_need = max(0, target_n - len(m_sepsis_pids))
    if n_m_need > 0 and len(m_non_sepsis_pids) > 0:
        m_sampled = rng.choice(
            m_non_sepsis_pids,
            size=min(n_m_need, len(m_non_sepsis_pids)),
            replace=False
        )
        keep_pids.update(m_sampled)

    df = df[df["patient_id"].isin(keep_pids)]
    return df.reset_index(drop=True)


def _dataset_row_removal(
    df_train: pd.DataFrame,
    female_val: str = 'F',
    rng: np.random.Generator = None,
    removal_fraction: float = 0.5,
) -> pd.DataFrame:
    """
    Remove 50% of non-sepsis rows for female patients.
    Removes entire patients (preserves time-series structure).
    Preserves all sepsis cases.

    Parameters
    ----------
    df_train : pd.DataFrame
        Dataset to perturb
    female_val : str
        Value representing female gender
    rng : np.random.Generator, optional
        Random number generator
    removal_fraction : float
        Fraction of non-sepsis females to remove (default 0.5 = 50%)

    Returns
    -------
    pd.DataFrame
        Perturbed dataset
    """
    if rng is None:
        rng = np.random.default_rng(42)

    df = df_train.copy()
    female_mask = df["Gender"] == female_val
    female_pids = df[female_mask]["patient_id"].unique()

    # Separate sepsis and non-sepsis patients
    sepsis_pids = set(df[female_mask & (df["SepsisLabel"] == 1)]["patient_id"].unique())
    non_sepsis_pids = np.array([pid for pid in female_pids if pid not in sepsis_pids])

    # Remove fraction of non-sepsis females
    n_remove = max(1, int(len(non_sepsis_pids) * removal_fraction))
    if len(non_sepsis_pids) > 0:
        remove_pids = rng.choice(
            non_sepsis_pids,
            size=min(n_remove, len(non_sepsis_pids)),
            replace=False
        )
    else:
        remove_pids = []

    df = df[~df["patient_id"].isin(remove_pids)]
    return df.reset_index(drop=True)


def _dataset_mar(
    df_train: pd.DataFrame,
    female_val: str = 'F',
    rng: np.random.Generator = None,
    missing_row_fraction: float = 0.25,
    missing_col_fraction: float = 0.25,
) -> pd.DataFrame:
    """
    Missingness-at-random: randomly select fraction of non-sepsis female rows
    and set fraction of their numeric measurements to NaN.
    Simulates differential data collection quality.

    Parameters
    ----------
    df_train : pd.DataFrame
        Dataset to perturb
    female_val : str
        Value representing female gender
    rng : np.random.Generator, optional
        Random number generator
    missing_row_fraction : float
        Fraction of non-sepsis female rows to perturb (default 0.25 = 25%)
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
    female_mask = (df["Gender"] == female_val).values
    non_sepsis_mask = (df["SepsisLabel"] == 0).values
    perturb_mask = female_mask & non_sepsis_mask

    n_available = perturb_mask.sum()

    # Select fraction of non-sepsis female rows to perturb
    n_perturb = max(1, int(n_available * missing_row_fraction))
    if n_available > 0:
        perturb_idx = rng.choice(
            np.where(perturb_mask)[0],
            size=min(n_perturb, n_available),
            replace=False
        )
    else:
        perturb_idx = []

    # For each perturbed row, set fraction of numeric columns to NaN
    numeric_cols = _get_numeric_cols(df)
    n_cols_blank = max(1, int(len(numeric_cols) * missing_col_fraction))

    for idx in perturb_idx:
        cols_to_blank = rng.choice(numeric_cols, size=n_cols_blank, replace=False)
        df.iloc[idx, df.columns.get_indexer(cols_to_blank)] = np.nan

    return df.reset_index(drop=True)
