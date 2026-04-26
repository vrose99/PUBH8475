"""
Time-series dataset perturbations for fairness evaluation.

All perturbations are applied to patient-hour rows, preserving the time-series structure.

Dataset hierarchy:
  D0 — Parent: original data with forced gender parity (larger gender group subsampled)
  D1A/D1B — Row removal: 75% of female/male rows removed from D0
  D2A/D2B — Missingness-at-random (MAR): 25% of measurements set to NaN for female/male rows
  D3A/D3B — Gaussian noise: 25% of measurements get random noise added for female/male rows

All variants are child datasets of D0, ensuring a controlled experiment comparing
the effect of single perturbation types while holding the base cohort constant.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

_BASE_NUMERIC_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]


def _get_numeric_cols(df) -> list:
    """
    Detect numeric clinical columns in the DataFrame.
    In the time-series dataset these are stored with a '_raw' suffix.
    Falls back to bare names if raw-suffixed columns are not present.
    """
    raw_cols = [f"{c}_raw" for c in _BASE_NUMERIC_COLS if f"{c}_raw" in df.columns]
    if raw_cols:
        return raw_cols
    return [c for c in _BASE_NUMERIC_COLS if c in df.columns]


def build_all_datasets(
    df_ts: pd.DataFrame,
    cfg: Config,
    rng: np.random.Generator,
) -> Dict[str, pd.DataFrame]:
    """
    Build all perturbation variants (D0–D3B) from a time-series dataset.
    Returns {dataset_id: df_ts_variant}.
    """
    sensitive_col = cfg.fairness.sensitive_column
    f_val = cfg.fairness.female_value
    m_val = cfg.fairness.male_value

    # ── D0: Parent — forced gender parity ────────────────────────────────────
    df_d0 = _dataset_parent_parity(df_ts, cfg, rng)

    # ── D1A/D1B: Row removal ─────────────────────────────────────────────────
    df_d1a = _dataset_row_removal(df_d0, f_val, sensitive_col, rng)
    df_d1b = _dataset_row_removal(df_d0, m_val, sensitive_col, rng)

    # ── D2A/D2B: MAR (missingness-at-random) ────────────────────────────────
    df_d2a = _dataset_mar(df_d0, f_val, sensitive_col, rng)
    df_d2b = _dataset_mar(df_d0, m_val, sensitive_col, rng)

    # ── D3A/D3B: Gaussian noise ─────────────────────────────────────────────
    df_d3a = _dataset_noise(df_d0, f_val, sensitive_col, rng)
    df_d3b = _dataset_noise(df_d0, m_val, sensitive_col, rng)

    variants = {
        "D0": df_d0,
        "D1A": df_d1a,
        "D1B": df_d1b,
        "D2A": df_d2a,
        "D2B": df_d2b,
        "D3A": df_d3a,
        "D3B": df_d3b,
    }

    for did, dff in variants.items():
        n = dff["patient_id"].nunique()
        rows = len(dff)
        f_rows = len(dff[dff[sensitive_col] == f_val])
        m_rows = len(dff[dff[sensitive_col] == m_val])
        logger.info(
            "%s: %d patients, %d rows | F: %d rows | M: %d rows",
            did, n, rows, f_rows, m_rows,
        )

    return variants


# ── D0: Parent dataset with forced gender parity ────────────────────────────

def _dataset_parent_parity(
    df_ts: pd.DataFrame,
    cfg: Config,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Create balanced gender cohort while preserving ALL sepsis cases.

    Strategy:
      1. Identify all sepsis patients in each gender group (keep all)
      2. Calculate how many non-sepsis patients we can keep per gender
      3. Use the minimum as the balanced target
      4. Randomly subsample non-sepsis patients from the majority group(s)

    This ensures:
      - Zero sepsis cases are removed
      - Male/female counts are equal
      - Non-sepsis subsampling is unbiased (random)
    """
    df = df_ts.copy()
    sensitive_col = cfg.fairness.sensitive_column
    f_val = cfg.fairness.female_value
    m_val = cfg.fairness.male_value

    f_mask = df[sensitive_col] == f_val
    m_mask = df[sensitive_col] == m_val

    f_pids = df[f_mask]["patient_id"].unique()
    m_pids = df[m_mask]["patient_id"].unique()

    # All sepsis patients (will be preserved)
    f_sepsis_pids = set(df[f_mask & (df["target"] == 1)]["patient_id"].unique())
    m_sepsis_pids = set(df[m_mask & (df["target"] == 1)]["patient_id"].unique())

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

    # Subsample non-sepsis patients to reach target
    keep_pids = set(f_sepsis_pids) | set(m_sepsis_pids)  # Keep all sepsis

    n_f_need = max(0, target_n - len(f_sepsis_pids))
    if n_f_need > 0 and len(f_non_sepsis_pids) > 0:
        f_sampled = rng.choice(f_non_sepsis_pids, size=min(n_f_need, len(f_non_sepsis_pids)), replace=False)
        keep_pids.update(f_sampled)

    n_m_need = max(0, target_n - len(m_sepsis_pids))
    if n_m_need > 0 and len(m_non_sepsis_pids) > 0:
        m_sampled = rng.choice(m_non_sepsis_pids, size=min(n_m_need, len(m_non_sepsis_pids)), replace=False)
        keep_pids.update(m_sampled)

    df = df[df["patient_id"].isin(keep_pids)]
    return df.reset_index(drop=True)


# ── D1A/D1B: Row removal ──────────────────────────────────────────────────

def _dataset_row_removal(
    df_ts: pd.DataFrame,
    target_gender: int,
    sensitive_col: str,
    rng: np.random.Generator,
    removal_fraction: float = 0.5,
) -> pd.DataFrame:
    """
    Remove 50% of rows for the target gender group.
    Removes at the patient level to preserve time-series structure.

    Preserves all sepsis cases (target=1) to avoid loss of rare positives.
    Removal is drawn only from non-sepsis cases (target=0).
    """
    df = df_ts.copy()
    target_mask = df[sensitive_col] == target_gender
    target_pids = df[target_mask]["patient_id"].unique()

    # Separate sepsis (target=1) and non-sepsis (target=0) patients
    sepsis_pids = set(df[target_mask & (df["target"] == 1)]["patient_id"].unique())
    non_sepsis_pids = np.array([pid for pid in target_pids if pid not in sepsis_pids])

    # Only remove from non-sepsis patients
    n_remove = max(1, int(len(non_sepsis_pids) * removal_fraction))
    if len(non_sepsis_pids) > 0:
        remove_pids = rng.choice(non_sepsis_pids, size=min(n_remove, len(non_sepsis_pids)), replace=False)
    else:
        remove_pids = []

    df = df[~df["patient_id"].isin(remove_pids)]
    return df.reset_index(drop=True)


# ── D2A/D2B: Missingness-at-random (MAR) ─────────────────────────────────

def _dataset_mar(
    df_ts: pd.DataFrame,
    target_gender: int,
    sensitive_col: str,
    rng: np.random.Generator,
    missing_fraction: float = 0.25,
) -> pd.DataFrame:
    """
    For target gender group: randomly select 25% of rows and set 25% of
    numeric measurements to NaN.  Simulates differential data collection quality.
    """
    df = df_ts.copy()
    target_mask = (df[sensitive_col] == target_gender).values
    n_target = target_mask.sum()

    # Select 25% of target rows to perturb
    n_perturb = max(1, int(n_target * missing_fraction))
    perturb_idx = rng.choice(np.where(target_mask)[0], size=n_perturb, replace=False)

    # For each perturbed row, blank out 25% of numeric columns
    numeric_cols = _get_numeric_cols(df)
    n_cols_blank = max(1, int(len(numeric_cols) * missing_fraction))

    for idx in perturb_idx:
        cols_to_blank = rng.choice(numeric_cols, size=n_cols_blank, replace=False)
        df.iloc[idx, df.columns.get_indexer(cols_to_blank)] = np.nan

    return df.reset_index(drop=True)


# ── D3A/D3B: Gaussian noise ──────────────────────────────────────────────

def _dataset_noise(
    df_ts: pd.DataFrame,
    target_gender: int,
    sensitive_col: str,
    rng: np.random.Generator,
    noise_fraction: float = 0.25,
    noise_std_mult: float = 1.0,
) -> pd.DataFrame:
    """
    For target gender group: randomly select 25% of rows and add Gaussian noise
    (1× std) to 25% of numeric measurements.
    """
    df = df_ts.copy()
    target_mask = (df[sensitive_col] == target_gender).values
    n_target = target_mask.sum()

    # Select 25% of target rows to perturb
    n_perturb = max(1, int(n_target * noise_fraction))
    perturb_idx = rng.choice(np.where(target_mask)[0], size=n_perturb, replace=False)

    numeric_cols = _get_numeric_cols(df)
    n_cols_noise = max(1, int(len(numeric_cols) * noise_fraction))

    for idx in perturb_idx:
        cols_to_noise = rng.choice(numeric_cols, size=n_cols_noise, replace=False)
        for col in cols_to_noise:
            val = df.iloc[idx][col]
            if pd.notna(val):
                std = float(df[col].std(skipna=True))
                noise = rng.normal(0, std * noise_std_mult)
                df.iloc[idx, df.columns.get_indexer([col])] = val + noise

    return df.reset_index(drop=True)
