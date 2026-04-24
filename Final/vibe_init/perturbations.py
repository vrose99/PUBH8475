"""
Dataset perturbation generators.

Each function takes the clean DataFrame and returns a perturbed copy tagged
with a 'dataset_id' string used throughout the rest of the pipeline.

Dataset IDs
-----------
  "D0"      original data
  "D1A"     row-removal — women underrepresented
  "D1B"     row-removal — men underrepresented
  "D2A"     MAR — missing values injected for women
  "D2B"     MAR — missing values injected for men
  "D3A"     noise — Gaussian noise injected for women  (optional)
  "D3B"     noise — Gaussian noise injected for men    (optional)

All functions preserve the original DataFrame (defensive copy).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import Config
from data_loader import LABEL_COL

logger = logging.getLogger(__name__)

# Columns that are metadata / demographics — never perturbed
_META_COLS = {
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
    LABEL_COL, "dataset_id",
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _numeric_feature_cols(df: pd.DataFrame, cfg: Config) -> list[str]:
    """
    Return numeric feature columns present in the (aggregated) DataFrame.

    After aggregation each raw signal becomes HR_mean, HR_std, HR_min, etc.
    We exclude demographics and metadata so only clinical features are perturbed.
    """
    sensitive = cfg.fairness.sensitive_column
    exclude = _META_COLS | {sensitive}
    return [
        c for c in df.select_dtypes(include="number").columns
        if c not in exclude
    ]


def _select_mar_columns(df: pd.DataFrame, cfg: Config) -> list[str]:
    """
    Choose which columns to blank out.  Uses cfg.perturbation.mar_columns if
    set; otherwise picks the mar_n_columns numeric columns with the highest
    completeness (realistic MAR pattern).
    """
    if cfg.perturbation.mar_columns:
        # validate that requested columns actually exist
        present = [c for c in cfg.perturbation.mar_columns if c in df.columns]
        if not present:
            logger.warning(
                "mar_columns %s not found in DataFrame — falling back to auto-select",
                cfg.perturbation.mar_columns,
            )
        else:
            return present

    candidates = _numeric_feature_cols(df, cfg)
    if not candidates:
        logger.error("No numeric feature columns found — check DataFrame schema")
        return []
    completeness = df[candidates].notna().mean().sort_values(ascending=False)
    return completeness.index[: cfg.perturbation.mar_n_columns].tolist()


def _select_noise_columns(df: pd.DataFrame, cfg: Config) -> list[str]:
    """Same column selection as MAR so D2 and D3 target the same features."""
    if cfg.perturbation.noise_columns:
        present = [c for c in cfg.perturbation.noise_columns if c in df.columns]
        if present:
            return present
        logger.warning("noise_columns not found — falling back to auto-select")

    candidates = _numeric_feature_cols(df, cfg)
    if not candidates:
        return []
    completeness = df[candidates].notna().mean().sort_values(ascending=False)
    return completeness.index[: cfg.perturbation.noise_n_columns].tolist()


def _group_mask(df: pd.DataFrame, cfg: Config, group: str) -> pd.Series:
    """Boolean mask selecting 'female' or 'male' rows."""
    col = cfg.fairness.sensitive_column
    val = cfg.fairness.female_value if group == "female" else cfg.fairness.male_value
    return df[col] == val


# ── Public perturbation functions ─────────────────────────────────────────────

def dataset_original(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset_id"] = "D0"
    return out


def dataset_row_removal(
    df: pd.DataFrame,
    cfg: Config,
    underrepresented_group: str,   # "female" | "male"
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Drop `row_removal_fraction` of the target group's rows so that group is
    severely underrepresented in training data.

    Running this for both groups separately (D1A=women removed, D1B=men
    removed) is intentional: the symmetric result — whichever group is
    underrepresented suffers in model performance — demonstrates the problem
    is structural rather than group-specific.
    """
    rng = rng or np.random.default_rng(cfg.random_state)
    out = df.copy()
    mask = _group_mask(out, cfg, underrepresented_group)
    group_idx = out.index[mask].tolist()
    n_drop = int(len(group_idx) * cfg.perturbation.row_removal_fraction)
    drop_idx = rng.choice(group_idx, size=n_drop, replace=False)
    out = out.drop(index=drop_idx).reset_index(drop=True)

    dataset_id = "D1A" if underrepresented_group == "female" else "D1B"
    out["dataset_id"] = dataset_id

    logger.info(
        "%s: dropped %d/%d %s rows (%.0f%% removed). Remaining group balance: "
        "female=%.1f%%",
        dataset_id,
        n_drop,
        len(group_idx),
        underrepresented_group,
        cfg.perturbation.row_removal_fraction * 100,
        100 * _group_mask(out, cfg, "female").mean(),
    )
    return out


def dataset_mar(
    df: pd.DataFrame,
    cfg: Config,
    affected_group: str,   # "female" | "male"
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Inject Missing-At-Random (MAR) values for a subgroup.

    For `mar_missing_fraction` of the target group's rows, set
    `mar_n_columns` clinical columns to NaN.  Column selection favours
    columns that are already reasonably complete (realistic MAR pattern).

    Note: patients outside the affected group are untouched, so downstream
    imputation must handle the resulting asymmetric missingness.
    """
    rng = rng or np.random.default_rng(cfg.random_state)
    out = df.copy()
    mar_cols = _select_mar_columns(out, cfg)
    mask = _group_mask(out, cfg, affected_group)
    group_idx = out.index[mask].tolist()
    n_affected = int(len(group_idx) * cfg.perturbation.mar_missing_fraction)
    affected_idx = rng.choice(group_idx, size=n_affected, replace=False)
    out.loc[affected_idx, mar_cols] = np.nan

    dataset_id = "D2A" if affected_group == "female" else "D2B"
    out["dataset_id"] = dataset_id

    logger.info(
        "%s: injected NaN into %d cols for %d/%d %s rows (%.0f%%). "
        "Columns: %s",
        dataset_id,
        len(mar_cols),
        n_affected,
        len(group_idx),
        affected_group,
        cfg.perturbation.mar_missing_fraction * 100,
        mar_cols,
    )
    return out


def dataset_noise(
    df: pd.DataFrame,
    cfg: Config,
    affected_group: str,   # "female" | "male"
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Inject Gaussian noise into a subset of columns for one subgroup.

    Noise scale = column_std * cfg.perturbation.noise_std_multiplier.
    This simulates systematic measurement error or data-quality differences
    across subgroups (e.g., different monitoring equipment, documentation
    practices).
    """
    rng = rng or np.random.default_rng(cfg.random_state)
    out = df.copy()
    noise_cols = _select_noise_columns(out, cfg)
    mask = _group_mask(out, cfg, affected_group)
    group_idx = out.index[mask].tolist()

    for col in noise_cols:
        col_std = out[col].std(skipna=True)
        noise = rng.normal(
            loc=0,
            scale=col_std * cfg.perturbation.noise_std_multiplier,
            size=len(group_idx),
        )
        out.loc[group_idx, col] = out.loc[group_idx, col] + noise

    dataset_id = "D3A" if affected_group == "female" else "D3B"
    out["dataset_id"] = dataset_id

    logger.info(
        "%s: added Gaussian noise (scale=%.1f×std) to %d cols for %s rows.",
        dataset_id,
        cfg.perturbation.noise_std_multiplier,
        len(noise_cols),
        affected_group,
    )
    return out


# ── Dataset catalogue builder ─────────────────────────────────────────────────

def build_all_datasets(
    df: pd.DataFrame,
    cfg: Config,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, pd.DataFrame]:
    """
    Return a dict mapping dataset_id → perturbed DataFrame according to
    the flags in cfg.

    Always includes "D0" (original).
    """
    rng = rng or np.random.default_rng(cfg.random_state)
    datasets: dict[str, pd.DataFrame] = {"D0": dataset_original(df)}

    if cfg.run_dataset_1a:
        d = dataset_row_removal(df, cfg, "female", rng)
        datasets["D1A"] = d

    if cfg.run_dataset_1b:
        d = dataset_row_removal(df, cfg, "male", rng)
        datasets["D1B"] = d

    if cfg.run_dataset_2a:
        d = dataset_mar(df, cfg, "female", rng)
        datasets["D2A"] = d

    if cfg.run_dataset_2b:
        d = dataset_mar(df, cfg, "male", rng)
        datasets["D2B"] = d

    if cfg.run_dataset_3a:
        d = dataset_noise(df, cfg, "female", rng)
        datasets["D3A"] = d

    if cfg.run_dataset_3b:
        d = dataset_noise(df, cfg, "male", rng)
        datasets["D3B"] = d

    logger.info("Built %d dataset variants: %s", len(datasets), list(datasets))
    return datasets
