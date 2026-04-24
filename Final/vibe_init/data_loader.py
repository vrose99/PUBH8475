"""
Load and aggregate the PhysioNet/Computing in Cardiology 2019 Sepsis dataset.

Data format: one PSV file per patient with hourly observations.
Download: https://physionet.org/content/challenge-2019/1.0.0/
  training_setA.zip  (20,336 patients)
  training_setB.zip  (20,000 patients)

Place extracted PSV files (p??????.psv) in config.data_dir.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)

# ── Column definitions ────────────────────────────────────────────────────────

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

DEMO_COLS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
LABEL_COL = "SepsisLabel"

NUMERIC_FEATURE_COLS = VITAL_COLS + LAB_COLS
ALL_FEATURE_COLS = NUMERIC_FEATURE_COLS + DEMO_COLS

# ── Aggregation helpers ───────────────────────────────────────────────────────

def _aggregate_patient(df: pd.DataFrame) -> pd.Series:
    """
    Collapse one patient's hourly rows into a single feature vector.

    Aggregation strategy (tunable):
      - Vitals / labs: mean, std, min, max, and last observed value.
      - Demographics: last row (constant per patient).
      - Label: 1 if sepsis ever occurred, else 0.
    """
    agg: dict = {}

    for col in NUMERIC_FEATURE_COLS:
        vals = df[col].dropna()
        if vals.empty:
            agg[f"{col}_mean"] = np.nan
            agg[f"{col}_std"]  = np.nan
            agg[f"{col}_min"]  = np.nan
            agg[f"{col}_max"]  = np.nan
            agg[f"{col}_last"] = np.nan
        else:
            agg[f"{col}_mean"] = vals.mean()
            agg[f"{col}_std"]  = vals.std()
            agg[f"{col}_min"]  = vals.min()
            agg[f"{col}_max"]  = vals.max()
            agg[f"{col}_last"] = vals.iloc[-1]

    for col in DEMO_COLS:
        agg[col] = df[col].iloc[-1]

    agg[LABEL_COL] = int(df[LABEL_COL].max())  # 1 if ever septic

    return pd.Series(agg)


# ── Public API ────────────────────────────────────────────────────────────────

def load_raw_psv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="|")


def load_dataset(
    cfg: Config,
    max_patients: Optional[int] = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load and aggregate all PSV files under cfg.data_dir.

    Returns a DataFrame with one row per patient, feature columns aggregated
    from hourly observations, and SepsisLabel as the binary target.

    Parameters
    ----------
    cfg          : project Config
    max_patients : cap for quick smoke-testing (None = load all)
    cache        : save/load a parquet cache to speed re-runs

    Returns
    -------
    pd.DataFrame  shape (n_patients, n_features + demographics + label)
    """
    cache_path = cfg.data_dir / "_aggregated_cache.parquet"

    if cache and cache_path.exists():
        logger.info("Loading cached aggregated dataset from %s", cache_path)
        return pd.read_parquet(cache_path)

    psv_files = sorted(cfg.data_dir.rglob("*.psv"))
    if not psv_files:
        raise FileNotFoundError(
            f"No PSV files found under {cfg.data_dir}.\n"
            "Download the PhysioNet 2019 challenge data and set cfg.data_dir."
        )

    if max_patients:
        psv_files = psv_files[:max_patients]

    logger.info("Aggregating %d patient files ...", len(psv_files))
    records = []
    for f in tqdm(psv_files, desc="Loading patients"):
        try:
            patient_df = load_raw_psv(f)
            records.append(_aggregate_patient(patient_df))
        except Exception as exc:
            logger.warning("Skipping %s — %s", f.name, exc)

    df = pd.DataFrame(records).reset_index(drop=True)

    # Coerce types
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df["Gender"] = df["Gender"].astype(float).astype("Int64")

    if cache:
        df.to_parquet(cache_path, index=False)
        logger.info("Cached aggregated dataset to %s", cache_path)

    logger.info(
        "Dataset loaded: %d patients, %.1f%% sepsis positive, "
        "%.1f%% female",
        len(df),
        100 * df[LABEL_COL].mean(),
        100 * (df["Gender"] == cfg.fairness.female_value).mean(),
    )
    return df


def get_feature_columns(df: pd.DataFrame, cfg: Config) -> list[str]:
    """Return model feature columns (everything except the label)."""
    drop = {LABEL_COL}
    return [c for c in df.columns if c not in drop]
