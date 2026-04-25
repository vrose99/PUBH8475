"""
Load and aggregate the PhysioNet/Computing in Cardiology 2019 Sepsis dataset.

Data format: one PSV file per patient with hourly observations.
Download: https://physionet.org/content/challenge-2019/1.0.0/
  training_setA.zip  (20,336 patients) — subdirectory training_setA/training/
  training_setB.zip  (20,000 patients) — subdirectory training_setB/training_setB/

Set cfg.data_dir to the parent folder that contains both training_setA and
training_setB (or any folder whose rglob("*.psv") finds PSV files).
Both sets are pooled automatically.
"""

import logging
from pathlib import Path
from typing import List, Optional

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


def _find_psv_files(data_dir: Path) -> List[Path]:
    """
    Discover PSV patient files under data_dir.

    Handles the PhysioNet 2019 layout where the two training sets live in
    separate subdirectories with slightly different nesting:
      <data_dir>/training_setA/training/p??????.psv
      <data_dir>/training_setB/training_setB/p??????.psv

    Any additional PSV files found by rglob are included as well, so the
    function works with flat directories and custom layouts too.
    """
    files = sorted(data_dir.rglob("*.psv"))
    if not files:
        raise FileNotFoundError(
            f"No PSV files found under {data_dir}.\n"
            "Download the PhysioNet 2019 challenge data (training_setA and\n"
            "training_setB) and point cfg.data_dir at their parent folder."
        )
    logger.info(
        "Found %d PSV files under %s (setA + setB combined)", len(files), data_dir
    )
    return files


def load_dataset(
    cfg: Config,
    max_patients: Optional[int] = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load and aggregate all PSV files under cfg.data_dir.

    Both training_setA and training_setB are discovered automatically via
    rglob and pooled into a single DataFrame (one row per patient).

    Parameters
    ----------
    cfg          : project Config — cfg.data_dir should be the folder
                   containing the training_setA and training_setB directories
    max_patients : cap total patients for smoke-testing (None = load all);
                   draws proportionally from each set when both are present
    cache        : save/load a parquet cache to speed re-runs; cache is
                   invalidated automatically when max_patients is set

    Returns
    -------
    pd.DataFrame  shape (n_patients, n_features + demographics + label)
    """
    # Don't use a stale full-dataset cache when a patient cap is requested.
    use_cache = cache and max_patients is None
    cache_path = cfg.data_dir / "_aggregated_cache.parquet"

    if use_cache and cache_path.exists():
        logger.info("Loading cached aggregated dataset from %s", cache_path)
        return pd.read_parquet(cache_path)

    psv_files = _find_psv_files(cfg.data_dir)

    if max_patients:
        # Sample evenly across files so both sets are represented.
        step = max(1, len(psv_files) // max_patients)
        psv_files = psv_files[::step][:max_patients]
        logger.info("Capped to %d patient files for smoke-test", len(psv_files))

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

    if use_cache:
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
