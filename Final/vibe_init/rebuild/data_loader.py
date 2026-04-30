"""
Fundamental data loader for physionet sepsis dataset.

Loads PSV files (one per patient) and constructs patient-hour level dataset.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_physionet_files(data_dir: Path) -> pd.DataFrame:
    """
    Load all PSV files from training sets into a single DataFrame.

    Each row is a (patient, hour) observation.
    Columns include: patient_id, hour, all vital and lab measurements, Gender.
    """
    data_dir = Path(data_dir)

    dfs = []

    for train_set in ["training_setA", "training_setB"]:
        # PSV files are in training_setX/training/ subdirectory
        set_dir = data_dir / train_set / "training"
        if not set_dir.exists():
            logger.warning("Dataset directory not found: %s", set_dir)
            continue

        # Find all .psv files
        psv_files = list(set_dir.glob("*.psv"))
        logger.info("Found %d PSV files in %s", len(psv_files), train_set)

        for psv_file in psv_files:
            patient_id = psv_file.stem
            df = pd.read_csv(psv_file, sep="|")
            df.insert(0, "patient_id", patient_id)
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No PSV files found in {data_dir}")

    df = pd.concat(dfs, ignore_index=True)
    logger.info(
        "Loaded %d total patient-hours from %d patients",
        len(df),
        df["patient_id"].nunique()
    )

    return df


def get_patient_list(df: pd.DataFrame) -> np.ndarray:
    """
    Get sorted array of unique patient IDs.
    """
    return np.sort(df["patient_id"].unique())


def get_patient_sepsis_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each patient, determine if they ever had sepsis.

    Returns DataFrame with columns: [patient_id, ever_septic]
    """
    # SepsisLabel = 1 if sepsis is active, 0 otherwise
    patient_status = (
        df.groupby("patient_id")["SepsisLabel"]
        .max()
        .reset_index()
        .rename(columns={"SepsisLabel": "ever_septic"})
    )
    return patient_status


def split_patients_by_status(
    df: pd.DataFrame,
    n_train_patients: int,
    random_state: int = 42,
    stratify_by_sepsis: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split patient IDs into training and bootstrap pool.

    Args:
        df: Full patient-hour DataFrame
        n_train_patients: Number of patients to use for training
        random_state: Random seed
        stratify_by_sepsis: If True, maintain sepsis rate in both splits

    Returns:
        (train_patient_ids, bootstrap_pool_patient_ids)
    """
    patient_status = get_patient_sepsis_status(df)
    patient_ids = patient_status["patient_id"].to_numpy()

    if stratify_by_sepsis:
        stratify = patient_status["ever_septic"].to_numpy()
    else:
        stratify = None

    # Split: n_train_patients go to training, rest go to bootstrap pool
    train_pids, bootstrap_pids = train_test_split(
        patient_ids,
        train_size=n_train_patients,
        random_state=random_state,
        stratify=stratify,
    )

    logger.info(
        "Patient split: %d training patients, %d in bootstrap pool",
        len(train_pids),
        len(bootstrap_pids),
    )

    return train_pids, bootstrap_pids


def get_rows_for_patients(
    df: pd.DataFrame,
    patient_ids: np.ndarray,
) -> pd.DataFrame:
    """
    Filter DataFrame to only rows belonging to specified patients.
    """
    return df[df["patient_id"].isin(patient_ids)].reset_index(drop=True)


def add_hours_until_sepsis(
    df: pd.DataFrame,
    keep_post_onset: bool = True,
) -> pd.DataFrame:
    """
    Compute ``hours_until_sepsis`` from ``ICULOS`` and ``SepsisLabel`` in the
    raw PhysioNet PSV data.

    PhysioNet 2019 Label Definition
    --------------------------------
    Per the PhysioNet 2019 Sepsis Challenge:
        "SepsisLabel = 1 if t ≥ t_sepsis − 6"

    This means SepsisLabel flips to 1 exactly 6 hours BEFORE true sepsis onset.
    To compute hours until TRUE sepsis onset (not label flip), we add 6:
        true_onset_iculos = ICULOS_first_label_1 + 6

    Definition (per patient)
    ------------------------
    * Septic patient  : first row where SepsisLabel == 1 identifies label-flip time.
        true_onset_iculos = ICULOS_label_flip + 6  (shift to true onset)
        hours_until_sepsis = true_onset_iculos − ICULOS_current
        • Positive  → pre-onset  (hours before true sepsis onset)
        • Zero      → true sepsis onset time (t_sepsis)
        • Negative  → post-onset (after true sepsis onset)
    * Non-septic patient : hours_until_sepsis = NaN for every row.

    The utility function uses NaN to distinguish septic vs non-septic rows.

    Args:
        df: DataFrame produced by ``load_physionet_files()``
            Must contain columns: ``patient_id``, ``ICULOS``, ``SepsisLabel``
        keep_post_onset: If True (default) keep all rows including post-onset
            ones (they contribute FN penalties to the utility score).
            If False, drop rows where hours_until_sepsis < 0.

    Returns:
        Copy of ``df`` with a new ``hours_until_sepsis`` column.
    """
    df = df.copy()
    df["hours_until_sepsis"] = np.nan   # default: non-septic / unknown

    results = []

    for pid, p_df in df.groupby("patient_id", sort=False):
        p_df = p_df.sort_values("ICULOS").copy()

        onset_rows = p_df[p_df["SepsisLabel"] == 1]

        if len(onset_rows) > 0:
            # First row where SepsisLabel = 1 is at t_sepsis - 6 (per PhysioNet definition)
            # Add 6 hours to get true sepsis onset time
            onset_iculos = int(onset_rows["ICULOS"].iloc[0])
            true_onset_iculos = onset_iculos + 6
            p_df["hours_until_sepsis"] = true_onset_iculos - p_df["ICULOS"]

            if not keep_post_onset:
                p_df = p_df[p_df["hours_until_sepsis"] >= 0]
        # else: non-septic — hours_until_sepsis stays NaN

        results.append(p_df)

    result_df = pd.concat(results, ignore_index=True)
    n_septic_rows = result_df["hours_until_sepsis"].notna().sum()
    n_patients_septic = (
        result_df.groupby("patient_id")["hours_until_sepsis"].any().sum()
    )
    logger.info(
        "hours_until_sepsis computed: %d septic rows across %d septic patients",
        n_septic_rows,
        n_patients_septic,
    )
    return result_df


def summarize_dataset(df: pd.DataFrame, label: str = ""):
    """
    Log summary statistics about a dataset.
    """
    n_patients = df["patient_id"].nunique()
    n_rows = len(df)
    sepsis_rate = df["SepsisLabel"].mean() * 100 if "SepsisLabel" in df.columns else None

    info = f"{label}: {n_patients:,} patients, {n_rows:,} patient-hours"
    if sepsis_rate is not None:
        info += f", {sepsis_rate:.1f}% sepsis"

    logger.info(info)
    return {"n_patients": n_patients, "n_rows": n_rows, "sepsis_rate": sepsis_rate}
