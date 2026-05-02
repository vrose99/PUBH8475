"""
Fundamental bootstrap resampling mechanism.

Bootstrap resampling operates at the patient level:
- Bootstrap pool: patients NOT used in training
- Each iteration: sample complete patient profiles with replacement
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BootstrapResampler:
    """
    Bootstrap resampler that operates on patient-level data.

    The bootstrap pool is a set of patient IDs not used in training.
    Each iteration samples complete patient profiles with replacement.
    """

    def __init__(
        self,
        bootstrap_pool_patient_ids: np.ndarray,
        full_df: pd.DataFrame,
        n_iterations: int = 100,
        bootstrap_sample_size: int = 50,
        random_state: int = 42,
    ):
        """
        Initialize bootstrap resampler.

        Args:
            bootstrap_pool_patient_ids: Array of patient IDs available for bootstrap
            full_df: Full patient-hour DataFrame (used to fetch rows)
            n_iterations: Number of bootstrap iterations
            bootstrap_sample_size: Number of PATIENT IDs to sample per iteration
            random_state: Random seed for reproducibility
        """
        self.bootstrap_pool = bootstrap_pool_patient_ids.copy()
        self.full_df = full_df
        self.n_iterations = n_iterations
        self.bootstrap_sample_size = bootstrap_sample_size
        self.rng = np.random.default_rng(random_state)

        logger.info(
            "Bootstrap resampler initialized: "
            "%d patients in pool, %d iterations, %d patients per sample",
            len(self.bootstrap_pool),
            self.n_iterations,
            self.bootstrap_sample_size,
        )

    def generate_iteration(self, iteration_idx: int) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generate one bootstrap iteration by stratified sampling on (gender, label) pairs.

        Ensures both genders and both labels are represented in each bootstrap sample
        to guarantee fairness metrics can be computed without division by zero.

        Args:
            iteration_idx: Iteration number (0-indexed)

        Returns:
            (sampled_patient_ids, rows_dataframe)
                sampled_patient_ids: Array of sampled patient IDs (length = bootstrap_sample_size)
                rows_dataframe: All rows from those patients
        """
        # Get all data for bootstrap pool
        pool_df = self.full_df[self.full_df["patient_id"].isin(self.bootstrap_pool)].copy()

        # Stratify by (Gender, SepsisLabel) to ensure representation
        strata_key = pool_df["Gender"].astype(str) + "_" + pool_df["SepsisLabel"].astype(str)
        unique_strata = strata_key.unique()

        sampled_pids = []
        patients_per_stratum = max(1, self.bootstrap_sample_size // len(unique_strata))

        for stratum in unique_strata:
            # Get all patients in this stratum
            stratum_mask = strata_key == stratum
            stratum_pids = pool_df[stratum_mask]["patient_id"].unique()

            # Sample from this stratum with replacement
            n_to_sample = min(patients_per_stratum, self.bootstrap_sample_size - len(sampled_pids))
            if n_to_sample > 0 and len(stratum_pids) > 0:
                sampled = self.rng.choice(
                    stratum_pids,
                    size=n_to_sample,
                    replace=True,
                )
                sampled_pids.extend(sampled)

        # If we haven't reached the target size, fill with random sampling
        if len(sampled_pids) < self.bootstrap_sample_size:
            remaining = self.bootstrap_sample_size - len(sampled_pids)
            additional = self.rng.choice(
                self.bootstrap_pool,
                size=remaining,
                replace=True,
            )
            sampled_pids.extend(additional)

        sampled_pids = np.array(sampled_pids)

        # Fetch all rows for these patients
        rows_df = self.full_df[self.full_df["patient_id"].isin(sampled_pids)].copy()
        rows_df.reset_index(drop=True, inplace=True)

        logger.debug(
            "Bootstrap iteration %d: %d unique patients, %d total rows, "
            "stratified by (Gender, SepsisLabel)",
            iteration_idx,
            len(np.unique(sampled_pids)),
            len(rows_df),
        )

        return sampled_pids, rows_df

    def generate_all_iterations(self) -> List[Tuple[np.ndarray, pd.DataFrame]]:
        """
        Generate all bootstrap iterations.

        Returns:
            List of (sampled_patient_ids, rows_dataframe) tuples
        """
        iterations = []
        for i in range(self.n_iterations):
            pids, rows = self.generate_iteration(i)
            iterations.append((pids, rows))

        logger.info("Generated %d bootstrap iterations", len(iterations))
        return iterations


def summarize_bootstrap_pool(
    bootstrap_pool_patient_ids: np.ndarray,
    full_df: pd.DataFrame,
) -> dict:
    """
    Log summary statistics about the bootstrap pool.
    """
    pool_df = full_df[full_df["patient_id"].isin(bootstrap_pool_patient_ids)]

    n_patients = len(bootstrap_pool_patient_ids)
    n_rows = len(pool_df)
    sepsis_rate = pool_df["SepsisLabel"].mean() * 100

    logger.info(
        "Bootstrap pool: %d patients, %d patient-hours, %.1f%% sepsis",
        n_patients,
        n_rows,
        sepsis_rate,
    )

    return {
        "n_patients": n_patients,
        "n_rows": n_rows,
        "sepsis_rate": sepsis_rate,
    }
