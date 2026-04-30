"""
Main entry point for fundamental bootstrap mechanism.

Demonstrates:
1. Loading physionet data
2. Splitting into training set and bootstrap pool at patient level
3. Initializing bootstrap resampler (without actual model fitting yet)
"""

import logging
from pathlib import Path

from config import Config, TrainingConfig, BootstrapConfig
from data_loader import (
    load_physionet_files,
    get_patient_list,
    split_patients_by_status,
    get_rows_for_patients,
    summarize_dataset,
)
from bootstrap import BootstrapResampler, summarize_bootstrap_pool

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """
    Fundamental bootstrap setup pipeline.
    """
    # Configuration
    cfg = Config(
        data_dir=Path("data/physionet_sepsis"),
        training=TrainingConfig(
            n_patients=100,
            random_state=42,
            stratify_by_sepsis=True,
        ),
        bootstrap=BootstrapConfig(
            n_iterations=100,
            bootstrap_sample_size=50,
            random_state=42,
        ),
    )

    logger.info("=" * 70)
    logger.info("FUNDAMENTAL BOOTSTRAP SETUP")
    logger.info("=" * 70)

    # 1. Load all physionet data
    logger.info("\n[STEP 1] Loading PhysioNet data...")
    df = load_physionet_files(cfg.data_dir)
    summarize_dataset(df, "Full dataset")

    # 2. Split patients: training vs bootstrap pool
    logger.info("\n[STEP 2] Splitting patients into training and bootstrap pool...")
    train_pids, bootstrap_pids = split_patients_by_status(
        df,
        n_train_patients=cfg.training.n_patients,
        random_state=cfg.training.random_state,
        stratify_by_sepsis=cfg.training.stratify_by_sepsis,
    )

    # 3. Get training data (will be used for model fitting)
    logger.info("\n[STEP 3] Extracting training data...")
    train_df = get_rows_for_patients(df, train_pids)
    summarize_dataset(train_df, "Training set")

    # 4. Summarize bootstrap pool
    logger.info("\n[STEP 4] Analyzing bootstrap pool...")
    bootstrap_stats = summarize_bootstrap_pool(bootstrap_pids, df)

    # 5. Initialize bootstrap resampler
    logger.info("\n[STEP 5] Initializing bootstrap resampler...")
    resampler = BootstrapResampler(
        bootstrap_pool_patient_ids=bootstrap_pids,
        full_df=df,
        n_iterations=cfg.bootstrap.n_iterations,
        bootstrap_sample_size=cfg.bootstrap.bootstrap_sample_size,
        random_state=cfg.bootstrap.random_state,
    )

    # 6. Show what a bootstrap iteration looks like (without running all)
    logger.info("\n[STEP 6] Demonstrating bootstrap iteration 0...")
    sampled_pids, bootstrap_df = resampler.generate_iteration(0)
    logger.info(
        "Sampled %d unique patients from pool, got %d total rows",
        len(set(sampled_pids)),
        len(bootstrap_df),
    )
    logger.info(
        "Note: %d of %d sampled patient slots were duplicates (with replacement)",
        cfg.bootstrap.bootstrap_sample_size - len(set(sampled_pids)),
        cfg.bootstrap.bootstrap_sample_size,
    )

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("BOOTSTRAP SETUP COMPLETE")
    logger.info("=" * 70)
    logger.info("\nConfiguration Summary:")
    logger.info("  Training size: %d patients", cfg.training.n_patients)
    logger.info("  Bootstrap pool size: %d patients", len(bootstrap_pids))
    logger.info("  Bootstrap iterations: %d", cfg.bootstrap.n_iterations)
    logger.info("  Bootstrap sample size: %d patients per iteration", cfg.bootstrap.bootstrap_sample_size)
    logger.info("\nReady for model fitting step (not yet implemented).")
    logger.info("For each training configuration:")
    logger.info("  1. Fit model on train_df")
    logger.info("  2. Evaluate on each of %d bootstrap samples", cfg.bootstrap.n_iterations)


if __name__ == "__main__":
    main()
