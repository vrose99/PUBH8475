"""
Example usage patterns for the bootstrap rebuild.

This file shows how to:
1. Set up custom training configurations
2. Iterate over bootstrap samples
3. Structure code for model fitting and evaluation
"""

import logging
from pathlib import Path

from config import Config, TrainingConfig, BootstrapConfig
from data_loader import (
    load_physionet_files,
    split_patients_by_status,
    get_rows_for_patients,
    summarize_dataset,
)
from bootstrap import BootstrapResampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_minimal_setup():
    """
    Minimal example: load data and set up bootstrap.
    """
    logger.info("\n=== Example 1: Minimal Setup ===")

    cfg = Config(
        data_dir=Path("data/physionet_sepsis"),
        training=TrainingConfig(n_patients=50),
        bootstrap=BootstrapConfig(n_iterations=10, bootstrap_sample_size=25),
    )

    df = load_physionet_files(cfg.data_dir)
    train_pids, bootstrap_pids = split_patients_by_status(
        df,
        n_train_patients=cfg.training.n_patients,
        random_state=cfg.training.random_state,
    )

    resampler = BootstrapResampler(
        bootstrap_pool_patient_ids=bootstrap_pids,
        full_df=df,
        n_iterations=cfg.bootstrap.n_iterations,
        bootstrap_sample_size=cfg.bootstrap.bootstrap_sample_size,
    )

    logger.info("Setup complete. Ready to fit models and evaluate on bootstrap.")


def example_2_iterate_bootstrap_samples():
    """
    Demonstrate iterating over bootstrap samples.
    """
    logger.info("\n=== Example 2: Bootstrap Iteration ===")

    cfg = Config(
        data_dir=Path("data/physionet_sepsis"),
        training=TrainingConfig(n_patients=100),
        bootstrap=BootstrapConfig(n_iterations=5, bootstrap_sample_size=50),
    )

    df = load_physionet_files(cfg.data_dir)
    train_pids, bootstrap_pids = split_patients_by_status(df, cfg.training.n_patients)
    train_df = get_rows_for_patients(df, train_pids)

    resampler = BootstrapResampler(
        bootstrap_pool_patient_ids=bootstrap_pids,
        full_df=df,
        n_iterations=cfg.bootstrap.n_iterations,
        bootstrap_sample_size=cfg.bootstrap.bootstrap_sample_size,
    )

    logger.info("Training set: %d rows", len(train_df))

    for i in range(resampler.n_iterations):
        sampled_pids, bootstrap_df = resampler.generate_iteration(i)

        logger.info(
            "Iteration %d: %d unique patients, %d rows",
            i,
            len(set(sampled_pids)),
            len(bootstrap_df),
        )

        # Here you would:
        # 1. Fit model on train_df
        # 2. Evaluate on bootstrap_df
        # 3. Store metrics


def example_3_custom_training_size():
    """
    Try different training sizes while keeping bootstrap fixed.
    """
    logger.info("\n=== Example 3: Multiple Training Sizes ===")

    data_dir = Path("data/physionet_sepsis")
    df = load_physionet_files(data_dir)

    training_sizes = [50, 100, 200]

    for n_train in training_sizes:
        logger.info(f"\nTraining with {n_train} patients:")

        train_pids, bootstrap_pids = split_patients_by_status(df, n_train)
        train_df = get_rows_for_patients(df, train_pids)

        summarize_dataset(train_df, f"  Training set ({n_train} patients)")
        logger.info("  Bootstrap pool: %d patients", len(bootstrap_pids))


def example_4_pseudo_model_loop():
    """
    Pseudo-code showing the structure for actual model fitting.

    Replace pass statements with actual model operations.
    """
    logger.info("\n=== Example 4: Model Fitting Structure (Pseudo-code) ===")

    cfg = Config(
        data_dir=Path("data/physionet_sepsis"),
        training=TrainingConfig(n_patients=100),
        bootstrap=BootstrapConfig(n_iterations=3, bootstrap_sample_size=50),
    )

    df = load_physionet_files(cfg.data_dir)
    train_pids, bootstrap_pids = split_patients_by_status(df, cfg.training.n_patients)
    train_df = get_rows_for_patients(df, train_pids)

    resampler = BootstrapResampler(
        bootstrap_pool_patient_ids=bootstrap_pids,
        full_df=df,
        n_iterations=cfg.bootstrap.n_iterations,
        bootstrap_sample_size=cfg.bootstrap.bootstrap_sample_size,
    )

    # 1. FIT MODEL ON TRAINING SET
    logger.info("\n1. Fitting model on training set (%d rows)...", len(train_df))
    # model = MyModel()
    # model.fit(train_df)  # <-- Model fitting happens once here

    # 2. EVALUATE ON BOOTSTRAP SAMPLES
    logger.info("2. Evaluating on %d bootstrap samples...\n", cfg.bootstrap.n_iterations)

    bootstrap_metrics = []

    for i in range(resampler.n_iterations):
        sampled_pids, bootstrap_df = resampler.generate_iteration(i)

        logger.info("Bootstrap sample %d: %d rows", i, len(bootstrap_df))

        # Evaluate the fitted model on this bootstrap sample
        # metrics = model.evaluate(bootstrap_df)
        # bootstrap_metrics.append(metrics)

        # Pseudo-operation
        pass

    logger.info("\n3. Aggregating bootstrap results...")
    # aggregated = aggregate_bootstrap_metrics(bootstrap_metrics)
    # This could be: mean, std, confidence intervals, etc.


if __name__ == "__main__":
    example_1_minimal_setup()
    example_2_iterate_bootstrap_samples()
    example_3_custom_training_size()
    example_4_pseudo_model_loop()

    logger.info("\n" + "=" * 70)
    logger.info("All examples completed.")
