"""
Quick test of model fitting and prediction on a small subset.
"""

import logging
import numpy as np
from pathlib import Path

from config import Config, TrainingConfig, BootstrapConfig
from data_loader import (
    load_physionet_files,
    split_patients_by_status,
    get_rows_for_patients,
)
from bootstrap import BootstrapResampler
from models import get_model
from training import extract_Xy, evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Quick smoke test of models."""
    logger.info("=" * 70)
    logger.info("QUICK MODEL TEST")
    logger.info("=" * 70)

    # Load data
    logger.info("\nLoading PhysioNet data...")
    df = load_physionet_files(Path("data/physionet_sepsis"))

    # Small split for speed
    logger.info("Splitting data (25 train, 50 bootstrap samples)...")
    train_pids, bootstrap_pids = split_patients_by_status(
        df, n_train_patients=25, random_state=42
    )

    train_df = get_rows_for_patients(df, train_pids)
    logger.info("  Training: %d patients, %d rows", len(train_pids), len(train_df))

    # Test each model
    model_names = ["glm", "xgboost"]

    for model_name in model_names:
        logger.info("\n" + "-" * 70)
        logger.info(f"Testing {model_name.upper()}")
        logger.info("-" * 70)

        # Create model
        logger.info("Creating model...")
        model = get_model(model_name)

        # Train
        logger.info("Fitting model...")
        X_train, y_train = extract_Xy(train_df, "SepsisLabel")
        model.fit(X_train, y_train)

        # Test on bootstrap sample
        logger.info("Generating bootstrap sample...")
        resampler = BootstrapResampler(
            bootstrap_pool_patient_ids=bootstrap_pids,
            full_df=df,
            n_iterations=1,
            bootstrap_sample_size=50,
        )
        pids, bootstrap_df = resampler.generate_iteration(0)

        # Evaluate
        logger.info("Evaluating...")
        metrics = evaluate_model(model, bootstrap_df, "SepsisLabel")

        logger.info(
            "Results: AUROC=%.4f, Accuracy=%.4f, Recall=%.4f, F1=%.4f",
            metrics.get("auroc", np.nan),
            metrics.get("accuracy", np.nan),
            metrics.get("recall", np.nan),
            metrics.get("f1", np.nan),
        )

    logger.info("\n" + "=" * 70)
    logger.info("ALL TESTS PASSED")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
