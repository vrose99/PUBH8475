"""
Test full pipeline: data loading -> train/test split -> model fit -> bootstrap eval.
Uses only GLM which works on all systems.
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
from models import LogisticGLM
from training import BootstrapEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Full pipeline test with small dataset."""
    logger.info("=" * 70)
    logger.info("FULL PIPELINE TEST")
    logger.info("=" * 70)

    cfg = Config(
        data_dir=Path("data/physionet_sepsis"),
        training=TrainingConfig(n_patients=50),
        bootstrap=BootstrapConfig(n_iterations=5, bootstrap_sample_size=30),
    )

    # Step 1: Load data
    logger.info("\n[STEP 1] Loading data...")
    df = load_physionet_files(cfg.data_dir)

    # Step 2: Split
    logger.info("\n[STEP 2] Splitting patients...")
    train_pids, bootstrap_pids = split_patients_by_status(
        df, cfg.training.n_patients, stratify_by_sepsis=True
    )
    train_df = get_rows_for_patients(df, train_pids)
    summarize_dataset(train_df, "Training set")

    # Step 3: Initialize bootstrap
    logger.info("\n[STEP 3] Setting up bootstrap...")
    resampler = BootstrapResampler(
        bootstrap_pool_patient_ids=bootstrap_pids,
        full_df=df,
        n_iterations=cfg.bootstrap.n_iterations,
        bootstrap_sample_size=cfg.bootstrap.bootstrap_sample_size,
    )

    # Step 4: Fit and evaluate
    logger.info("\n[STEP 4] Fitting model and evaluating...")
    model = LogisticGLM()
    evaluator = BootstrapEvaluator(model, train_df, label_column="SepsisLabel")

    logger.info("\nEvaluating on bootstrap samples...")
    for i in range(cfg.bootstrap.n_iterations):
        pids, bootstrap_df = resampler.generate_iteration(i)
        evaluator.evaluate_iteration(bootstrap_df, i)
        logger.info(f"  Completed {i+1}/{cfg.bootstrap.n_iterations}")

    # Step 5: Aggregate
    logger.info("\n[STEP 5] Aggregating results...")
    agg = evaluator.aggregate_bootstrap_metrics()

    # Step 6: Display
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)

    for metric_name in ["auroc", "accuracy", "recall", "f1"]:
        if metric_name in agg:
            stat = agg[metric_name]
            logger.info(
                "%s: mean=%.4f ± %.4f (median=%.4f, range=[%.4f, %.4f])",
                metric_name,
                stat["mean"],
                stat["std"],
                stat["median"],
                stat["min"],
                stat["max"],
            )

    summary_df = evaluator.summary()
    logger.info("\nPer-iteration summary:\n%s", summary_df.to_string(index=False))

    logger.info("\n" + "=" * 70)
    logger.info("SUCCESS: Full pipeline executed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
