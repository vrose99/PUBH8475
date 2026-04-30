"""
Complete workflow: fit model on training data, evaluate on bootstrap samples.

Demonstrates:
1. Loading and splitting PhysioNet data
2. Fitting three model types (GLM, XGBoost, GRU)
3. Evaluating each model on bootstrap samples
4. Aggregating and displaying results
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
from models import get_model
from training import BootstrapEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """
    Full pipeline: train models and evaluate on bootstrap samples.
    """
    # Configuration
    cfg = Config(
        data_dir=Path("data/physionet_sepsis"),
        training=TrainingConfig(
            n_patients=200,
            random_state=42,
            stratify_by_sepsis=True,
        ),
        bootstrap=BootstrapConfig(
            n_iterations=20,
            bootstrap_sample_size=100,
            random_state=42,
        ),
    )

    logger.info("=" * 70)
    logger.info("MODEL FITTING AND BOOTSTRAP EVALUATION")
    logger.info("=" * 70)

    # 1. Load and split data
    logger.info("\n[STEP 1] Loading and splitting PhysioNet data...")
    df = load_physionet_files(cfg.data_dir)
    train_pids, bootstrap_pids = split_patients_by_status(
        df,
        n_train_patients=cfg.training.n_patients,
        random_state=cfg.training.random_state,
        stratify_by_sepsis=cfg.training.stratify_by_sepsis,
    )

    train_df = get_rows_for_patients(df, train_pids)
    summarize_dataset(train_df, "Training set")

    # 2. Initialize bootstrap resampler
    logger.info("\n[STEP 2] Initializing bootstrap resampler...")
    resampler = BootstrapResampler(
        bootstrap_pool_patient_ids=bootstrap_pids,
        full_df=df,
        n_iterations=cfg.bootstrap.n_iterations,
        bootstrap_sample_size=cfg.bootstrap.bootstrap_sample_size,
        random_state=cfg.bootstrap.random_state,
    )

    # 3. Fit and evaluate models
    model_names = ["glm", "xgboost"]
    try:
        import torch
        model_names.append("gru")
    except ImportError:
        logger.warning("PyTorch not available, skipping GRU model")

    results = {}

    for model_name in model_names:
        logger.info("\n" + "=" * 70)
        logger.info(f"MODEL: {model_name.upper()}")
        logger.info("=" * 70)

        # Create and train model
        logger.info(f"\nTraining {model_name}...")
        model = get_model(model_name)
        evaluator = BootstrapEvaluator(model, train_df, label_column="SepsisLabel")

        # Evaluate on bootstrap samples
        logger.info(f"\nEvaluating on {cfg.bootstrap.n_iterations} bootstrap samples...")

        for i in range(cfg.bootstrap.n_iterations):
            sampled_pids, bootstrap_df = resampler.generate_iteration(i)
            evaluator.evaluate_iteration(bootstrap_df, iteration_idx=i)

            if (i + 1) % 5 == 0:
                logger.info(f"  Completed {i + 1} / {cfg.bootstrap.n_iterations} iterations")

        # Aggregate results
        logger.info(f"\nAggregating bootstrap results...")
        agg = evaluator.aggregate_bootstrap_metrics()

        results[model_name] = {
            "evaluator": evaluator,
            "aggregated": agg,
            "summary_df": evaluator.summary(),
        }

        # Display results
        logger.info(f"\n{model_name.upper()} Bootstrap Results:")
        logger.info("-" * 70)

        for metric_name in ["auroc", "accuracy", "precision", "recall", "f1"]:
            if metric_name in agg:
                stat = agg[metric_name]
                logger.info(
                    "  %s: mean=%.4f ± %.4f (median=%.4f, range=[%.4f, %.4f])",
                    metric_name,
                    stat["mean"],
                    stat["std"],
                    stat["median"],
                    stat["min"],
                    stat["max"],
                )

    # 4. Compare models
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)

    comparison_df_rows = []

    for model_name, data in results.items():
        agg = data["aggregated"]
        if "auroc" in agg:
            comparison_df_rows.append({
                "Model": model_name.upper(),
                "AUROC": f"{agg['auroc']['mean']:.4f} ± {agg['auroc']['std']:.4f}",
                "Accuracy": f"{agg['accuracy']['mean']:.4f} ± {agg['accuracy']['std']:.4f}",
                "Recall": f"{agg['recall']['mean']:.4f} ± {agg['recall']['std']:.4f}",
                "F1": f"{agg['f1']['mean']:.4f} ± {agg['f1']['std']:.4f}",
            })

    if comparison_df_rows:
        import pandas as pd
        comparison_df = pd.DataFrame(comparison_df_rows)
        logger.info("\n" + comparison_df.to_string(index=False))

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("\nConfiguration used:")
    logger.info("  Training set: %d patients", cfg.training.n_patients)
    logger.info("  Bootstrap pool: %d patients", len(bootstrap_pids))
    logger.info("  Bootstrap iterations: %d", cfg.bootstrap.n_iterations)
    logger.info("  Bootstrap sample size: %d patients per iteration", cfg.bootstrap.bootstrap_sample_size)
    logger.info("\nModels fit and evaluated: %s", ", ".join([m.upper() for m in model_names]))


if __name__ == "__main__":
    main()
