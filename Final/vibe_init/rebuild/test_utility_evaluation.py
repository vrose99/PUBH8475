"""
Test utility evaluation with bootstrap samples.

Demonstrates:
1. Loading PhysioNet data with hours_until_sepsis
2. Fitting a model on training data
3. Evaluating utility on bootstrap samples
4. Aggregating utility across iterations
5. Per-group utility fairness evaluation
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
from utility import UtilityEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Full pipeline with utility evaluation."""
    logger.info("=" * 70)
    logger.info("UTILITY EVALUATION WITH BOOTSTRAP")
    logger.info("=" * 70)

    cfg = Config(
        data_dir=Path("data/physionet_sepsis"),
        training=TrainingConfig(n_patients=100),
        bootstrap=BootstrapConfig(n_iterations=10, bootstrap_sample_size=50),
    )

    # 1. Load data
    logger.info("\n[STEP 1] Loading PhysioNet data...")
    df = load_physionet_files(cfg.data_dir)
    logger.info("Dataset loaded: %d patients", df["patient_id"].nunique())

    # 2. Split
    logger.info("\n[STEP 2] Splitting patients...")
    train_pids, bootstrap_pids = split_patients_by_status(
        df, cfg.training.n_patients, stratify_by_sepsis=True
    )
    train_df = get_rows_for_patients(df, train_pids)
    summarize_dataset(train_df, "Training set")

    # 3. Initialize bootstrap
    logger.info("\n[STEP 3] Setting up bootstrap...")
    resampler = BootstrapResampler(
        bootstrap_pool_patient_ids=bootstrap_pids,
        full_df=df,
        n_iterations=cfg.bootstrap.n_iterations,
        bootstrap_sample_size=cfg.bootstrap.bootstrap_sample_size,
    )

    # 4. Fit and evaluate with utility
    logger.info("\n[STEP 4] Fitting model and evaluating utility...")
    model = LogisticGLM()
    evaluator = BootstrapEvaluator(
        model,
        train_df,
        label_column="SepsisLabel",
        hours_until_sepsis_column="hours_until_sepsis",
        patient_id_column="patient_id",
    )

    logger.info("Evaluating on %d bootstrap samples...", cfg.bootstrap.n_iterations)

    for i in range(cfg.bootstrap.n_iterations):
        pids, bootstrap_df = resampler.generate_iteration(i)
        evaluator.evaluate_iteration(
            bootstrap_df,
            i,
            compute_per_group=True,
            group_column="Gender",
        )

        if (i + 1) % 5 == 0:
            logger.info(f"  Completed {i + 1} / {cfg.bootstrap.n_iterations} iterations")

    # 5. Aggregate results
    logger.info("\n[STEP 5] Aggregating bootstrap results...")
    agg = evaluator.aggregate_bootstrap_metrics()

    # 6. Display results
    logger.info("\n" + "=" * 70)
    logger.info("BOOTSTRAP RESULTS (Standard Metrics)")
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

    # 7. Utility evaluation
    logger.info("\n" + "=" * 70)
    logger.info("UTILITY EVALUATION (PhysioNet 2019)")
    logger.info("=" * 70)

    if "utility" in agg:
        stat = agg["utility"]
        logger.info(
            "utility: mean=%.4f ± %.4f (median=%.4f, range=[%.4f, %.4f])",
            stat["mean"],
            stat["std"],
            stat["median"],
            stat["min"],
            stat["max"],
        )
    else:
        logger.warning("Utility metric not found in aggregated results")

    # 8. Per-group utility fairness
    logger.info("\n" + "=" * 70)
    logger.info("UTILITY FAIRNESS (By Gender)")
    logger.info("=" * 70)

    summary_df = evaluator.summary()

    # Compute per-group utility from stored per_group metrics
    group_utils = {}
    for metric_dict in evaluator.bootstrap_metrics:
        if "per_group" in metric_dict:
            for group_val, group_metrics in metric_dict["per_group"].items():
                if group_val not in group_utils:
                    group_utils[group_val] = []
                if "utility" in group_metrics:
                    group_utils[group_val].append(group_metrics["utility"])

    if group_utils:
        for group_val in sorted(group_utils.keys()):
            utils = group_utils[group_val]
            if len(utils) > 0:
                mean_u = sum(utils) / len(utils)
                std_u = (sum((u - mean_u) ** 2 for u in utils) / len(utils)) ** 0.5
                median_u = sorted(utils)[len(utils) // 2]
                logger.info(
                    "Group %s: mean=%.4f ± %.4f (median=%.4f, n=%d)",
                    group_val,
                    mean_u,
                    std_u,
                    median_u,
                    len(utils),
                )

        # Fairness gap
        valid_groups = {g: utils for g, utils in group_utils.items() if len(utils) > 0}
        if len(valid_groups) == 2:
            groups = sorted(valid_groups.keys())
            utils_0 = sum(valid_groups[groups[0]]) / len(valid_groups[groups[0]])
            utils_1 = sum(valid_groups[groups[1]]) / len(valid_groups[groups[1]])
            gap = abs(utils_0 - utils_1)
            logger.info(f"Utility fairness gap: {gap:.4f} (absolute difference)")
    else:
        logger.info("No per-group utility metrics available (hours_until_sepsis not in data)")

    # 9. Summary table
    logger.info("\n" + "=" * 70)
    logger.info("PER-ITERATION SUMMARY")
    logger.info("=" * 70)

    summary_table = summary_df[
        ["iteration", "auroc", "recall", "utility", "n_samples", "n_positive"]
    ]
    logger.info("\n%s", summary_table.to_string(index=False))

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE: Utility evaluation finished")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
