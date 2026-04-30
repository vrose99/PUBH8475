"""
Fairness evaluation utilities.

Evaluates model performance separately for demographic groups (e.g., Gender)
across bootstrap samples, comparing metrics like AUROC and recall by group.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_fairness_gap(
    metrics_by_group: Dict,
    metric_name: str = "auroc",
) -> Dict[str, float]:
    """
    Compute fairness gaps between groups.

    Args:
        metrics_by_group: {group_value: {metric: value}}
        metric_name: Which metric to compare (e.g., "auroc", "recall")

    Returns:
        {
            'group_0': value,
            'group_1': value,
            'absolute_gap': |value_0 - value_1|,
            'relative_gap': (max - min) / min,
            'favored_group': group with higher value,
        }
    """
    if metric_name not in list(metrics_by_group.values())[0]:
        raise ValueError(f"Metric '{metric_name}' not found in group metrics")

    values_by_group = {
        group: metrics[metric_name]
        for group, metrics in metrics_by_group.items()
    }

    values = list(values_by_group.values())
    groups = list(values_by_group.keys())

    result = {**values_by_group}
    result["absolute_gap"] = float(abs(values[0] - values[1]))
    result["relative_gap"] = float(
        (max(values) - min(values)) / min(values) if min(values) != 0 else np.nan
    )
    result["favored_group"] = groups[np.argmax(values)]

    return result


def aggregate_fairness_across_bootstrap(
    evaluator,
    group_column: str = "Gender",
) -> Dict[str, Dict]:
    """
    Compute per-group metrics for each bootstrap iteration and aggregate.

    Requires that evaluator.bootstrap_metrics[i] contains "per_group" key.

    Args:
        evaluator: BootstrapEvaluator instance (after evaluate_iteration calls)
        group_column: Column used for grouping (must match evaluation calls)

    Returns:
        {
            metric_name: {
                group_value: {mean, std, median, min, max},
            }
        }
    """
    # Collect per-group metrics across all iterations
    per_group_data = {}

    for metric_iter in evaluator.bootstrap_metrics:
        if "per_group" not in metric_iter:
            logger.warning("Per-group metrics not found. Run evaluate_iteration() with compute_per_group=True")
            return {}

        per_group = metric_iter["per_group"]

        for group_val, group_metrics in per_group.items():
            if group_val not in per_group_data:
                per_group_data[group_val] = {}

            for metric_name, value in group_metrics.items():
                if metric_name not in per_group_data[group_val]:
                    per_group_data[group_val][metric_name] = []

                if not np.isnan(value):
                    per_group_data[group_val][metric_name].append(value)

    # Aggregate across iterations
    aggregated = {}

    for metric_name in ["auroc", "accuracy", "precision", "recall", "f1"]:
        aggregated[metric_name] = {}

        for group_val in sorted(per_group_data.keys()):
            if metric_name in per_group_data[group_val]:
                values = per_group_data[group_val][metric_name]

                if values:
                    aggregated[metric_name][group_val] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "median": float(np.median(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "n_iterations": len(values),
                    }

    return aggregated


def print_fairness_report(
    agg_fairness: Dict[str, Dict],
    overall_metrics: Dict,
) -> None:
    """
    Print a nicely formatted fairness report.

    Args:
        agg_fairness: Output from aggregate_fairness_across_bootstrap()
        overall_metrics: Output from evaluator.aggregate_bootstrap_metrics()
    """
    print("\n" + "=" * 70)
    print("FAIRNESS EVALUATION REPORT")
    print("=" * 70)

    # Overall metrics
    print("\nOVERALL METRICS (Across All Bootstrap Samples):")
    print("-" * 70)

    for metric_name in ["auroc", "accuracy", "recall", "f1"]:
        if metric_name in overall_metrics:
            stat = overall_metrics[metric_name]
            print(
                f"{metric_name:10s}: {stat['mean']:7.4f} ± {stat['std']:7.4f} "
                f"(median: {stat['median']:.4f}, range: [{stat['min']:.4f}, {stat['max']:.4f}])"
            )

    # Per-group metrics
    if agg_fairness:
        print("\nPER-GROUP METRICS (By Gender):")
        print("-" * 70)

        for metric_name in ["auroc", "accuracy", "recall", "f1"]:
            if metric_name in agg_fairness and agg_fairness[metric_name]:
                print(f"\n{metric_name.upper()}:")

                for group_val in sorted(agg_fairness[metric_name].keys()):
                    stat = agg_fairness[metric_name][group_val]
                    print(
                        f"  Group {group_val}: {stat['mean']:7.4f} ± {stat['std']:7.4f} "
                        f"(median: {stat['median']:.4f})"
                    )

                # Fairness gap
                gap = evaluate_fairness_gap(
                    {
                        group: agg_fairness[metric_name][group]
                        for group in agg_fairness[metric_name].keys()
                    },
                    metric_name,
                )

                print(
                    f"  Absolute Gap: {gap['absolute_gap']:.4f} "
                    f"(Relative: {gap['relative_gap']:.1%})"
                )
                print(f"  Favored Group: {gap['favored_group']}")

    print("\n" + "=" * 70)


def fairness_summary_table(
    agg_fairness: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Return per-group metrics as a table for export.

    Args:
        agg_fairness: Output from aggregate_fairness_across_bootstrap()

    Returns:
        DataFrame with one row per metric-group combination
    """
    rows = []

    for metric_name in ["auroc", "accuracy", "precision", "recall", "f1"]:
        if metric_name in agg_fairness:
            for group_val in sorted(agg_fairness[metric_name].keys()):
                stat = agg_fairness[metric_name][group_val]
                rows.append({
                    "metric": metric_name,
                    "group": group_val,
                    "mean": stat["mean"],
                    "std": stat["std"],
                    "median": stat["median"],
                    "min": stat["min"],
                    "max": stat["max"],
                    "n_iterations": stat["n_iterations"],
                })

    return pd.DataFrame(rows)


# Example usage
if __name__ == "__main__":
    """
    Example: Run fairness evaluation on GLM with bootstrap
    """
    import logging
    from pathlib import Path

    from config import Config, TrainingConfig, BootstrapConfig
    from data_loader import (
        load_physionet_files,
        split_patients_by_status,
        get_rows_for_patients,
    )
    from bootstrap import BootstrapResampler
    from models import LogisticGLM
    from training import BootstrapEvaluator

    logging.basicConfig(level=logging.INFO)

    # Setup
    cfg = Config(
        training=TrainingConfig(n_patients=50),
        bootstrap=BootstrapConfig(n_iterations=10, bootstrap_sample_size=30),
    )

    # Load data
    df = load_physionet_files(cfg.data_dir)
    train_pids, bootstrap_pids = split_patients_by_status(df, cfg.training.n_patients)
    train_df = get_rows_for_patients(df, train_pids)

    # Bootstrap
    resampler = BootstrapResampler(
        bootstrap_pool_patient_ids=bootstrap_pids,
        full_df=df,
        n_iterations=cfg.bootstrap.n_iterations,
        bootstrap_sample_size=cfg.bootstrap.bootstrap_sample_size,
    )

    # Train and evaluate with per-group metrics
    model = LogisticGLM()
    evaluator = BootstrapEvaluator(model, train_df, label_column="SepsisLabel")

    print("\nEvaluating with per-group metrics...")
    for i in range(cfg.bootstrap.n_iterations):
        pids, bootstrap_df = resampler.generate_iteration(i)
        evaluator.evaluate_iteration(
            bootstrap_df,
            i,
            compute_per_group=True,
            group_column="Gender",
        )

    # Aggregate
    overall = evaluator.aggregate_bootstrap_metrics()
    fairness = aggregate_fairness_across_bootstrap(evaluator, group_column="Gender")

    # Print report
    print_fairness_report(fairness, overall)

    # Table export
    table = fairness_summary_table(fairness)
    print("\nFairness Metrics Table:")
    print(table.to_string(index=False))
