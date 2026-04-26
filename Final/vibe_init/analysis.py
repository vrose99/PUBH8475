"""
Post-hoc fairness analysis for the time-series evaluation.

After run_timeseries_evaluation() produces a results DataFrame this module:
  1. Computes delta vs. baseline (no mitigation) for each fairness metric
  2. Ranks mitigation strategies by fairness improvement and utility cost
  3. Produces supporting figures

Entry point: run_analysis(results, cfg) → saves tables + figures, returns summary dict.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import Config

logger = logging.getLogger(__name__)

MITIGATION_LABELS = {
    "none":                   "Baseline",
    "reweighting":            "Reweighting",
    "smote":                  "SMOTE",
    "fairness_penalty":       "Fairness Penalty",
    "threshold_optimization": "Threshold Optimization",
}

_TS_FAIRNESS_METRICS = [
    "detection_lead_gap_hours",
    "missed_rate_gap",
    "alarm_fatigue_rate_gap",
    "overall_physionet_utility",
]

_TS_PERFORMANCE_METRIC = "overall_physionet_utility"
_TS_FAIRNESS_METRIC    = "detection_lead_gap_hours"


def _save(fig: plt.Figure, path: Path, dpi: int = 150):
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    logger.info("Saved %s", path)


def compute_mitigation_delta(
    results: pd.DataFrame,
    metrics: Optional[list] = None,
) -> pd.DataFrame:
    """
    For every (model, mitigation) row compute delta vs. baseline ("none").
    Positive delta means the metric increased vs. baseline.
    Returns DataFrame with added <metric>_delta columns.
    """
    if metrics is None:
        metrics = _TS_FAIRNESS_METRICS
    metrics = [m for m in metrics if m in results.columns]

    baseline = (
        results[results["mitigation"] == "none"]
        [["model"] + metrics]
        .rename(columns={m: f"{m}_baseline" for m in metrics})
    )

    merged = results.merge(baseline, on="model", how="left")
    for m in metrics:
        merged[f"{m}_delta"] = merged[m] - merged[f"{m}_baseline"]

    return merged


def mitigation_ranking(
    results: pd.DataFrame,
    cfg: Config,
    fairness_metric: str = _TS_FAIRNESS_METRIC,
    performance_metric: str = _TS_PERFORMANCE_METRIC,
) -> pd.DataFrame:
    """
    Rank mitigation strategies by mean fairness gap (abs) and utility score.
    Returns DataFrame sorted by abs(fairness_metric).
    """
    available_metrics = [m for m in [fairness_metric, performance_metric] if m in results.columns]
    if not available_metrics:
        return pd.DataFrame()

    agg = (
        results
        .groupby("mitigation")[available_metrics]
        .mean()
        .reset_index()
    )
    if fairness_metric in agg.columns:
        agg[f"abs_{fairness_metric}"] = agg[fairness_metric].abs()
        agg.sort_values(f"abs_{fairness_metric}", inplace=True)

    agg["mitigation_label"] = agg["mitigation"].map(lambda m: MITIGATION_LABELS.get(m, m))

    out = cfg.output_dir / "tables" / "analysis_mitigation_ranking.csv"
    agg.round(4).to_csv(out, index=False)
    logger.info("Saved %s", out)
    return agg


def figure_fairness_utility_tradeoff(
    results: pd.DataFrame,
    cfg: Config,
):
    """
    Scatter: x = abs(detection_lead_gap_hours), y = overall_physionet_utility.
    Each point = (model, mitigation). Ideal = bottom-left (fair + high utility).
    """
    x_col = "detection_lead_gap_hours"
    y_col = "overall_physionet_utility"

    if x_col not in results.columns or y_col not in results.columns:
        logger.warning("Tradeoff columns not found — skipping figure")
        return

    plot_df = results.copy()
    plot_df["abs_gap"] = plot_df[x_col].abs()
    plot_df["Mitigation"] = plot_df["mitigation"].map(lambda m: MITIGATION_LABELS.get(m, m))

    palette = sns.color_palette("Set2", n_colors=plot_df["Mitigation"].nunique())
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="abs_gap",
        y=y_col,
        hue="Mitigation",
        style="Mitigation",
        palette=palette,
        ax=ax,
        s=80,
        alpha=0.85,
    )
    ax.set_xlabel("|Detection Lead Gap| hours  (lower = fairer)")
    ax.set_ylabel("PhysioNet Utility Score  (higher = better)")
    ax.set_title("Fairness–Utility Trade-off by Mitigation Strategy")
    ax.legend(title="Mitigation", bbox_to_anchor=(1.01, 1), loc="upper left")

    out = cfg.output_dir / "figures" / "analysis_fairness_utility_tradeoff.png"
    _save(fig, out)


def figure_delta_bars(
    results_delta: pd.DataFrame,
    cfg: Config,
):
    """
    Grouped bar chart of metric Δ vs. baseline, per mitigation strategy.
    One bar group per metric, one bar per mitigation (excluding baseline).
    """
    delta_cols = [c for c in results_delta.columns if c.endswith("_delta")]
    if not delta_cols:
        logger.warning("No delta columns found — skipping delta bars figure")
        return

    non_base = results_delta[results_delta["mitigation"] != "none"].copy()
    if non_base.empty:
        return

    non_base["Mitigation"] = non_base["mitigation"].map(lambda m: MITIGATION_LABELS.get(m, m))
    agg = non_base.groupby("Mitigation")[delta_cols].mean()
    agg.columns = [c.replace("_delta", "").replace("_", " ") for c in agg.columns]

    fig, ax = plt.subplots(figsize=(max(8, len(agg.columns) * 2), 5))
    agg.T.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Metric Δ vs. Baseline — positive = increased, negative = decreased")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Δ vs. baseline")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Mitigation", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()

    out = cfg.output_dir / "figures" / "analysis_delta_bars.png"
    _save(fig, out)


def run_analysis(results: pd.DataFrame, cfg: Config) -> dict:
    """
    Run the full post-hoc analysis suite for the time-series evaluation.
    Returns a summary dict with key findings.
    """
    logger.info("Running post-hoc fairness analysis ...")

    results_delta = compute_mitigation_delta(results)
    ranking_df    = mitigation_ranking(results, cfg)

    figure_fairness_utility_tradeoff(results, cfg)
    figure_delta_bars(results_delta, cfg)

    baseline_row = ranking_df[ranking_df["mitigation"] == "none"] if not ranking_df.empty else pd.DataFrame()
    best_row     = ranking_df[ranking_df["mitigation"] != "none"].iloc[:1] if not ranking_df.empty else pd.DataFrame()

    f_col = f"abs_{_TS_FAIRNESS_METRIC}"
    best_mit   = best_row.iloc[0]["mitigation"] if not best_row.empty else "n/a"
    best_gap   = float(best_row.iloc[0][f_col]) if not best_row.empty and f_col in best_row.columns else float("nan")
    base_gap   = float(baseline_row.iloc[0][f_col]) if not baseline_row.empty and f_col in baseline_row.columns else float("nan")

    summary = {
        "best_mitigation": best_mit,
        "best_gap":        round(best_gap,  4) if np.isfinite(best_gap)  else None,
        "baseline_gap":    round(base_gap,  4) if np.isfinite(base_gap)  else None,
        "ranking":         ranking_df.to_dict("records") if not ranking_df.empty else [],
    }

    logger.info(
        "Analysis complete. Best mitigation: %s (gap |%.3f| vs baseline |%.3f|)",
        best_mit, best_gap or float("nan"), base_gap or float("nan"),
    )
    return summary
