"""
Post-hoc fairness analysis.

After run_evaluation() produces a results DataFrame this module answers:
  1. How much did each mitigation strategy improve (or worsen) fairness?
  2. Which (dataset, model) cells exhibit the "illusion of fairness"?
  3. What is the clinical-utility cost of the fairness gap?
  4. Which strategy is most effective, averaged across conditions?

Entry point: run_analysis(results, cfg) → saves tables + figures and returns
a summary dict.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from config import Config

logger = logging.getLogger(__name__)

_PALETTE = "RdYlGn_r"

DATASET_LABELS = {
    "D0":  "Original",
    "D1A": "Row-rm (F↓)", "D1B": "Row-rm (M↓)",
    "D2A": "MAR (F)",      "D2B": "MAR (M)",
    "D3A": "Noise (F)",    "D3B": "Noise (M)",
}

MITIGATION_LABELS = {
    "none":             "Baseline",
    "reweighting":      "Reweighting",
    "smote":            "SMOTE",
    "fairness_penalty": "Fairness Penalty",
    "robust_model":     "Robust Model",
}


def _save(fig: plt.Figure, path: Path, dpi: int = 150):
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    logger.info("Saved %s", path)


# ── 1. Mitigation delta table ─────────────────────────────────────────────────

def compute_mitigation_delta(
    results: pd.DataFrame,
    metrics: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    For every (dataset, model, mitigation) row compute delta vs. baseline ("none").

    Positive delta means the metric *increased*; for fairness-gap metrics
    (where smaller is better) a negative delta means improvement.

    Returns a DataFrame with added <metric>_delta columns.
    """
    if metrics is None:
        metrics = [
            "equalized_odds_diff", "auroc_gap", "tpr_gap", "fpr_gap",
            "ppv_gap", "sufficiency_diff", "demographic_parity_diff",
            "worst_group_auroc", "overall_auroc",
            "clinical_utility_gap",
        ]
    metrics = [m for m in metrics if m in results.columns]

    baseline = (
        results[results["mitigation"] == "none"]
        [["dataset_id", "model"] + metrics]
        .rename(columns={m: f"{m}_baseline" for m in metrics})
    )

    merged = results.merge(baseline, on=["dataset_id", "model"], how="left")
    for m in metrics:
        merged[f"{m}_delta"] = merged[m] - merged[f"{m}_baseline"]

    return merged


# ── 2. Illusion-of-fairness report ───────────────────────────────────────────

def illusion_of_fairness_report(results: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Flag rows where:
      - equalized_odds_diff  < 0.05   (small apparent gap)
      - worst_group_auroc    < 0.55   (near-random on worst group)

    These cells demonstrate that a "fair" aggregate can mask clinical failure.
    """
    if "worst_group_auroc" not in results.columns:
        logger.warning("worst_group_auroc not in results — skipping illusion report")
        return pd.DataFrame()

    flagged = results[
        (results["equalized_odds_diff"] < 0.05) &
        (results["worst_group_auroc"] < 0.55)
    ][["dataset_id", "model", "mitigation",
       "equalized_odds_diff", "worst_group_auroc",
       "overall_auroc"]].copy()

    out = cfg.output_dir / "tables" / "analysis_illusion_of_fairness.csv"
    flagged.round(4).to_csv(out, index=False)
    logger.info("Illusion-of-fairness cells: %d  →  %s", len(flagged), out)
    return flagged


# ── 3. Mitigation ranking table ───────────────────────────────────────────────

def mitigation_ranking(
    results: pd.DataFrame,
    cfg: Config,
    fairness_metric: str = "equalized_odds_diff",
    performance_metric: str = "overall_auroc",
) -> pd.DataFrame:
    """
    Rank mitigation strategies by:
      - mean fairness metric (lower is better for gap metrics)
      - mean performance metric (higher is better)
      - Pareto-efficiency flag (not dominated on both axes)

    Returns a DataFrame sorted by fairness metric.
    """
    agg = (
        results
        .groupby("mitigation")[[fairness_metric, performance_metric]]
        .mean()
        .reset_index()
        .rename(columns={
            fairness_metric:   f"mean_{fairness_metric}",
            performance_metric: f"mean_{performance_metric}",
        })
    )
    agg.sort_values(f"mean_{fairness_metric}", inplace=True)
    agg["mitigation_label"] = agg["mitigation"].map(
        lambda m: MITIGATION_LABELS.get(m, m)
    )

    out = cfg.output_dir / "tables" / "analysis_mitigation_ranking.csv"
    agg.round(4).to_csv(out, index=False)
    logger.info("Saved %s", out)
    return agg


# ── 4. Clinical utility cost of the gap ───────────────────────────────────────

def clinical_utility_gap_summary(
    results: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    For each (model, mitigation) pair averaged across datasets, report:
      - female_clinical_utility
      - male_clinical_utility
      - clinical_utility_gap  (female − male)

    This translates abstract fairness numbers into patient-care terms.
    """
    keep = [
        "model", "mitigation",
        "female_clinical_utility", "male_clinical_utility", "clinical_utility_gap",
    ]
    keep = [c for c in keep if c in results.columns]
    if len(keep) < 3:
        return pd.DataFrame()

    agg = results[keep].groupby(["model", "mitigation"]).mean().reset_index()

    out = cfg.output_dir / "tables" / "analysis_clinical_utility.csv"
    agg.round(4).to_csv(out, index=False)
    logger.info("Saved %s", out)
    return agg


# ── 5. Figures ────────────────────────────────────────────────────────────────

def figure_mitigation_delta_heatmap(
    results_delta: pd.DataFrame,
    cfg: Config,
    delta_metric: str = "equalized_odds_diff_delta",
):
    """
    Heatmap: rows = mitigation, cols = dataset, colour = Δ fairness gap.
    Averaged across models.  Green = improvement (gap decreased), red = worse.
    """
    if delta_metric not in results_delta.columns:
        logger.warning("%s not found — skipping delta heatmap", delta_metric)
        return

    subset = results_delta[results_delta["mitigation"] != "none"]
    pivot = subset.pivot_table(
        index="mitigation",
        columns="dataset_id",
        values=delta_metric,
        aggfunc="mean",
    )
    pivot.index = [MITIGATION_LABELS.get(i, i) for i in pivot.index]
    pivot.columns = [DATASET_LABELS.get(c, c) for c in pivot.columns]

    # For gap metrics: negative delta = improvement (green), positive = worse (red)
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.4), 4))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn_r",   # red = increased gap (worse), green = reduced gap (better)
        center=0,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": f"Δ {delta_metric.replace('_delta','').replace('_',' ')}"},
    )
    ax.set_title(
        f"Mitigation effect on {delta_metric.replace('_delta','').replace('_',' ')} "
        "(negative = improvement)"
    )
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    base_metric = delta_metric.replace("_delta", "")
    out = cfg.output_dir / "figures" / f"analysis_delta_heatmap_{base_metric}.png"
    _save(fig, out)


def figure_fairness_performance_tradeoff(
    results: pd.DataFrame,
    cfg: Config,
    fairness_metric: str = "equalized_odds_diff",
    performance_metric: str = "overall_auroc",
):
    """
    Scatter: x = fairness gap, y = AUROC, one point per (dataset, model, mitigation).
    Colour-codes mitigation strategy.  Ideal = bottom-right (fair + accurate).
    """
    plot_df = results.copy()
    plot_df["mitigation_label"] = plot_df["mitigation"].map(
        lambda m: MITIGATION_LABELS.get(m, m)
    )

    if fairness_metric not in plot_df.columns or performance_metric not in plot_df.columns:
        return

    palette = sns.color_palette("Set2", n_colors=plot_df["mitigation_label"].nunique())
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x=fairness_metric,
        y=performance_metric,
        hue="mitigation_label",
        style="mitigation_label",
        palette=palette,
        ax=ax,
        s=60,
        alpha=0.8,
    )
    ax.set_xlabel(fairness_metric.replace("_", " ").title() + " (lower = fairer)")
    ax.set_ylabel(performance_metric.replace("_", " ").title() + " (higher = better)")
    ax.set_title("Fairness–Performance trade-off by mitigation strategy")
    ax.legend(title="Mitigation", bbox_to_anchor=(1.01, 1), loc="upper left")

    out = cfg.output_dir / "figures" / f"analysis_tradeoff_{fairness_metric}.png"
    _save(fig, out)


def figure_clinical_utility_comparison(
    results: pd.DataFrame,
    cfg: Config,
):
    """
    Grouped bar: female vs. male clinical utility per (model, mitigation),
    averaged across datasets.  Negative values indicate net harm (more missed
    sepsis than benefit from correct detections).
    """
    needed = {"female_clinical_utility", "male_clinical_utility"}
    if not needed.issubset(results.columns):
        logger.warning("clinical_utility columns missing — skipping figure")
        return

    agg = (
        results
        .groupby(["model", "mitigation"])[list(needed)]
        .mean()
        .reset_index()
    )
    agg["mitigation_label"] = agg["mitigation"].map(
        lambda m: MITIGATION_LABELS.get(m, m)
    )
    melted = agg.melt(
        id_vars=["model", "mitigation_label"],
        value_vars=["female_clinical_utility", "male_clinical_utility"],
        var_name="group",
        value_name="clinical_utility",
    )
    melted["group"] = melted["group"].map({
        "female_clinical_utility": "Female",
        "male_clinical_utility":   "Male",
    })

    models = agg["model"].unique()
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = melted[melted["model"] == model]
        sns.barplot(
            data=sub,
            x="mitigation_label",
            y="clinical_utility",
            hue="group",
            palette={"Female": "#E07B78", "Male": "#5B8DB8"},
            ax=ax,
            edgecolor="white",
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(model.replace("_", " ").title())
        ax.set_xlabel("Mitigation")
        ax.set_ylabel("Utility score / patient" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=30)
        ax.get_legend().remove() if ax is not axes[-1] else ax.legend(title="Group")

    fig.suptitle("Clinical utility score by group and mitigation", y=1.02)
    out = cfg.output_dir / "figures" / "analysis_clinical_utility.png"
    _save(fig, out)


def figure_worst_group_auroc(
    results: pd.DataFrame,
    cfg: Config,
):
    """
    Line chart: worst-group AUROC across dataset variants for each model.
    Highlights whether mitigation lifts the floor for the worst-off group.
    """
    if "worst_group_auroc" not in results.columns:
        return

    models = results["model"].unique()
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    palette = sns.color_palette("Set2", n_colors=results["mitigation"].nunique())
    mit_colors = dict(zip(results["mitigation"].unique(), palette))

    for ax, model in zip(axes, models):
        sub = results[results["model"] == model]
        for mit, grp in sub.groupby("mitigation"):
            ordered = grp.sort_values("dataset_id")
            ax.plot(
                ordered["dataset_id"],
                ordered["worst_group_auroc"],
                marker="o",
                label=MITIGATION_LABELS.get(mit, mit),
                color=mit_colors.get(mit),
            )
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(model.replace("_", " ").title())
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Worst-group AUROC" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=30)

    axes[-1].legend(title="Mitigation", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.suptitle("Worst-group AUROC — guarding against illusion of fairness", y=1.02)
    out = cfg.output_dir / "figures" / "analysis_worst_group_auroc.png"
    _save(fig, out)


# ── Entry point ───────────────────────────────────────────────────────────────

def run_analysis(
    results: pd.DataFrame,
    cfg: Config,
) -> dict:
    """
    Run the full post-hoc analysis suite.

    Parameters
    ----------
    results : DataFrame from evaluation.run_evaluation()
    cfg     : project Config

    Returns a summary dict with key findings.
    """
    logger.info("Running post-hoc fairness analysis ...")

    results_delta = compute_mitigation_delta(results)

    illusion_df = illusion_of_fairness_report(results, cfg)
    ranking_df  = mitigation_ranking(results, cfg)
    utility_df  = clinical_utility_gap_summary(results, cfg)

    figure_mitigation_delta_heatmap(results_delta, cfg, "equalized_odds_diff_delta")
    figure_mitigation_delta_heatmap(results_delta, cfg, "auroc_gap_delta")
    figure_fairness_performance_tradeoff(results, cfg)
    figure_clinical_utility_comparison(results, cfg)
    figure_worst_group_auroc(results, cfg)

    # Compute top-line summary numbers for report.py
    baseline = results[results["mitigation"] == "none"]
    best_mit = ranking_df.iloc[0]["mitigation"] if not ranking_df.empty else "n/a"
    best_gap = (
        ranking_df.iloc[0][f"mean_equalized_odds_diff"]
        if not ranking_df.empty and "mean_equalized_odds_diff" in ranking_df.columns
        else np.nan
    )
    baseline_gap = (
        ranking_df[ranking_df["mitigation"] == "none"][f"mean_equalized_odds_diff"].values[0]
        if "none" in ranking_df["mitigation"].values and "mean_equalized_odds_diff" in ranking_df.columns
        else np.nan
    )

    summary = {
        "n_illusion_cells":  len(illusion_df),
        "best_mitigation":   best_mit,
        "best_eod_gap":      round(float(best_gap),   4) if np.isfinite(best_gap)   else None,
        "baseline_eod_gap":  round(float(baseline_gap), 4) if np.isfinite(baseline_gap) else None,
        "ranking":           ranking_df.to_dict("records"),
    }

    logger.info(
        "Analysis complete. Best mitigation: %s (EO-diff %.3f vs baseline %.3f). "
        "Illusion-of-fairness cells: %d",
        best_mit, best_gap or float("nan"), baseline_gap or float("nan"), len(illusion_df),
    )
    return summary
