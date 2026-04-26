"""
Figures and tables for the time-series fairness analysis.

All functions accept the results DataFrame produced by
evaluation_timeseries.run_timeseries_evaluation() and save output to
cfg.output_dir/{figures,tables}/.

Key outputs
-----------
  figure_1_utility_bars()       — PhysioNet utility score per model × mitigation
  figure_2_detection_lead()     — detection lead hours per group, before/after mitigation
  figure_3_fairness_metrics()   — 3 gap metrics per model × mitigation
  figure_4_mitigation_delta()   — improvement vs. baseline for each fairness metric
  table_1_summary()             — full results CSV + LaTeX
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
    "none":             "Baseline",
    "reweighting":      "Reweighting",
    "smote":            "SMOTE",
    "fairness_penalty": "Fairness Penalty",
}

_FEMALE_COLOR = "#E07B78"
_MALE_COLOR   = "#5B8DB8"


def _save(fig: plt.Figure, path: Path, dpi: int = 150):
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    logger.info("Saved %s", path)


def _mit_label(m: str) -> str:
    return MITIGATION_LABELS.get(m, m)


# ── Figure 1: PhysioNet utility score ────────────────────────────────────────

def figure_1_utility_bars(results: pd.DataFrame, cfg: Config):
    """
    Grouped bar chart: PhysioNet 2019 utility score per model,
    one bar group per mitigation strategy.
    """
    col = "overall_physionet_utility"
    if col not in results.columns:
        logger.warning("Column %s not found — skipping figure 1", col)
        return

    plot_df = results[["model", "mitigation", col]].copy()
    plot_df["Mitigation"] = plot_df["mitigation"].map(_mit_label)

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = sns.color_palette("Set2", n_colors=plot_df["Mitigation"].nunique())
    sns.barplot(
        data=plot_df,
        x="model",
        y=col,
        hue="Mitigation",
        palette=palette,
        ax=ax,
        edgecolor="white",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", label="Inaction baseline")
    ax.set_title("PhysioNet 2019 Utility Score  [1 = perfect, 0 = inaction, <0 = harmful]")
    ax.set_xlabel("Model")
    ax.set_ylabel("Normalised utility score")
    ax.legend(title="Mitigation", bbox_to_anchor=(1.01, 1), loc="upper left")

    out = cfg.output_dir / "figures" / "fig1_utility_score.png"
    _save(fig, out)
    return out


# ── Figure 2: Per-group detection lead hours ─────────────────────────────────

def figure_2_detection_lead(results: pd.DataFrame, cfg: Config):
    """
    Side-by-side bars: PhysioNet utility for female vs. male,
    one subplot per model.  Fairness-aligned metric focusing on actual reward.
    """
    f_col = "female_physionet_utility"
    m_col = "male_physionet_utility"
    if f_col not in results.columns or m_col not in results.columns:
        logger.warning("PhysioNet utility columns not found — skipping figure 2")
        return

    models = results["model"].unique()
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = results[results["model"] == model].copy()
        sub["Mitigation"] = sub["mitigation"].map(_mit_label)

        melted = pd.melt(
            sub,
            id_vars=["Mitigation"],
            value_vars=[f_col, m_col],
            var_name="group",
            value_name="utility",
        )
        melted["group"] = melted["group"].map(
            {f_col: "Female", m_col: "Male"}
        )

        sns.barplot(
            data=melted,
            x="Mitigation",
            y="utility",
            hue="group",
            palette={"Female": _FEMALE_COLOR, "Male": _MALE_COLOR},
            ax=ax,
            edgecolor="white",
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(model.replace("_", " ").title())
        ax.set_xlabel("Mitigation")
        ax.set_ylabel("PhysioNet utility" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=20)
        if ax is not axes[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Group")

    fig.suptitle(
        "Detection Lead Hours by Group — positive = early warning before onset",
        y=1.02,
    )
    out = cfg.output_dir / "figures" / "fig2_detection_lead_hours.png"
    _save(fig, out)
    return out


# ── Figure 3: Three fairness gap metrics ─────────────────────────────────────

def figure_3_fairness_metrics(results: pd.DataFrame, cfg: Config):
    """
    Three-panel bar chart showing each fairness gap metric per model × mitigation.
      Panel 1: detection_lead_gap_hours  (ideal = 0)
      Panel 2: missed_rate_gap           (ideal = 0)
      Panel 3: alarm_fatigue_rate_gap    (ideal = 0)
    """
    metrics = [
        ("detection_lead_gap_hours", "Detection Lead Gap (hours)\n[Female − Male, ideal = 0]"),
        ("missed_rate_gap",          "Missed Rate Gap\n[Female − Male, ideal = 0]"),
        ("alarm_fatigue_rate_gap",   "Alarm Fatigue Rate Gap\n[Female − Male, ideal = 0]"),
    ]

    available = [(col, lbl) for col, lbl in metrics if col in results.columns]
    if not available:
        logger.warning("No fairness gap columns found — skipping figure 3")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    palette = sns.color_palette("Set2", n_colors=results["model"].nunique())

    for ax, (col, ylabel) in zip(axes, available):
        plot_df = results[["model", "mitigation", col]].copy()
        plot_df["Mitigation"] = plot_df["mitigation"].map(_mit_label)

        sns.barplot(
            data=plot_df,
            x="Mitigation",
            y=col,
            hue="model",
            palette=palette,
            ax=ax,
            edgecolor="white",
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(ylabel, fontsize=9)
        ax.set_xlabel("Mitigation")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20)
        if ax is not axes[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")

    fig.suptitle("Fairness Gap Metrics — lower absolute value is fairer", y=1.02)
    out = cfg.output_dir / "figures" / "fig3_fairness_gap_metrics.png"
    _save(fig, out)
    return out


# ── Figure 4: Mitigation delta heatmap ───────────────────────────────────────

def figure_4_mitigation_delta(results: pd.DataFrame, cfg: Config):
    """
    Heatmap of improvement vs. baseline (none) for each fairness metric.
    Rows = (model, mitigation), columns = metrics.
    Green = improvement, red = worsened.
    """
    gap_cols = [
        c for c in ["detection_lead_gap_hours", "missed_rate_gap",
                    "alarm_fatigue_rate_gap", "overall_physionet_utility"]
        if c in results.columns
    ]
    if not gap_cols:
        return

    non_base = results[results["mitigation"] != "none"].copy()
    if non_base.empty:
        logger.info("No mitigations to compare against baseline — skipping figure 4")
        return

    # Compute delta: average each metric per (model, mitigation) pair across datasets
    agg = non_base.groupby(["model", "mitigation"])[gap_cols].mean()
    baseline_agg = results[results["mitigation"] == "none"].groupby("model")[gap_cols].mean()

    rows = []
    for (model, mitigation), row in agg.iterrows():
        if model not in baseline_agg.index:
            continue
        base_row = baseline_agg.loc[model]
        delta = {c: row[c] - base_row[c] for c in gap_cols}
        delta["model"] = model
        delta["mitigation"] = _mit_label(mitigation)
        rows.append(delta)

    if not rows:
        return

    delta_df = pd.DataFrame(rows).set_index(["model", "mitigation"])[gap_cols]
    delta_df.columns = [c.replace("_", " ") for c in delta_df.columns]

    # Ensure all values are numeric (handle any remaining Series)
    delta_df = delta_df.astype(float)
    vals = delta_df.values
    vmax_abs = max(abs(float(np.nanmin(vals))), abs(float(np.nanmax(vals))), 0.01)

    fig, ax = plt.subplots(figsize=(max(7, len(delta_df.columns) * 1.5), max(4, len(delta_df) * 0.6)))
    sns.heatmap(
        delta_df,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        vmin=-vmax_abs,
        vmax=vmax_abs,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": "Δ vs. baseline"},
    )
    ax.set_title("Mitigation Δ vs. Baseline  [green = improvement]")
    ax.set_xlabel("Metric")
    ax.set_ylabel("(Model, Mitigation)")
    ax.tick_params(axis="x", rotation=20)

    out = cfg.output_dir / "figures" / "fig4_mitigation_delta.png"
    _save(fig, out)
    return out


# ── Tables ────────────────────────────────────────────────────────────────────

def table_1_summary(results: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Export a summary CSV (and LaTeX) with all metrics per (model, mitigation)."""
    priority_cols = [
        "dataset_id", "model", "mitigation",
        "overall_physionet_utility",
        "female_physionet_utility", "male_physionet_utility",
        "physionet_utility_gap",
        "pct_detected_at_all_gap", "pct_in_optimal_window_gap",
        "missed_rate_gap", "alarm_fatigue_rate_gap",
        "disparate_impact", "equal_opportunity_diff", "equalized_odds_diff",
        "demographic_parity_diff", "sufficiency_diff",
    ]
    keep = [c for c in priority_cols if c in results.columns]
    remaining = [c for c in results.columns if c not in keep]
    tbl = results[keep + remaining].round(4)

    csv_path = cfg.output_dir / "tables" / "table1_summary.csv"
    tbl.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    try:
        latex_path = cfg.output_dir / "tables" / "table1_summary.tex"
        tbl[keep].to_latex(latex_path, index=False, float_format="%.3f")
        logger.info("Saved %s", latex_path)
    except Exception:
        pass

    return tbl


def table_2_fairness_pivot(results: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Pivot table: rows = (model, mitigation), columns = three fairness gap metrics
    + utility score. Suitable for a publication table.
    """
    gap_cols = [
        c for c in [
            "overall_physionet_utility",
            "detection_lead_gap_hours",
            "missed_rate_gap",
            "alarm_fatigue_rate_gap",
        ]
        if c in results.columns
    ]
    if not gap_cols:
        return pd.DataFrame()

    pivot = (
        results.groupby(["model", "mitigation"])[gap_cols]
        .mean()
        .round(4)
    )
    csv_path = cfg.output_dir / "tables" / "table2_fairness_pivot.csv"
    pivot.to_csv(csv_path)
    logger.info("Saved %s", csv_path)
    return pivot


# ── Convenience: render all standard figures ──────────────────────────────────

def render_all(results: pd.DataFrame, cfg: Config):
    """Render the full standard figure set for the time-series analysis."""
    logger.info("Rendering figures ...")

    if results.empty or "mitigation" not in results.columns:
        logger.warning("Results are empty or malformed — skipping figure generation.")
        return

    figure_1_utility_bars(results, cfg)
    figure_2_detection_lead(results, cfg)
    figure_3_fairness_metrics(results, cfg)
    figure_4_mitigation_delta(results, cfg)
    table_1_summary(results, cfg)
    table_2_fairness_pivot(results, cfg)

    logger.info("All figures and tables saved to %s", cfg.output_dir)
