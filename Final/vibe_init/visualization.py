"""
Figures and tables for the fairness analysis.

All functions accept the results DataFrame produced by evaluation.run_evaluation()
and save output to cfg.output_dir/{figures,tables}/.

Key outputs
-----------
  figure_1_fairness_heatmap()     — heatmap per dataset, rows=models, cols=mitigations
  figure_2_group_barplot()        — per-group metric bars for baseline (no mitigation)
  figure_3_sweep_lines()          — parameter sweep curves (appendix)
  table_1_summary()               — LaTeX / CSV summary table
  table_2_gap_heatmap()           — gap metric pivot, publication-ready
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

DATASET_LABELS = {
    "D0":  "Original",
    "D1A": "Row removal\n(women ↓)",
    "D1B": "Row removal\n(men ↓)",
    "D2A": "MAR\n(women)",
    "D2B": "MAR\n(men)",
    "D3A": "Noise\n(women)",
    "D3B": "Noise\n(men)",
}

MITIGATION_LABELS = {
    "none":            "Baseline",
    "reweighting":     "Reweighting",
    "smote":           "SMOTE",
    "fairness_penalty":"Fairness\nPenalty",
    "robust_model":    "Robust\nModel",
}

_PALETTE = "RdYlGn_r"   # red=bad (large gap), green=good (small gap)


def _save(fig: plt.Figure, path: Path, dpi: int = 150):
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Figure 1: Heatmap of fairness gap per (dataset × model) ──────────────────

def figure_1_fairness_heatmap(
    results: pd.DataFrame,
    cfg: Config,
    metric: str = "equalized_odds_diff",
    mitigation: str = "none",
):
    """
    Heatmap: rows = models, columns = dataset variants, colour = gap metric.
    Shows the problem before mitigation.
    """
    subset = results[results["mitigation"] == mitigation].copy()
    pivot = subset.pivot_table(
        index="model", columns="dataset_id", values=metric, aggfunc="mean"
    )
    # Rename axes
    pivot.columns = [DATASET_LABELS.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.4), 4))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=_PALETTE,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": metric.replace("_", " ").title()},
    )
    ax.set_title(
        f"Fairness gap ({metric.replace('_',' ')}) — {MITIGATION_LABELS.get(mitigation, mitigation)}",
        pad=12,
    )
    ax.set_xlabel("Dataset variant")
    ax.set_ylabel("Model")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    out = cfg.output_dir / "figures" / f"fig1_heatmap_{metric}_{mitigation}.png"
    _save(fig, out)
    return out


# ── Figure 2: Before vs. after mitigation for one dataset ────────────────────

def figure_2_mitigation_comparison(
    results: pd.DataFrame,
    cfg: Config,
    dataset_id: str = "D1A",
    metric: str = "auroc_gap",
):
    """
    Grouped bar chart: x = model, hue = mitigation strategy, y = gap metric.
    Rows for a fixed dataset variant.
    """
    subset = results[results["dataset_id"] == dataset_id].copy()
    subset["mitigation_label"] = subset["mitigation"].map(
        lambda m: MITIGATION_LABELS.get(m, m)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = sns.color_palette("Set2", n_colors=len(subset["mitigation"].unique()))
    sns.barplot(
        data=subset,
        x="model",
        y=metric,
        hue="mitigation_label",
        palette=palette,
        ax=ax,
        edgecolor="white",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(
        f"Mitigation comparison — {DATASET_LABELS.get(dataset_id, dataset_id)} "
        f"({metric.replace('_', ' ')})"
    )
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.legend(title="Mitigation", bbox_to_anchor=(1.01, 1), loc="upper left")

    out = cfg.output_dir / "figures" / f"fig2_mitigation_{dataset_id}_{metric}.png"
    _save(fig, out)
    return out


# ── Figure 3: Per-group performance bars (baseline, D0 vs perturbed) ─────────

def figure_3_group_performance(
    results: pd.DataFrame,
    cfg: Config,
    model_name: str = "logistic_regression",
    metric_pair: tuple = ("female_auroc", "male_auroc"),
    mitigation: str = "none",
):
    """
    Side-by-side bars comparing female vs. male on a given metric across
    all dataset variants.
    """
    subset = results[
        (results["model"] == model_name) & (results["mitigation"] == mitigation)
    ].copy()
    subset["dataset_label"] = subset["dataset_id"].map(
        lambda d: DATASET_LABELS.get(d, d)
    )

    m_female, m_male = metric_pair
    melted = pd.melt(
        subset,
        id_vars=["dataset_label"],
        value_vars=[m_female, m_male],
        var_name="group",
        value_name="metric_value",
    )
    melted["group"] = melted["group"].map(
        {m_female: "Female", m_male: "Male"}
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=melted,
        x="dataset_label",
        y="metric_value",
        hue="group",
        palette={"Female": "#E07B78", "Male": "#5B8DB8"},
        ax=ax,
        edgecolor="white",
    )
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Per-group {m_female.split('_')[1].upper()} — {model_name} ({mitigation})"
    )
    ax.set_xlabel("Dataset variant")
    ax.set_ylabel(m_female.split("_", 1)[1].upper())
    ax.legend(title="Group")

    out = cfg.output_dir / "figures" / f"fig3_group_{model_name}_{m_female.split('_')[1]}.png"
    _save(fig, out)
    return out


# ── Figure 4: Parameter sweep (appendix) ─────────────────────────────────────

def figure_4_sweep(
    sweep_results: pd.DataFrame,
    cfg: Config,
    metric: str = "equalized_odds_diff",
    model_name: str = "logistic_regression",
):
    """
    Line plots showing how fairness gap changes as perturbation severity varies.
    One subplot per sweep parameter.
    """
    sweep_results = sweep_results[sweep_results["model"] == model_name]
    params = sweep_results["sweep_param"].unique()
    n = len(params)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
        sub = sweep_results[
            (sweep_results["sweep_param"] == param) &
            (sweep_results["mitigation"] == "none")
        ].sort_values("sweep_value")
        ax.plot(sub["sweep_value"], sub[metric], marker="o", color="#E07B78")
        ax.set_xlabel(param.replace("_", " ").title())
        ax.set_ylabel(metric.replace("_", " ").title() if ax is axes[0] else "")
        ax.set_title(param.replace("_", " ").replace("fraction", "rate"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.suptitle(
        f"Fairness gap vs. perturbation severity — {model_name}", y=1.02
    )
    out = cfg.output_dir / "figures" / f"fig4_sweep_{metric}_{model_name}.png"
    _save(fig, out)
    return out


# ── Figure 5: Full mitigation × dataset heatmap grid ─────────────────────────

def figure_5_full_grid(
    results: pd.DataFrame,
    cfg: Config,
    metric: str = "equalized_odds_diff",
):
    """
    One heatmap per model: rows = mitigation, cols = dataset.
    Suitable for a two-slide 'before / after' comparison.
    """
    models = results["model"].unique()
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        sub = results[results["model"] == model_name]
        pivot = sub.pivot_table(
            index="mitigation", columns="dataset_id", values=metric, aggfunc="mean"
        )
        pivot.index = [MITIGATION_LABELS.get(i, i) for i in pivot.index]
        pivot.columns = [DATASET_LABELS.get(c, c).replace("\n", " ") for c in pivot.columns]

        sns.heatmap(
            pivot,
            ax=ax,
            cmap=_PALETTE,
            annot=True,
            fmt=".3f",
            linewidths=0.5,
            vmin=0,
            cbar=(ax is axes[-1]),
        )
        ax.set_title(model_name.replace("_", " ").title())
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Mitigation" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        f"{metric.replace('_', ' ').title()} — model × mitigation × dataset", y=1.02
    )
    out = cfg.output_dir / "figures" / f"fig5_full_grid_{metric}.png"
    _save(fig, out)
    return out


# ── Tables ────────────────────────────────────────────────────────────────────

def table_1_summary(
    results: pd.DataFrame,
    cfg: Config,
    metrics: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Export a summary CSV (and LaTeX) with mean metric per (dataset, model, mitigation).
    """
    if metrics is None:
        metrics = [
            "overall_auroc", "overall_tpr", "overall_fpr",
            "female_auroc", "male_auroc", "auroc_gap",
            "equalized_odds_diff", "demographic_parity_diff",
        ]
    keep_cols = ["dataset_id", "model", "mitigation"] + [
        m for m in metrics if m in results.columns
    ]
    tbl = results[keep_cols].round(4)

    csv_path = cfg.output_dir / "tables" / "table1_summary.csv"
    tbl.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    try:
        latex_path = cfg.output_dir / "tables" / "table1_summary.tex"
        tbl.to_latex(latex_path, index=False, float_format="%.3f")
        logger.info("Saved %s", latex_path)
    except Exception:
        pass

    return tbl


def table_2_gap_heatmap_data(
    results: pd.DataFrame,
    cfg: Config,
    gap_metric: str = "equalized_odds_diff",
) -> pd.DataFrame:
    """
    Return a pivot table: rows = (model, mitigation), columns = dataset_id.
    Suitable for a publication heatmap table.
    """
    pivot = results.pivot_table(
        index=["model", "mitigation"],
        columns="dataset_id",
        values=gap_metric,
        aggfunc="mean",
    ).round(4)

    csv_path = cfg.output_dir / "tables" / f"table2_{gap_metric}.csv"
    pivot.to_csv(csv_path)
    logger.info("Saved %s", csv_path)
    return pivot


# ── Figure 6: Per-group calibration (ECE) ────────────────────────────────────

def figure_6_calibration_gap(
    results: pd.DataFrame,
    cfg: Config,
    mitigation: str = "none",
):
    """
    Grouped bar: ECE for female vs. male across models and dataset variants.
    ECE gap reveals whether the model is systematically miscalibrated for one
    group — important for clinical decision-making.
    """
    needed = {"female_ece", "male_ece"}
    if not needed.issubset(results.columns):
        logger.warning("ECE columns not in results — skipping calibration figure")
        return

    subset = results[results["mitigation"] == mitigation].copy()

    melted = pd.melt(
        subset,
        id_vars=["model", "dataset_id"],
        value_vars=["female_ece", "male_ece"],
        var_name="group",
        value_name="ECE",
    )
    melted["group"] = melted["group"].map(
        {"female_ece": "Female", "male_ece": "Male"}
    )
    melted["dataset_label"] = melted["dataset_id"].map(
        lambda d: DATASET_LABELS.get(d, d).replace("\n", " ")
    )

    models = subset["model"].unique()
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = melted[melted["model"] == model]
        sns.barplot(
            data=sub,
            x="dataset_label",
            y="ECE",
            hue="group",
            palette={"Female": "#E07B78", "Male": "#5B8DB8"},
            ax=ax,
            edgecolor="white",
        )
        ax.set_title(model.replace("_", " ").title())
        ax.set_xlabel("Dataset")
        ax.set_ylabel("ECE (lower = better)" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=30)
        if ax is not axes[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Group")

    fig.suptitle(
        f"Expected Calibration Error by group — {MITIGATION_LABELS.get(mitigation, mitigation)}",
        y=1.02,
    )
    out = cfg.output_dir / "figures" / f"fig6_calibration_ece_{mitigation}.png"
    _save(fig, out)
    return out


# ── Figure 7: Per-metric per-group summary table plot ────────────────────────

def figure_7_metric_summary(
    results: pd.DataFrame,
    cfg: Config,
    dataset_id: str = "D0",
    mitigation: str = "none",
):
    """
    Horizontal bar chart summarising all per-group metrics for one
    (dataset, mitigation) cell, averaged across models.
    """
    subset = results[
        (results["dataset_id"] == dataset_id) &
        (results["mitigation"] == mitigation)
    ]
    if subset.empty:
        return

    metric_map = {
        "AUROC":            ("female_auroc", "male_auroc"),
        "TPR (Recall)":     ("female_tpr",   "male_tpr"),
        "FPR":              ("female_fpr",   "male_fpr"),
        "Precision (PPV)":  ("female_precision", "male_precision"),
        "F1":               ("female_f1",    "male_f1"),
        "ECE":              ("female_ece",   "male_ece"),
        "Clin. Utility":    ("female_clinical_utility", "male_clinical_utility"),
    }

    rows = []
    for label, (fc, mc) in metric_map.items():
        if fc in subset.columns and mc in subset.columns:
            rows.append({
                "Metric":  label,
                "Female":  subset[fc].mean(),
                "Male":    subset[mc].mean(),
            })

    if not rows:
        return

    df_plot = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, max(4, len(rows) * 0.7)))
    y = np.arange(len(rows))
    height = 0.35
    bars_f = ax.barh(y + height / 2, df_plot["Female"], height, label="Female", color="#E07B78")
    bars_m = ax.barh(y - height / 2, df_plot["Male"],   height, label="Male",   color="#5B8DB8")
    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["Metric"])
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Metric value")
    ax.set_title(
        f"Per-group metrics — {DATASET_LABELS.get(dataset_id, dataset_id)} / "
        f"{MITIGATION_LABELS.get(mitigation, mitigation)}"
    )
    ax.legend()
    fig.tight_layout()

    out = cfg.output_dir / "figures" / f"fig7_metric_summary_{dataset_id}_{mitigation}.png"
    _save(fig, out)
    return out


# ── Convenience: render all standard figures ──────────────────────────────────

def render_all(
    results: pd.DataFrame,
    cfg: Config,
    sweep_results: Optional[pd.DataFrame] = None,
):
    """Render the full standard figure set."""
    logger.info("Rendering figures ...")

    figure_1_fairness_heatmap(results, cfg, metric="equalized_odds_diff")
    figure_1_fairness_heatmap(results, cfg, metric="auroc_gap")

    for did in results["dataset_id"].unique():
        figure_2_mitigation_comparison(results, cfg, dataset_id=did)

    for model in results["model"].unique():
        figure_3_group_performance(results, cfg, model_name=model)

    figure_5_full_grid(results, cfg, metric="equalized_odds_diff")
    figure_5_full_grid(results, cfg, metric="auroc_gap")

    figure_6_calibration_gap(results, cfg, mitigation="none")

    for did in ["D0", "D1A"]:
        if did in results["dataset_id"].values:
            figure_7_metric_summary(results, cfg, dataset_id=did)

    if sweep_results is not None and not sweep_results.empty:
        for model in results["model"].unique():
            figure_4_sweep(sweep_results, cfg, model_name=model)

    table_1_summary(results, cfg)
    table_2_gap_heatmap_data(results, cfg, gap_metric="equalized_odds_diff")
    table_2_gap_heatmap_data(results, cfg, gap_metric="auroc_gap")

    logger.info("All figures and tables saved to %s", cfg.output_dir)
