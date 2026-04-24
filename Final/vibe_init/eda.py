"""
Exploratory Data Analysis for the fairness pipeline.

Produces group-level summaries, missingness patterns, and feature-distribution
comparisons that contextualise the fairness findings.  All figures land in
cfg.output_dir/figures/eda_* and all tables in cfg.output_dir/tables/eda_*.

Entry points
------------
  run_eda(df_clean, datasets, cfg)
      df_clean  — the unperturbed aggregated DataFrame from data_loader
      datasets  — {dataset_id: DataFrame} dict from perturbations.build_all_datasets
      cfg       — Config
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
from data_loader import LABEL_COL

logger = logging.getLogger(__name__)

_FEMALE_COLOR = "#E07B78"
_MALE_COLOR   = "#5B8DB8"
_GROUP_PALETTE = {"Female": _FEMALE_COLOR, "Male": _MALE_COLOR}


def _save(fig: plt.Figure, path: Path, dpi: int = 150):
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    logger.info("Saved %s", path)


def _group_label(df: pd.DataFrame, cfg: Config) -> pd.Series:
    return df[cfg.fairness.sensitive_column].map(
        {cfg.fairness.female_value: "Female", cfg.fairness.male_value: "Male"}
    )


# ── 1. Cohort summary table ───────────────────────────────────────────────────

def cohort_summary(
    datasets: dict[str, pd.DataFrame],
    cfg: Config,
) -> pd.DataFrame:
    """
    Build a per-dataset × per-group summary table:
      n, sepsis_prevalence (%), missing_rate (%)
    Saved as eda_cohort_summary.csv.
    """
    records = []
    for did, df in datasets.items():
        for gval, glabel in [
            (cfg.fairness.female_value, "Female"),
            (cfg.fairness.male_value,   "Male"),
        ]:
            g = df[df[cfg.fairness.sensitive_column] == gval]
            n = len(g)
            if n == 0:
                continue
            num_cols = [
                c for c in g.columns
                if c not in {LABEL_COL, "dataset_id", cfg.fairness.sensitive_column}
                and pd.api.types.is_numeric_dtype(g[c])
            ]
            miss_rate = g[num_cols].isnull().values.mean() if num_cols else np.nan
            records.append({
                "dataset_id":  did,
                "group":       glabel,
                "n":           n,
                "prevalence":  g[LABEL_COL].mean(),
                "missing_rate": miss_rate,
            })

    tbl = pd.DataFrame(records)
    out = cfg.output_dir / "tables" / "eda_cohort_summary.csv"
    tbl.round(4).to_csv(out, index=False)
    logger.info("Saved %s", out)
    return tbl


# ── 2. Group size bar chart ───────────────────────────────────────────────────

def figure_eda_group_sizes(
    datasets: dict[str, pd.DataFrame],
    cfg: Config,
):
    """Bar chart: n per group across all dataset variants."""
    rows = []
    dataset_labels = {
        "D0": "Original", "D1A": "Row-rm\n(F↓)", "D1B": "Row-rm\n(M↓)",
        "D2A": "MAR\n(F)", "D2B": "MAR\n(M)", "D3A": "Noise\n(F)", "D3B": "Noise\n(M)",
    }
    for did, df in datasets.items():
        for gval, glabel in [
            (cfg.fairness.female_value, "Female"),
            (cfg.fairness.male_value,   "Male"),
        ]:
            rows.append({
                "Dataset": dataset_labels.get(did, did),
                "Group":   glabel,
                "n":       (df[cfg.fairness.sensitive_column] == gval).sum(),
            })

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.6), 4))
    sns.barplot(
        data=plot_df, x="Dataset", y="n", hue="Group",
        palette=_GROUP_PALETTE, ax=ax, edgecolor="white",
    )
    ax.set_title("Cohort size by group and dataset variant")
    ax.set_ylabel("Number of patients")
    ax.legend(title="Group")
    _save(fig, cfg.output_dir / "figures" / "eda_group_sizes.png")


# ── 3. Prevalence bar chart ───────────────────────────────────────────────────

def figure_eda_prevalence(
    datasets: dict[str, pd.DataFrame],
    cfg: Config,
):
    """Bar chart: sepsis prevalence per group across dataset variants."""
    rows = []
    dataset_labels = {
        "D0": "Original", "D1A": "Row-rm\n(F↓)", "D1B": "Row-rm\n(M↓)",
        "D2A": "MAR\n(F)", "D2B": "MAR\n(M)", "D3A": "Noise\n(F)", "D3B": "Noise\n(M)",
    }
    for did, df in datasets.items():
        for gval, glabel in [
            (cfg.fairness.female_value, "Female"),
            (cfg.fairness.male_value,   "Male"),
        ]:
            g = df[df[cfg.fairness.sensitive_column] == gval]
            if len(g) == 0:
                continue
            rows.append({
                "Dataset":    dataset_labels.get(did, did),
                "Group":      glabel,
                "Prevalence": g[LABEL_COL].mean(),
            })

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.6), 4))
    sns.barplot(
        data=plot_df, x="Dataset", y="Prevalence", hue="Group",
        palette=_GROUP_PALETTE, ax=ax, edgecolor="white",
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_title("Sepsis prevalence by group and dataset variant")
    ax.set_ylabel("Prevalence")
    ax.legend(title="Group")
    _save(fig, cfg.output_dir / "figures" / "eda_prevalence.png")


# ── 4. Missingness heatmap ────────────────────────────────────────────────────

def figure_eda_missingness(
    datasets: dict[str, pd.DataFrame],
    cfg: Config,
    top_n: int = 20,
):
    """
    Heatmap of feature-level missingness for female vs. male across dataset
    variants.  Only the top_n columns (by overall missing rate) are shown to
    keep the figure readable.
    """
    exc = {LABEL_COL, "dataset_id", cfg.fairness.sensitive_column}

    def _miss_series(df, gval):
        g = df[df[cfg.fairness.sensitive_column] == gval]
        num_cols = [c for c in g.columns if c not in exc and pd.api.types.is_numeric_dtype(g[c])]
        return g[num_cols].isnull().mean()

    # Collect (dataset, group) → missingness Series
    data_rows = {}
    for did, df in datasets.items():
        for gval, glabel in [
            (cfg.fairness.female_value, "Female"),
            (cfg.fairness.male_value,   "Male"),
        ]:
            key = f"{did}\n({glabel[0]})"
            data_rows[key] = _miss_series(df, gval)

    miss_df = pd.DataFrame(data_rows).T.fillna(0)

    if miss_df.empty or miss_df.shape[1] == 0:
        logger.warning("No missing data found — skipping missingness heatmap")
        return

    # Select top_n columns by mean missingness
    top_cols = miss_df.mean(axis=0).nlargest(top_n).index.tolist()
    miss_df = miss_df[top_cols]

    fig, ax = plt.subplots(figsize=(max(10, len(top_cols) * 0.6), max(5, len(miss_df) * 0.5)))
    sns.heatmap(
        miss_df,
        ax=ax,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        annot=(miss_df.shape[0] * miss_df.shape[1] <= 200),
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={"label": "Missing fraction"},
    )
    ax.set_title("Feature missingness by dataset variant and group")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Dataset (group)")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, cfg.output_dir / "figures" / "eda_missingness.png")


# ── 5. Feature distribution comparison (original data only) ──────────────────

def figure_eda_feature_distributions(
    df: pd.DataFrame,
    cfg: Config,
    top_n: int = 12,
):
    """
    Box-plots comparing female vs. male distributions for the top_n most
    complete numeric features in the original (unperturbed) data.
    """
    exc = {LABEL_COL, "dataset_id", cfg.fairness.sensitive_column}
    num_cols = [
        c for c in df.columns
        if c not in exc and pd.api.types.is_numeric_dtype(df[c])
    ]
    # Select most-complete columns
    completeness = df[num_cols].notnull().mean().sort_values(ascending=False)
    selected = completeness.head(top_n).index.tolist()

    df_plot = df[selected + [cfg.fairness.sensitive_column]].copy()
    df_plot["Group"] = _group_label(df_plot, cfg)

    n_cols = 4
    n_rows = int(np.ceil(len(selected) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(selected):
        ax = axes[i]
        sns.boxplot(
            data=df_plot,
            x="Group", y=col,
            palette=_GROUP_PALETTE,
            ax=ax,
            showfliers=False,
            width=0.5,
        )
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature distributions: Female vs. Male (original data)", y=1.01)
    fig.tight_layout()
    _save(fig, cfg.output_dir / "figures" / "eda_feature_distributions.png")


# ── 6. Sepsis-positive feature comparison ────────────────────────────────────

def figure_eda_sepsis_feature_distributions(
    df: pd.DataFrame,
    cfg: Config,
    top_n: int = 8,
):
    """
    Among sepsis-positive patients only: compare feature distributions by sex.
    Highlights whether the two groups present differently even within cases.
    """
    pos = df[df[LABEL_COL] == 1].copy()
    if len(pos) == 0:
        return

    exc = {LABEL_COL, "dataset_id", cfg.fairness.sensitive_column}
    num_cols = [
        c for c in pos.columns
        if c not in exc and pd.api.types.is_numeric_dtype(pos[c])
    ]
    completeness = pos[num_cols].notnull().mean().sort_values(ascending=False)
    selected = completeness.head(top_n).index.tolist()

    pos["Group"] = _group_label(pos, cfg)

    n_cols = 4
    n_rows = int(np.ceil(len(selected) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(selected):
        ax = axes[i]
        sns.boxplot(
            data=pos,
            x="Group", y=col,
            palette=_GROUP_PALETTE,
            ax=ax,
            showfliers=False,
            width=0.5,
        )
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature distributions among sepsis-positive patients: Female vs. Male", y=1.01)
    fig.tight_layout()
    _save(fig, cfg.output_dir / "figures" / "eda_sepsis_feature_distributions.png")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_eda(
    df_clean: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    cfg: Config,
):
    """
    Run the full EDA suite.

    Parameters
    ----------
    df_clean  — unperturbed aggregated DataFrame (D0 equivalent)
    datasets  — {dataset_id: DataFrame} dict from perturbations.build_all_datasets
    cfg       — project Config
    """
    logger.info("Running EDA ...")

    cohort_summary(datasets, cfg)
    figure_eda_group_sizes(datasets, cfg)
    figure_eda_prevalence(datasets, cfg)
    figure_eda_missingness(datasets, cfg)
    figure_eda_feature_distributions(df_clean, cfg)
    figure_eda_sepsis_feature_distributions(df_clean, cfg)

    logger.info("EDA complete — outputs in %s", cfg.output_dir)
