"""
Jupyter Notebook — Early-Detection Fairness Setup
==================================================
Copy each section into its own cell.

This notebook answers:
  "Does the sepsis-prediction model warn female and male patients
   equally early — and does bias mitigation close that gap?"

The target is NOT "did this patient get sepsis?" (binary classification)
but "will sepsis occur in the next 6 hours?" evaluated repeatedly at
every ICU hour, with fairness measured as the *time between model
alarm and actual onset*, compared across demographic groups.
"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Imports & path setup                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

PIPELINE_DIR = Path("/Users/vrose/ClaudeContainer/PUBH8475/Final/vibe_init")
sys.path.insert(0, str(PIPELINE_DIR))

from config import Config
from data_loader_timeseries import (
    load_timeseries_dataset,
    patient_level_split,
    split_Xy_sensitive,
    get_feature_columns,
    LABEL_COL_TS,
    VITAL_COLS,
    LAB_COLS,
    NUMERIC_COLS,
)
from fairness_timeseries import (
    compute_detection_fairness_report,
    detection_lead_hours,
    detection_summary_table,
)
from evaluation_timeseries import (
    evaluate_ts_single,
    run_timeseries_evaluation,
    build_timeseries_variants,
    run_timeseries_all_variants,
)
from preprocessing import build_preprocessor
from models import get_model, list_models
from mitigation import get_mitigation

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
print("✓ Imports OK")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Config                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

cfg = Config()
cfg.data_dir   = Path("/Users/vrose/ClaudeContainer/PUBH8475/Final/vibe_init/data/physionet_sepsis")
cfg.output_dir = Path("/Users/vrose/ClaudeContainer/PUBH8475/Final/vibe_init/outputs")
cfg.random_state = 42
cfg.__post_init__()

# ── Lookahead window ──────────────────────────────────────────────
# 6-hour horizon: "will sepsis onset occur within the next 6 hours?"
# Clinically motivated: sepsis bundles (antibiotics, fluid resus)
# should be initiated ≥ 1 hour before onset for maximum benefit.
LOOKAHEAD_HOURS = 6
WINDOW_HOURS    = 6   # rolling feature window

rng = np.random.default_rng(cfg.random_state)
print(f"✓ Config  |  lookahead={LOOKAHEAD_HOURS}h  |  window={WINDOW_HOURS}h")
print(f"  Data dir: {cfg.data_dir}")
print(f"  Output:   {cfg.output_dir}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Load time-series dataset (cached after first run)      ║
# ╚══════════════════════════════════════════════════════════════════╝

# First run: reads every PSV file (~20 k), builds rolling features,
#            saves a parquet cache.
# Subsequent runs: loads the cache in seconds.
#
# max_patients=500 for a quick smoke test; set None for all patients.

df_ts = load_timeseries_dataset(
    cfg,
    lookahead_hours=LOOKAHEAD_HOURS,
    window_hours=WINDOW_HOURS,
    max_patients=None,   # ← set e.g. 500 for a quick test
    cache=True,
)

print(f"\n✓ Loaded time-series dataset")
print(f"  Shape         : {df_ts.shape}")
print(f"  Unique patients: {df_ts['patient_id'].nunique()}")
print(f"  Total hours    : {len(df_ts):,}")
print(f"  % positive     : {df_ts[LABEL_COL_TS].mean()*100:.1f}%  "
      f"(sepsis within {LOOKAHEAD_HOURS}h)")
print(f"  Female hours   : {(df_ts['Gender']==0).sum():,}  "
      f"({(df_ts['Gender']==0).mean()*100:.1f}%)")
print(f"  Male hours     : {(df_ts['Gender']==1).sum():,}  "
      f"({(df_ts['Gender']==1).mean()*100:.1f}%)")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 4 — EDA: baseline detection-timing disparity               ║
# ╚══════════════════════════════════════════════════════════════════╝

# Naive check: if we alarm at threshold=0.5 on the raw label,
# how many hours before onset does each group first get flagged?
# (Replace with model predictions after training.)

print("\n── Positive prevalence by group ──")
for gval, glabel in [(0, "Female"), (1, "Male")]:
    grp = df_ts[df_ts["Gender"] == gval]
    prev = grp[LABEL_COL_TS].mean()
    n_patients = grp["patient_id"].nunique()
    septic_patients = grp[grp[LABEL_COL_TS] == 1]["patient_id"].nunique()
    print(f"  {glabel}: {n_patients} patients, {septic_patients} septic, "
          f"{prev*100:.1f}% positive rows")

# Distribution of hours_until_sepsis for septic patients
septic = df_ts[df_ts["hours_until_sepsis"].notna()].copy()
print(f"\n── Hours until sepsis (septic patients only) ──")
for gval, glabel in [(0, "Female"), (1, "Male")]:
    grp = septic[septic["Gender"] == gval]["hours_until_sepsis"]
    print(f"  {glabel}: mean={grp.mean():.1f}h  "
          f"median={grp.median():.1f}h  "
          f"max={grp.max():.0f}h")

# Plot: distribution of ICU hours at detection window boundary
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (gval, glabel, col) in zip(axes, [
    (0, "Female", "#E07B78"),
    (1, "Male",   "#5B8DB8"),
]):
    grp = septic[septic["Gender"] == gval]["hours_until_sepsis"]
    ax.hist(grp, bins=30, color=col, edgecolor="white", alpha=0.8)
    ax.axvline(LOOKAHEAD_HOURS, color="black", linestyle="--",
               label=f"Lookahead = {LOOKAHEAD_HOURS}h")
    ax.set_title(f"{glabel} — hours until sepsis onset")
    ax.set_xlabel("Hours until sepsis")
    ax.set_ylabel("Patient-hours")
    ax.legend()
plt.suptitle("Distribution of hours until sepsis onset by sex", y=1.02)
plt.tight_layout()
plt.savefig(cfg.output_dir / "figures" / "ts_eda_hours_until_sepsis.png",
            bbox_inches="tight")
plt.show()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 5 — Quick single-model early-detection evaluation          ║
# ╚══════════════════════════════════════════════════════════════════╝

# Evaluate logistic regression with no mitigation on D0.
result_lr = evaluate_ts_single(
    df_ts=df_ts,
    model_name="logistic_regression",
    mitigation_name="none",
    cfg=cfg,
    rng=rng,
)

print("── Logistic Regression (no mitigation) — Early-Detection Fairness ──\n")

print(f"{'Metric':<40} {'Female':>10} {'Male':>10} {'Gap (F−M)':>12}")
print("─" * 75)
show_metrics = [
    "median_detection_lead_hours",
    "pct_detected_before_onset",
    "pct_missed",
    "alarm_fatigue_rate",
    "auroc",
    "tpr",
    "fpr",
    "clinical_utility",
]
for m in show_metrics:
    fv = result_lr.get(f"female_{m}", float("nan"))
    mv = result_lr.get(f"male_{m}",   float("nan"))
    gv = result_lr.get(f"{m}_gap",    float("nan"))
    print(f"  {m:<38} {fv:>10.3f} {mv:>10.3f} {gv:>12.3f}")

print()
print(f"  Equalized-odds diff          : {result_lr['equalized_odds_diff']:.3f}")
print(f"  Early-detection parity (hrs) : {result_lr['early_detection_parity']:.3f}")
print(f"  Illusion of fairness flag    : {result_lr['illusion_of_fairness']}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 6 — Full grid: models × mitigations                        ║
# ╚══════════════════════════════════════════════════════════════════╝

# Runs all model × mitigation combinations on D0.
# Typically 3 models × 5 mitigations = 15 cells.
# Comment out models you don't need for speed.

cfg.model.models = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
]
cfg.mitigation.strategies = [
    "none",
    "reweighting",
    "smote",
    "fairness_penalty",
    "robust_model",
]

results_ts = run_timeseries_evaluation(df_ts, cfg, rng, dataset_id="D0_ts")
results_ts.to_csv(cfg.output_dir / "tables" / "results_timeseries_D0.csv", index=False)
print(f"✓ Grid complete: {len(results_ts)} cells")
print(f"  Saved → outputs/tables/results_timeseries_D0.csv")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 7 — View: detection timing fairness summary                ║
# ╚══════════════════════════════════════════════════════════════════╝

summary = detection_summary_table(results_ts)
print(summary.to_string())


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 8 — Plot: detection lead hours by group and mitigation     ║
# ╚══════════════════════════════════════════════════════════════════╝

if "female_median_detection_lead_hours" in results_ts.columns:

    plot_df = results_ts.melt(
        id_vars=["model", "mitigation"],
        value_vars=["female_median_detection_lead_hours",
                    "male_median_detection_lead_hours"],
        var_name="group",
        value_name="lead_hours",
    )
    plot_df["group"] = plot_df["group"].map({
        "female_median_detection_lead_hours": "Female",
        "male_median_detection_lead_hours":   "Male",
    })
    plot_df["mitigation"] = plot_df["mitigation"].map({
        "none":             "Baseline",
        "reweighting":      "Reweighting",
        "smote":            "SMOTE",
        "fairness_penalty": "Fairness Penalty",
        "robust_model":     "Robust Model",
    })

    models = results_ts["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = plot_df[plot_df["model"] == model]
        sns.barplot(
            data=sub,
            x="mitigation",
            y="lead_hours",
            hue="group",
            palette={"Female": "#E07B78", "Male": "#5B8DB8"},
            ax=ax,
            edgecolor="white",
        )
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(model.replace("_", " ").title())
        ax.set_xlabel("Mitigation")
        ax.set_ylabel("Median detection lead (hours)\n← late | early →"
                      if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=30)
        if ax is not axes[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Group")

    fig.suptitle(
        f"Early detection lead time by group and mitigation\n"
        f"Positive = warned before onset | Negative = warned after onset",
        y=1.03,
    )
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "figures" / "ts_detection_lead_by_mitigation.png",
                bbox_inches="tight")
    plt.show()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 9 — Plot: missed alarm rate by group and mitigation        ║
# ╚══════════════════════════════════════════════════════════════════╝

if "female_pct_missed" in results_ts.columns:
    miss_df = results_ts.melt(
        id_vars=["model", "mitigation"],
        value_vars=["female_pct_missed", "male_pct_missed"],
        var_name="group",
        value_name="pct_missed",
    )
    miss_df["group"] = miss_df["group"].map({
        "female_pct_missed": "Female",
        "male_pct_missed":   "Male",
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=miss_df,
        x="mitigation",
        y="pct_missed",
        hue="group",
        palette={"Female": "#E07B78", "Male": "#5B8DB8"},
        ax=ax,
        edgecolor="white",
    )
    ax.set_title("Missed alarm rate by group and mitigation\n(% of septic patients never warned)")
    ax.set_ylabel("% patients never warned")
    ax.set_xlabel("Mitigation")
    ax.legend(title="Group")
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "figures" / "ts_missed_alarm_rate.png",
                bbox_inches="tight")
    plt.show()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 10 — Dataset variants (perturbation effects on early det.) ║
# ╚══════════════════════════════════════════════════════════════════╝

# Build D1A/D1B/D2A/D2B/D3A/D3B variants from the time-series data
# and evaluate the full grid on each.
# ⚠ This is slow — comment out if just exploring.

print("Building time-series dataset variants...")
ts_variants = build_timeseries_variants(df_ts, cfg, rng)

print(f"\n✓ Variants built: {list(ts_variants.keys())}")
for did, dv in ts_variants.items():
    n_f = (dv["Gender"] == 0).sum()
    n_m = (dv["Gender"] == 1).sum()
    print(f"  {did}: {len(dv):,} rows ({n_f:,}F / {n_m:,}M)")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 11 — Run all variants                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

results_all_ts = run_timeseries_all_variants(ts_variants, cfg, rng)
results_all_ts.to_csv(
    cfg.output_dir / "tables" / "results_timeseries_all_variants.csv",
    index=False,
)
print(f"✓ Full grid: {len(results_all_ts)} cells across {len(ts_variants)} variants")
print(f"  Saved → outputs/tables/results_timeseries_all_variants.csv")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 12 — Heatmap: detection parity across variants             ║
# ╚══════════════════════════════════════════════════════════════════╝

metric = "early_detection_parity"   # |median lead gap| in hours

if metric in results_all_ts.columns:
    models = results_all_ts["model"].unique()
    fig, axes = plt.subplots(1, len(models),
                              figsize=(6 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]

    mit_labels = {
        "none":             "Baseline",
        "reweighting":      "Reweighting",
        "smote":            "SMOTE",
        "fairness_penalty": "Fairness Penalty",
        "robust_model":     "Robust Model",
    }
    ds_labels = {
        "D0_ts": "Original", "D1A_ts": "Row-rm (F↓)", "D1B_ts": "Row-rm (M↓)",
        "D2A_ts": "MAR (F)", "D2B_ts": "MAR (M)",
        "D3A_ts": "Noise (F)", "D3B_ts": "Noise (M)",
    }

    for ax, model in zip(axes, models):
        sub = results_all_ts[results_all_ts["model"] == model]
        pivot = sub.pivot_table(
            index="mitigation", columns="dataset_id",
            values=metric, aggfunc="mean"
        )
        pivot.index   = [mit_labels.get(i, i) for i in pivot.index]
        pivot.columns = [ds_labels.get(c, c) for c in pivot.columns]

        sns.heatmap(
            pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
            linewidths=0.5, vmin=0,
            cbar=(ax is axes[-1]),
            cbar_kws={"label": "Detection gap (hours)"},
        )
        ax.set_title(model.replace("_", " ").title())
        ax.set_xlabel("Dataset variant")
        ax.set_ylabel("Mitigation" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        "Early-detection parity gap (hours) — lower = fairer\n"
        "= |median female lead − median male lead|",
        y=1.04,
    )
    plt.tight_layout()
    plt.savefig(
        cfg.output_dir / "figures" / "ts_early_detection_parity_heatmap.png",
        bbox_inches="tight",
    )
    plt.show()
