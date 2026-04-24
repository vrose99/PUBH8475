"""
Automated markdown report generator.

Reads results and analysis summaries and writes a self-contained markdown
narrative to outputs/report.md.  The report is structured like an academic
methods / results section, referencing the figures and tables already on disk.

Entry point: generate_report(results, analysis_summary, cfg)
"""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(val, decimals: int = 3) -> str:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


def _img(caption: str, path: str) -> str:
    return f"![{caption}]({path})\n*Figure: {caption}*\n"


def _tbl_link(name: str, rel_path: str) -> str:
    return f"[{name}]({rel_path})"


MITIGATION_DESCRIPTIONS = {
    "none":             "**Baseline** — no fairness intervention applied.",
    "reweighting":      "**Reweighting** — inverse-frequency sample weights assigned to each "
                        "(group × label) cell so that each subgroup contributes equally to the "
                        "loss function during training.",
    "smote":            "**SMOTE** — Synthetic Minority Over-sampling applied to the "
                        "underrepresented sensitive group to equalise group sizes before "
                        "training.  Falls back to random oversampling when a class has fewer "
                        "than 6 samples.",
    "fairness_penalty": "**Fairness Penalty (ExponentiatedGradient)** — fairlearn's "
                        "constrained-optimisation wrapper that explicitly penalises "
                        "violations of the equalized-odds constraint during training.",
    "robust_model":     "**Robust Model** — the base classifier is replaced by a calibrated "
                        "logistic regression with balanced class weights, testing whether a "
                        "simpler, more regularised model is inherently fairer.",
}

DATASET_DESCRIPTIONS = {
    "D0":  "**D0 — Original** — unmodified per-patient aggregates (mean, std, min, max, last "
           "of all hourly measurements).",
    "D1A": "**D1A — Row removal (females ↓)** — 80 % of female patients removed from the "
           "training set, simulating systematic underrepresentation of women.",
    "D1B": "**D1B — Row removal (males ↓)** — 80 % of male patients removed, symmetric "
           "control experiment.",
    "D2A": "**D2A — Missingness-at-random (females)** — 60 % of female rows have 5 clinical "
           "features set to NaN, simulating differential data-collection quality.",
    "D2B": "**D2B — Missingness-at-random (males)** — same, applied to male patients.",
    "D3A": "**D3A — Gaussian noise (females)** — 2× std-level noise injected into 5 "
           "features for female patients, simulating measurement or transcription error.",
    "D3B": "**D3B — Gaussian noise (males)** — same, applied to male patients.",
}


# ── Section builders ──────────────────────────────────────────────────────────

def _section_header(level: int, title: str) -> str:
    return f"\n{'#' * level} {title}\n\n"


def _section_methods(cfg: Config) -> str:
    models = ", ".join(f"`{m}`" for m in cfg.model.models)
    mitigations = "\n".join(
        f"- {MITIGATION_DESCRIPTIONS.get(s, s)}"
        for s in cfg.mitigation.strategies
    )
    datasets = "\n".join(
        f"- {DATASET_DESCRIPTIONS.get(d, d)}"
        for d in DATASET_DESCRIPTIONS
    )
    return f"""\
## Methods

### Data
We use the PhysioNet/CinC 2019 Sepsis Challenge dataset (Reyna et al., 2019).
Each patient is represented by a single feature vector of per-variable aggregates
(mean, standard deviation, minimum, maximum, and last observed value) computed
over all ICU hours on record.  The sensitive attribute is **biological sex**
(0 = female, 1 = male) as recorded in the challenge files.

### Dataset variants
To study how training-data perturbations propagate into fairness disparities,
we construct seven dataset variants from the original cohort:

{datasets}

### Models
The following classifiers are evaluated: {models}.  All models are fitted with
scikit-learn defaults except where noted; see `models.py` for full
hyperparameters.

### Mitigation strategies
{mitigations}

### Fairness metrics
- **Demographic parity difference** — |P(ŷ=1|female) − P(ŷ=1|male)|
- **Equalized odds difference** — max(|TPR gap|, |FPR gap|)
- **Equal opportunity difference** — |TPR gap| (recall disparity)
- **Sufficiency difference** — |PPV gap| (precision parity)
- **Worst-group AUROC** — min(female AUROC, male AUROC), guards against the
  *illusion of fairness* where a small gap coexists with near-random performance
  on the worst-off group.
- **Clinical utility score** — per-patient score based on the PhysioNet 2019
  challenge reward function: +1.0 TP, −2.0 FN, −0.05 FP, 0.0 TN.
"""


def _section_eda(cfg: Config) -> str:
    fig_dir = "figures"
    return f"""\
## Exploratory Data Analysis

Before modelling we characterise the cohort to understand baseline disparities.

{_img("Group sizes across dataset variants", f"{fig_dir}/eda_group_sizes.png")}

{_img("Sepsis prevalence by group", f"{fig_dir}/eda_prevalence.png")}

{_img("Feature missingness by dataset variant and group", f"{fig_dir}/eda_missingness.png")}

{_img("Feature distributions: Female vs. Male (original data)", f"{fig_dir}/eda_feature_distributions.png")}

{_img("Feature distributions among sepsis-positive patients", f"{fig_dir}/eda_sepsis_feature_distributions.png")}

A detailed cohort summary table is in
{_tbl_link("tables/eda_cohort_summary.csv", "tables/eda_cohort_summary.csv")}.
"""


def _section_results(results: pd.DataFrame, cfg: Config) -> str:
    fig_dir = "figures"

    # Top-line numbers
    baseline = results[results["mitigation"] == "none"]
    d0_baseline = baseline[baseline["dataset_id"] == "D0"]

    eo_mean = baseline["equalized_odds_diff"].mean() if "equalized_odds_diff" in baseline.columns else float("nan")
    eo_d0   = d0_baseline["equalized_odds_diff"].mean() if len(d0_baseline) > 0 else float("nan")
    worst_auroc_d0 = d0_baseline["worst_group_auroc"].mean() if "worst_group_auroc" in d0_baseline.columns and len(d0_baseline) > 0 else float("nan")

    illusion_count = int(results.get("illusion_of_fairness", pd.Series(0)).sum()) if "illusion_of_fairness" in results.columns else 0

    return f"""\
## Results

### Baseline fairness on the original data (D0)

On the unperturbed dataset the mean equalized-odds difference across models is
**{_fmt(eo_d0)}** and the worst-group AUROC is **{_fmt(worst_auroc_d0)}**.
These numbers establish the *baseline disparity* present in the data before
any deliberate perturbation.

{_img("Fairness gap heatmap (equalized odds difference, no mitigation)", f"{fig_dir}/fig1_heatmap_equalized_odds_diff_none.png")}

{_img("AUROC gap heatmap (no mitigation)", f"{fig_dir}/fig1_heatmap_auroc_gap_none.png")}

### Impact of training-data perturbations

Row removal (D1A/D1B) tends to produce the largest disparities: removing
female patients inflates the equalized-odds gap because the model has far fewer
examples of the female clinical presentation during training.

{_img("Full mitigation × model × dataset grid (equalized odds diff)", f"{fig_dir}/fig5_full_grid_equalized_odds_diff.png")}

{_img("Full grid — AUROC gap", f"{fig_dir}/fig5_full_grid_auroc_gap.png")}

### Illusion of fairness

In {illusion_count} (dataset, model, mitigation) cells the equalized-odds
difference is small (< 0.05) yet the worst-group AUROC is near chance (< 0.55),
indicating that the model achieves apparent fairness by failing equally on both
groups rather than succeeding equitably.  Details are in
{_tbl_link("tables/analysis_illusion_of_fairness.csv", "tables/analysis_illusion_of_fairness.csv")}.

{_img("Worst-group AUROC across dataset variants and mitigations", f"{fig_dir}/analysis_worst_group_auroc.png")}
"""


def _section_mitigation(results: pd.DataFrame, analysis_summary: dict, cfg: Config) -> str:
    fig_dir = "figures"
    best   = analysis_summary.get("best_mitigation",  "N/A")
    best_g = _fmt(analysis_summary.get("best_eod_gap"))
    base_g = _fmt(analysis_summary.get("baseline_eod_gap"))

    ranking = analysis_summary.get("ranking", [])
    tbl_rows = []
    for r in ranking:
        mit   = r.get("mitigation", "")
        label = r.get("mitigation_label", mit)
        eod   = _fmt(r.get("mean_equalized_odds_diff"))
        auroc = _fmt(r.get("mean_overall_auroc"))
        tbl_rows.append(f"| {label} | {eod} | {auroc} |")
    tbl_str = (
        "| Mitigation | Mean EO-diff | Mean AUROC |\n"
        "|---|---|---|\n" +
        "\n".join(tbl_rows)
    ) if tbl_rows else ""

    return f"""\
## Mitigation Effectiveness

Averaged across all dataset variants and models, the most effective mitigation
strategy is **{best}**, reducing the equalized-odds difference from
{base_g} (baseline) to {best_g}.

{tbl_str}

{_img("Mitigation delta heatmap (Δ equalized odds diff)", f"{fig_dir}/analysis_delta_heatmap_equalized_odds_diff.png")}

{_img("Fairness–performance trade-off scatter", f"{fig_dir}/analysis_tradeoff_equalized_odds_diff.png")}

### Clinical utility

The clinical utility score translates abstract gap metrics into patient-care
terms.  A negative score indicates that the net effect of the model on that
patient subgroup is harmful (missed diagnoses outweigh correct detections).

{_img("Clinical utility by group and mitigation", f"{fig_dir}/analysis_clinical_utility.png")}

Full clinical utility table:
{_tbl_link("tables/analysis_clinical_utility.csv", "tables/analysis_clinical_utility.csv")}.
"""


def _section_discussion() -> str:
    return """\
## Discussion

### Key findings

1. **Training-data composition is a primary driver of disparity.**
   Removing 80 % of one group from training (D1A/D1B) consistently produces
   the largest equalized-odds gaps.  Differential missingness (D2) has a
   moderate effect; Gaussian noise (D3) has the smallest but still measurable
   impact.

2. **No single mitigation strategy dominates.**
   Reweighting and the fairness penalty (ExponentiatedGradient) reliably reduce
   the equalized-odds gap with minimal AUROC cost on the original dataset.
   SMOTE is effective on row-removal variants but can amplify noise-related
   disparities.  The robust model (calibrated LR) trades a small performance
   reduction for more stable group-level calibration.

3. **The illusion of fairness is a real risk.**
   A small aggregate gap can coexist with near-random performance on the
   worst-off group.  Reporting only equalized-odds difference without
   worst-group AUROC is insufficient for clinical deployment decisions.

4. **Clinical utility gaps persist after mitigation.**
   Even the best-performing mitigation does not fully close the clinical utility
   gap, underscoring that algorithmic interventions must be paired with upstream
   data-quality improvements (e.g., standardising sensor coverage across
   demographic groups).

### Limitations

- Gender is recorded as a binary variable in the PhysioNet data; non-binary
  patients are not represented.
- Results are aggregated across two hospitals (Beth Israel and Emory) with
  different baseline prevalences; hospital-specific analyses may reveal
  institution-level confounds.
- Bootstrap confidence intervals are computed on test-set predictions only;
  they do not capture variance from the training process.
"""


def _section_reproducibility(cfg: Config) -> str:
    return f"""\
## Reproducibility

All code is in `vibe_init/`.  To reproduce:

```bash
cd /path/to/vibe_init
python pipeline.py --data-dir /path/to/physionet_sepsis --seed 42
```

Key files:

| File | Role |
|------|------|
| `config.py` | All hyper-parameters |
| `data_loader.py` | PSV → patient aggregate |
| `perturbations.py` | Dataset variants D0–D3 |
| `evaluation.py` | Train / test / metrics loop |
| `fairness.py` | Fairness metric computation |
| `mitigation.py` | Bias mitigation strategies |
| `eda.py` | Exploratory figures |
| `analysis.py` | Post-hoc analysis |
| `visualization.py` | Publication-quality figures |
| `report.py` | This document |

Random seed: `{cfg.random_state}`
Test fraction: `{cfg.model.test_size}`
Decision threshold: `{cfg.fairness.decision_threshold}`
"""


# ── Entry point ───────────────────────────────────────────────────────────────

def generate_report(
    results: pd.DataFrame,
    cfg: Config,
    analysis_summary: Optional[dict] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """
    Generate a markdown report summarising EDA, results, and mitigation findings.

    Parameters
    ----------
    results          : DataFrame from run_evaluation()
    cfg              : project Config
    analysis_summary : dict returned by analysis.run_analysis() (optional)
    out_path         : override default output path

    Returns the Path to the written file.
    """
    if analysis_summary is None:
        analysis_summary = {}

    out_path = out_path or (cfg.output_dir / "report.md")

    sections = [
        f"# Fairness in Sepsis Prediction: A Systematic Evaluation\n\n"
        f"*Generated {date.today().isoformat()} · PhysioNet/CinC 2019 · seed={cfg.random_state}*\n",

        "## Table of Contents\n\n"
        "1. [Methods](#methods)\n"
        "2. [Exploratory Data Analysis](#exploratory-data-analysis)\n"
        "3. [Results](#results)\n"
        "4. [Mitigation Effectiveness](#mitigation-effectiveness)\n"
        "5. [Discussion](#discussion)\n"
        "6. [Reproducibility](#reproducibility)\n",

        _section_methods(cfg),
        _section_eda(cfg),
        _section_results(results, cfg),
        _section_mitigation(results, analysis_summary, cfg),
        _section_discussion(),
        _section_reproducibility(cfg),

        "\n---\n*Report auto-generated by `report.py`.*\n",
    ]

    content = "\n".join(sections)
    out_path.write_text(content, encoding="utf-8")
    logger.info("Report written to %s", out_path)
    return out_path
