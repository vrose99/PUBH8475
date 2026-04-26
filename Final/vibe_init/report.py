"""
Automated markdown report generator (time-series mode).

Writes a self-contained markdown narrative to outputs/report.md.

Entry point: generate_report(results, cfg, analysis_summary=None)
"""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


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
    "reweighting":      "**Reweighting** — inverse-frequency sample weights per (group × label) "
                        "cell so each subgroup contributes equally to the training loss.",
    "smote":            "**SMOTE** — Synthetic Minority Over-sampling applied to the "
                        "underrepresented gender group to equalise sizes before training. "
                        "Falls back to random oversampling when a class has fewer than 6 samples.",
    "fairness_penalty": "**Fairness Penalty (ExponentiatedGradient)** — fairlearn's "
                        "constrained-optimisation wrapper that explicitly penalises violations "
                        "of the equalized-odds constraint during training.",
}


def _section_methods(cfg: Config) -> str:
    models = ", ".join(f"`{m}`" for m in cfg.model.models)
    mitigations = "\n".join(
        f"- {MITIGATION_DESCRIPTIONS.get(s, s)}"
        for s in cfg.mitigation.strategies
    )
    return f"""\
## Methods

### Data
We use the PhysioNet/CinC 2019 Sepsis Challenge dataset (Reyna et al., 2019).
Each observation is one patient-hour. Features include raw clinical values plus
rolling 6-hour statistics (mean, std, min, max, trend). The prediction target is
a binary early-detection label: will sepsis onset occur within the next 6 hours?

The sensitive attribute is **biological sex** (0 = female, 1 = male) as recorded
in the challenge files.  Train/test splits are performed at the patient level to
prevent temporal leakage.

### Models
The following classifiers are evaluated: {models}.
See `models.py` for full hyperparameters.

### Mitigation strategies
{mitigations}

### Utility function
The **PhysioNet 2019 challenge utility score** is the primary performance metric.
It rewards early warnings (alarm triggered before onset) and penalises missed
sepsis and alarm fatigue using a piecewise linear reward schedule aligned with
clinical practice.  Normalised so that 1 = perfect early detection and
0 = inaction (no alarms).

### Fairness metrics
Three gap metrics quantify disparity between female and male patients:

1. **Detection lead gap (hours)** — difference in median detection lead times
   (female − male). Positive = females warned earlier.
2. **Missed-rate gap** — difference in the fraction of septic patients who
   received no alarm at all.
3. **Alarm fatigue rate gap** — difference in false-alarm rates for non-septic
   patients.

All three ideal values are 0 (equal treatment of both groups).
"""


def _section_results(results: pd.DataFrame, cfg: Config) -> str:
    fig_dir = "figures"
    baseline = results[results["mitigation"] == "none"]

    util_mean = baseline["overall_physionet_utility"].mean() if "overall_physionet_utility" in baseline.columns else float("nan")
    lead_gap  = baseline["detection_lead_gap_hours"].mean()  if "detection_lead_gap_hours"  in baseline.columns else float("nan")
    miss_gap  = baseline["missed_rate_gap"].mean()           if "missed_rate_gap"           in baseline.columns else float("nan")
    fatigue_gap = baseline["alarm_fatigue_rate_gap"].mean()  if "alarm_fatigue_rate_gap"    in baseline.columns else float("nan")

    return f"""\
## Results

### Phase A — Baseline (no mitigation)

Without any fairness intervention the mean PhysioNet utility score across models
is **{_fmt(util_mean)}**.  The three fairness gaps are:

| Metric | Mean (female − male) |
|--------|----------------------|
| Detection lead gap (hours) | {_fmt(lead_gap)} |
| Missed-rate gap | {_fmt(miss_gap)} |
| Alarm fatigue rate gap | {_fmt(fatigue_gap)} |

{_img("PhysioNet utility score per model and mitigation", f"{fig_dir}/fig1_utility_score.png")}

{_img("Detection lead hours per group (female vs. male)", f"{fig_dir}/fig2_detection_lead_hours.png")}

### Phase B — After mitigation

{_img("Three fairness gap metrics per model and mitigation", f"{fig_dir}/fig3_fairness_gap_metrics.png")}

{_img("Mitigation Δ vs. baseline heatmap", f"{fig_dir}/fig4_mitigation_delta.png")}

Full results table:
{_tbl_link("tables/table1_summary.csv", "tables/table1_summary.csv")}.
"""


def _section_mitigation(results: pd.DataFrame, analysis_summary: dict, cfg: Config) -> str:
    fig_dir = "figures"
    best   = analysis_summary.get("best_mitigation", "N/A")
    best_g = _fmt(analysis_summary.get("best_gap"))
    base_g = _fmt(analysis_summary.get("baseline_gap"))

    ranking = analysis_summary.get("ranking", [])
    tbl_rows = []
    for r in ranking:
        mit   = r.get("mitigation", "")
        label = r.get("mitigation_label", mit)
        gap   = _fmt(r.get(f"abs_detection_lead_gap_hours"))
        util  = _fmt(r.get("overall_physionet_utility"))
        tbl_rows.append(f"| {label} | {gap} | {util} |")
    tbl_str = (
        "| Mitigation | |Det. Lead Gap| (hrs) | Utility Score |\n"
        "|---|---|---|\n" +
        "\n".join(tbl_rows)
    ) if tbl_rows else ""

    return f"""\
## Mitigation Effectiveness

Across all models the most effective strategy for reducing detection lead gap
is **{best}** (gap = {best_g} vs. baseline {base_g}).

{tbl_str}

{_img("Fairness–utility trade-off scatter", f"{fig_dir}/analysis_fairness_utility_tradeoff.png")}

{_img("Metric Δ vs. baseline (all mitigations)", f"{fig_dir}/analysis_delta_bars.png")}

Mitigation ranking table:
{_tbl_link("tables/analysis_mitigation_ranking.csv", "tables/analysis_mitigation_ranking.csv")}.
"""


def _section_reproducibility(cfg: Config) -> str:
    return f"""\
## Reproducibility

```bash
cd /path/to/vibe_init
python pipeline.py --data-dir /path/to/physionet_sepsis --seed {cfg.random_state}
```

| File | Role |
|------|------|
| `config.py` | All hyperparameters |
| `data_loader_timeseries.py` | PSV → patient-hour dataset |
| `evaluation_timeseries.py` | Train/test/metrics loop |
| `fairness_timeseries.py` | Time-series fairness metrics |
| `mitigation.py` | Bias mitigation strategies |
| `analysis.py` | Post-hoc analysis |
| `visualization.py` | Publication-quality figures |
| `report.py` | This document |

Random seed: `{cfg.random_state}` · Test fraction: `{cfg.model.test_size}` · Threshold: `{cfg.fairness.decision_threshold}`
"""


def generate_report(
    results: pd.DataFrame,
    cfg: Config,
    analysis_summary: Optional[dict] = None,
    out_path: Optional[Path] = None,
) -> Path:
    if analysis_summary is None:
        analysis_summary = {}

    out_path = out_path or (cfg.output_dir / "report.md")

    sections = [
        f"# Fairness in Sepsis Early Detection: Time-Series Analysis\n\n"
        f"*Generated {date.today().isoformat()} · PhysioNet/CinC 2019 · seed={cfg.random_state}*\n",

        "## Table of Contents\n\n"
        "1. [Methods](#methods)\n"
        "2. [Results](#results)\n"
        "3. [Mitigation Effectiveness](#mitigation-effectiveness)\n"
        "4. [Reproducibility](#reproducibility)\n",

        _section_methods(cfg),
        _section_results(results, cfg),
        _section_mitigation(results, analysis_summary, cfg),
        _section_reproducibility(cfg),

        "\n---\n*Report auto-generated by `report.py`.*\n",
    ]

    content = "\n".join(sections)
    out_path.write_text(content, encoding="utf-8")
    logger.info("Report written to %s", out_path)
    return out_path
