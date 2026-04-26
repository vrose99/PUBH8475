# Fairness Pipeline — PhysioNet Sepsis

Modular ML fairness analysis pipeline for the PhysioNet/CinC 2019 Sepsis
Prediction Challenge dataset.

---

## Directory layout

```
vibe_init/
├── config.py               ← all tunable parameters (start here)
├── data_loader.py          ← load + aggregate patient PSV files
├── perturbations.py        ← dataset variants D0–D3
├── models.py               ← model registry (add models here)
├── fairness.py             ← fairness metric computation
├── preprocessing.py        ← imputation + scaling pipeline
├── mitigation.py           ← reweighting, SMOTE, fairness penalty, robust model
├── evaluation.py           ← full evaluation loop → results DataFrame
├── visualization.py        ← figures and tables
├── pipeline.py             ← main script
├── make_synthetic_data.py  ← generate fake data for smoke-testing
└── requirements.txt
```

---

## Quick start

```bash
pip install -r requirements.txt

# Option A — smoke test with synthetic data (no download needed)
python make_synthetic_data.py --n-patients 1000 --out data/synthetic_sepsis
python pipeline.py --data-dir data/synthetic_sepsis --max-patients 1000

# Option B — real PhysioNet data
# 1. Register and download from https://physionet.org/content/challenge-2019/1.0.0/ 
# 2. Extract training_setA.zip (and optionally training_setB.zip)
# 3. Run:
python pipeline.py --data-dir path/to/training_setA

# Parameter sweep (appendix figures)
python pipeline.py --data-dir data/... --sweep
```

Outputs land in `outputs/figures/` and `outputs/tables/`.

---

## Dataset variants

| ID  | Description                              | Configures via                        |
|-----|------------------------------------------|---------------------------------------|
| D0  | Original data                            | —                                     |
| D1A | Row removal — women underrepresented     | `perturbation.row_removal_fraction`   |
| D1B | Row removal — men underrepresented       | same                                  |
| D2A | MAR — missingness injected for women     | `perturbation.mar_*`                  |
| D2B | MAR — missingness injected for men       | same                                  |
| D3A | Noise — Gaussian noise for women         | `perturbation.noise_*`                |
| D3B | Noise — Gaussian noise for men           | same                                  |

D1A and D1B are both run by default: the symmetric result
(whichever group is underrepresented suffers in model performance)
demonstrates that the problem is structural rather than group-specific.

Toggle variants in `config.py`:
```python
cfg.run_dataset_3a = True   # enable optional noise variants
cfg.run_dataset_3b = True
```

---

## Tuning the analysis

All parameters live in `config.py`.  Key knobs:

```python
cfg.perturbation.row_removal_fraction = 0.80   # 80% of target group removed
cfg.perturbation.mar_n_columns        = 5      # N columns blanked out
cfg.perturbation.mar_missing_fraction = 0.60   # 60% of group rows affected
cfg.model.imputation_strategy         = "median"  # "mean" | "median" | "knn"
cfg.sweep.run_sweep                   = True    # appendix parameter sweep
```

---

## Extending the pipeline

**Add a model:**  insert one entry in `models.MODEL_REGISTRY` in `models.py`.

**Add a mitigation:**  write a function decorated with `@register("name")`
in `mitigation.py` following the existing pattern.

**Add a fairness metric:**  add to `per_group_metrics()` in `fairness.py`;
it will automatically appear in all reports.

**Add a dataset perturbation:**  write a function following the signature in
`perturbations.py` and call it from `build_all_datasets()`.

---

## Outputs

| File                                    | Description                              |
|-----------------------------------------|------------------------------------------|
| `outputs/tables/results_all.csv`        | Raw metric values for every cell         |
| `outputs/tables/table1_summary.csv/.tex`| Publication summary table                |
| `outputs/tables/table2_*.csv`           | Gap-metric pivot tables                  |
| `outputs/figures/fig1_heatmap_*.png`    | Baseline fairness gap heatmap            |
| `outputs/figures/fig2_mitigation_*.png` | Before/after mitigation bar charts       |
| `outputs/figures/fig3_group_*.png`      | Per-group performance bars               |
| `outputs/figures/fig4_sweep_*.png`      | Parameter sweep curves (appendix)        |
| `outputs/figures/fig5_full_grid_*.png`  | Full model × mitigation × dataset grid   |

---

## Mitigation strategies

| Key               | Method                                              |
|-------------------|-----------------------------------------------------|
| `none`            | Baseline — no correction                            |
| `reweighting`     | Inverse-frequency sample weights per (group, label) |
| `smote`           | Oversample minority sensitive group in training      |
| `fairness_penalty`| fairlearn ExponentiatedGradient + EqualizedOdds     |

---

## Future work ideas

- Extend to race/ethnicity or SES as sensitive attributes (drop-in: change
  `cfg.fairness.sensitive_column`)
- Apply to MIMIC-IV (see Mhasawade et al. 2021 in references)
- Test Noise perturbation variants (D3A/D3B) at different SNR levels
- Intersectional fairness (gender × age group × severity) via
  fairlearn's `MetricFrame` with multi-column sensitive features
- Calibration analysis per group (reliability diagrams)
- Confidence-interval reporting via bootstrap across evaluation cells

---

## References

- PhysioNet 2019 Challenge: https://physionet.org/content/challenge-2019/1.0.0/
- Fairlearn library: https://fairlearn.org
- Mhasawade et al. (2021) MIMIC-IV fairness: https://www.nature.com/articles/s41598-022-11012-2
- Chen et al. (2023) Fairness in clinical ML: https://dl.acm.org/doi/full/10.1145/3494672
- MultiFair: https://www.academia.edu/85010689/MultiFair_Multi_Group_Fairness_in_Machine_Learning
