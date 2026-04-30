# Which Notebook Should I Run?

## Quick Answer

| Situation | Use This | Runtime |
|-----------|----------|---------|
| **Kernel keeps crashing** | `test_utility_models_quick.ipynb` | 30 sec |
| **Need working example** | `test_utility_models_quick.ipynb` | 30 sec |
| **Want LogisticGLM results** | `test_utility_at_scales.ipynb` | 5-10 min |
| **Want 2-3 model comparison** | `test_utility_models_minimal.ipynb` | 2-3 min |
| **Want full evaluation** | `test_utility_model_comparison.ipynb` | 10-15 min |

---

## Detailed Comparison

### `test_utility_models_quick.ipynb` ✅ START HERE
**Purpose:** Diagnose issues and verify pipeline works

**Configuration:**
- Model: **LogisticGLM only** (no PyTorch needed)
- Training sizes: [50, 100]
- Bootstrap sizes: [25]
- Iterations: **2 per config**
- Total: **4 evaluations**

**Runtime:** ~30-60 seconds

**Outputs:**
- Step-by-step diagnostic output
- Full error messages if anything fails
- 1 CSV file with results

**Use when:**
- Kernel crashes in other notebooks
- You want to verify the pipeline works
- You're debugging issues
- First-time running on a new computer

**Dependencies:** pandas, numpy, sklearn (no PyTorch, no XGBoost)

---

### `test_utility_at_scales.ipynb` ✅ RECOMMENDED FOR PRODUCTION
**Purpose:** Full evaluation of LogisticGLM model across training set sizes

**Configuration:**
- Model: **LogisticGLM only**
- Training sizes: [50, 100, 200, 300]
- Bootstrap sizes: [25, 50, 100]
- Iterations: **10 per config**
- Total: **120 evaluations**

**Runtime:** ~5-10 minutes (varies by CPU)

**Outputs:**
- 2 CSV files (detailed + summary)
- 4 PNG visualizations (metrics, distributions, heatmaps, fairness)
- Complete performance analysis

**Use when:**
- You want production-quality results for LogisticGLM
- You don't have PyTorch installed
- You want to see how utility scales with training size
- Creating figures for reports/presentations

**Dependencies:** pandas, numpy, sklearn, matplotlib, seaborn

---

### `test_utility_models_minimal.ipynb` ✅ FOR MULTI-MODEL COMPARISON
**Purpose:** Quick comparison of LogisticGLM vs GRU

**Configuration:**
- Models: **LogisticGLM + GRU** (no XGBoost, to avoid dependency)
- Training sizes: [50, 100]
- Bootstrap sizes: [25]
- Iterations: **3 per config**
- Total: **12 evaluations**

**Runtime:** ~2-3 minutes (varies by CPU; GRU is slower)

**Outputs:**
- 2 CSV files (detailed + summary)
- 3 PNG visualizations (metrics, distributions, fairness)
- Model comparison results

**Use when:**
- PyTorch is installed
- You want to compare LogisticGLM vs GRU
- You want a quick evaluation (not full production run)
- You want to see RNN performance on this dataset

**Dependencies:** pandas, numpy, sklearn, matplotlib, seaborn, PyTorch

**Install PyTorch:**
```bash
pip install torch
```

---

### `test_utility_model_comparison.ipynb` ✅ FULL EVALUATION
**Purpose:** Comprehensive evaluation of all three models

**Configuration:**
- Models: **LogisticGLM, XGBoost, GRU**
- Training sizes: [50, 100, 200, 300]
- Bootstrap sizes: [25, 50]
- Iterations: **5 per config**
- Total: **90 evaluations**

**Runtime:** 10-15 minutes (varies by CPU; GRU is slow)

**Outputs:**
- 2 CSV files (detailed + summary)
- 4 PNG visualizations
- Complete multi-model analysis

**Use when:**
- You have both PyTorch and XGBoost installed
- You want to compare all three models
- You want full production evaluation
- Creating comprehensive analysis for publication

**Install dependencies:**
```bash
pip install torch xgboost
```

---

## Decision Tree

```
Do you want to:

[1] Fix a kernel crash / Debug issues?
    → Run: test_utility_models_quick.ipynb

[2] Get production LogisticGLM results?
    → Run: test_utility_at_scales.ipynb

[3] Compare LogisticGLM vs GRU?
    → Install PyTorch: pip install torch
    → Run: test_utility_models_minimal.ipynb

[4] Compare all 3 models (LogisticGLM, XGBoost, GRU)?
    → Install PyTorch: pip install torch
    → Install XGBoost: pip install xgboost
    → Run: test_utility_model_comparison.ipynb
```

---

## Recommended Workflow

### Scenario 1: First Time / Just Verifying
1. Run `test_utility_models_quick.ipynb` (30 sec)
   - Verify the pipeline works
   - Check for errors in diagnostic output

### Scenario 2: Production Results (No GRU)
1. Run `test_utility_at_scales.ipynb` (5-10 min)
   - Get LogisticGLM results across training sizes
   - Generates publication-quality figures

### Scenario 3: Multi-Model Comparison (With GRU)
1. Install PyTorch: `pip install torch`
2. Run `test_utility_models_minimal.ipynb` (2-3 min)
   - Compare LogisticGLM vs GRU
   - Quick evaluation

3. (Optional) Run `test_utility_at_scales.ipynb` for full LogisticGLM results

### Scenario 4: Full Evaluation (All Models)
1. Install dependencies:
   ```bash
   pip install torch xgboost
   ```
2. Run `test_utility_model_comparison.ipynb` (10-15 min)
   - Compare all three models
   - Comprehensive analysis

---

## What Each Notebook Generates

| Notebook | CSVs | PNGs | Time |
|----------|------|------|------|
| quick | 1 | 0 | 30s |
| at_scales | 2 | 4 | 5-10m |
| minimal | 2 | 3 | 2-3m |
| comparison | 2 | 4 | 10-15m |

---

## File Outputs

### `test_utility_models_quick.ipynb`
- `utility_logisticglm_quick.csv` — Results for 4 evaluations

### `test_utility_at_scales.ipynb`
- `utility_scaling_results.csv` — Detailed results (120 rows)
- `utility_scaling_summary.csv` — Aggregated by training size
- `scaling_metrics.png` — Utility/AUROC vs training size
- `heatmap_utility_auroc.png` — 2D heatmap
- `utility_boxplots.png` — Distribution plots
- `utility_fairness_gap.png` — Gender fairness analysis

### `test_utility_models_minimal.ipynb`
- `utility_model_comparison_results_minimal.csv` — Detailed results (12 rows)
- `utility_model_comparison_summary_minimal.csv` — Aggregated
- `model_comparison_metrics_minimal.png` — Utility/AUROC trends
- `model_distributions_minimal.png` — Distribution plots
- `fairness_gender_gap_minimal.png` — Fairness gap

### `test_utility_model_comparison.ipynb`
- `utility_model_comparison_results.csv` — Detailed results (90 rows)
- `utility_model_comparison_summary.csv` — Aggregated
- `model_comparison_metrics.png` — Metrics by training size
- `model_distributions.png` — Utility/AUROC distributions
- (others as available)

---

## If You're Still Having Issues

1. **Start with:** `test_utility_models_quick.ipynb`
   - This has diagnostic output for every step
   - Will tell you exactly where it fails

2. **Report the error from:**
   - Step 6 or 7 of "SINGLE CONFIG TEST" section
   - This tells me which part of the pipeline is failing

3. **Then check:** `TROUBLESHOOTING.md`
   - Common issues and fixes
   - Environment setup

---

## Summary

- **Just verifying?** → `test_utility_models_quick.ipynb` (30 sec)
- **Want results?** → `test_utility_at_scales.ipynb` (5-10 min)
- **Want models compared?** → `test_utility_models_minimal.ipynb` (2-3 min)
- **Want everything?** → `test_utility_model_comparison.ipynb` (10-15 min)
