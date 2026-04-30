# Utility Function Integration Summary

## What Was Added

Complete integration of the **PhysioNet 2019 Challenge utility function** into the bootstrap evaluation framework.

## New Files

### Core Modules
1. **utility.py** (~300 lines)
   - `physionet_utility()` — Main utility computation function
   - `evaluate_utility_at_threshold()` — Evaluate at specific threshold
   - `find_optimal_threshold()` — Search for best threshold
   - `evaluate_utility_per_group()` — Per-group (fairness) evaluation
   - `UtilityEvaluator` class — Comprehensive utility evaluation

### Documentation
2. **UTILITY.md** (~450 lines)
   - Complete utility function documentation
   - Usage examples
   - Integration with bootstrap
   - Threshold optimization
   - Fairness evaluation

3. **UTILITY_INTEGRATION.md** (this file)
   - Summary of changes
   - How to use
   - Notebook guide

### Testing
4. **test_utility_evaluation.py** (~190 lines)
   - End-to-end utility evaluation example
   - Bootstrap integration test
   - Per-group fairness evaluation demo

5. **test_utility_at_scales.ipynb** (Jupyter notebook)
   - Comprehensive notebook for testing utility at different scales
   - Tests training sizes: 50, 100, 200, 300
   - Tests bootstrap sizes: 25, 50, 100
   - Visualizations and statistical analysis

## Changes to Existing Modules

### training.py
- Added `Optional` import for type hints
- Imported `physionet_utility` and `evaluate_utility_per_group` from utility module
- Extended `evaluate_model()` to compute utility if `hours_until_sepsis_column` available
- Extended `evaluate_model_per_group()` to include utility metrics
- Updated `BootstrapEvaluator.__init__()` to accept `hours_until_sepsis_column` and `patient_id_column`
- Updated `evaluate_iteration()` to compute utility
- Updated `aggregate_bootstrap_metrics()` to include utility in aggregation

## How to Use

### Quick Start: Evaluate Utility in Bootstrap

```python
from training import BootstrapEvaluator
from models import LogisticGLM

# Initialize evaluator (model is fit automatically)
evaluator = BootstrapEvaluator(
    LogisticGLM(),
    train_df,
    label_column="SepsisLabel",
    hours_until_sepsis_column="hours_until_sepsis",  # Required for utility
    patient_id_column="patient_id",
)

# Evaluate on bootstrap samples
for i in range(100):
    pids, bootstrap_df = resampler.generate_iteration(i)
    metrics = evaluator.evaluate_iteration(bootstrap_df, i)
    # metrics includes "utility" key

# Aggregate
agg = evaluator.aggregate_bootstrap_metrics()
print(f"Utility: {agg['utility']['mean']:.4f} ± {agg['utility']['std']:.4f}")
```

### Threshold Optimization

```python
from utility import find_optimal_threshold

best_threshold, max_utility = find_optimal_threshold(y_proba, hours_until_sepsis)
print(f"Optimal threshold: {best_threshold:.3f}")
print(f"Maximum utility: {max_utility:.4f}")
```

### Per-Group Fairness Evaluation

```python
from utility import evaluate_utility_per_group

# Compute utility separately by gender
utility_by_gender = evaluate_utility_per_group(
    y_proba, 
    hours_until_sepsis,
    group_column=gender_values,
    threshold=0.5
)

print(f"Female utility: {utility_by_gender[0]:.4f}")
print(f"Male utility: {utility_by_gender[1]:.4f}")
print(f"Fairness gap: {abs(utility_by_gender[0] - utility_by_gender[1]):.4f}")
```

## Running the Notebook

### Option 1: Jupyter Lab/Notebook
```bash
jupyter notebook rebuild/test_utility_at_scales.ipynb
```

### Option 2: VS Code
Open the notebook file directly in VS Code and run cells interactively.

### What the Notebook Does

1. **Loads data** — PhysioNet sepsis dataset (20K patients)

2. **Tests configurations** — Matrix of:
   - Training sizes: 50, 100, 200, 300
   - Bootstrap sizes: 25, 50, 100
   - Iterations: 10 per config (fast for testing, use 100+ in production)

3. **Trains models** — Uses LogisticGLM for speed

4. **Evaluates metrics**:
   - AUROC (diagnostic accuracy)
   - Recall (true positive rate)
   - Accuracy
   - Utility (clinical value)

5. **Produces visualizations**:
   - Line plots: Performance vs training size
   - Heatmaps: Training size × Bootstrap size
   - Box plots: Distribution of metrics
   - PNG files saved to disk

6. **Statistical analysis**:
   - Scaling trends (how performance changes with size)
   - Stability analysis (variation across configurations)
   - Recommendations for production vs testing

## Important Notes

### hours_until_sepsis Column

The utility function requires `hours_until_sepsis` (hours before sepsis onset).

**In raw PhysioNet data**: This column is NOT present. You have two options:

**Option A: Compute from SepsisLabel**
```python
def compute_hours_until_sepsis(df):
    """Compute hours before sepsis onset from SepsisLabel."""
    hours = []
    for patient_id in df['patient_id'].unique():
        p_df = df[df['patient_id'] == patient_id]
        # Find first row where SepsisLabel = 1
        sepsis_rows = p_df[p_df['SepsisLabel'] == 1]
        if len(sepsis_rows) > 0:
            onset_idx = sepsis_rows.index[0]
            # Assign hours_until_sepsis as distance from this row
            p_df_copy = p_df.copy()
            p_df_copy['hours_until_sepsis'] = onset_idx - p_df_copy.index
            p_df_copy.loc[p_df_copy.index >= onset_idx, 'hours_until_sepsis'] = np.nan
            hours.append(p_df_copy)
    return pd.concat(hours, ignore_index=True)
```

**Option B: Use timeseries data loader**
The existing `data_loader_timeseries.py` in the main codebase computes this automatically.

### Utility Scores in Test Output

If you see `utility: NaN`, this means `hours_until_sepsis` wasn't available. The code gracefully handles this by:
- Not crashing
- Returning `np.nan` for utility
- Continuing with other metrics (AUROC, recall, etc.)

### Performance Expectations

With the raw PhysioNet data (no hours_until_sepsis):
- AUROC: 0.61–0.78 (depending on training size)
- Recall: 0.39–0.62 (identifies ~40-60% of at-risk patients)
- Utility: NaN (not computable without hours_until_sepsis)

## Comparison: With/Without Utility

| Aspect | Without Utility | With Utility |
|--------|-----------------|--------------|
| Metrics | AUROC, Recall, Accuracy | All above + clinical reward |
| Focus | Statistical accuracy | Clinical value |
| Threshold | Fixed at 0.5 | Can optimize |
| Fairness | Equal accuracy | Equal clinical value |
| Early detection | Not prioritized | Explicitly rewarded |
| False alarms | Not penalized | Penalized (-0.05 each) |

## Files Modified Summary

```
rebuild/
├── utility.py                      [NEW] Main utility module
├── training.py                     [MODIFIED] Added utility computation
├── test_utility_evaluation.py      [NEW] Test script
├── test_utility_at_scales.ipynb    [NEW] Jupyter notebook
├── UTILITY.md                      [NEW] Comprehensive documentation
└── UTILITY_INTEGRATION.md          [NEW] This file
```

## Next Steps

1. **Run the test script**:
   ```bash
   python rebuild/test_utility_evaluation.py
   ```

2. **Explore the notebook**:
   ```bash
   jupyter notebook rebuild/test_utility_at_scales.ipynb
   ```

3. **Compute hours_until_sepsis** for your data if needed (see "Important Notes" above)

4. **Optimize thresholds** using `find_optimal_threshold()` for your specific clinical setting

5. **Evaluate fairness** using `evaluate_utility_per_group()` across demographics

6. **Implement mitigation strategies** to improve fairness gaps if detected

## References

- PhysioNet 2019 Challenge: https://physionet.org/content/challenge-2019/
- Official utility function: https://github.com/physionet/python-challenge/blob/master/evaluate_sepsis_score.py
- Challenge paper: Reyna et al. (2019, Critical Care Medicine)

## Questions?

See:
- **UTILITY.md** for detailed documentation
- **MODELS.md** for model documentation
- **TRAINING.md** for training utilities
- **README.md** for overall architecture
