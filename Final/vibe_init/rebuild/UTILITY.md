# PhysioNet 2019 Utility Function Evaluation

The rebuild includes integration of the PhysioNet 2019 Challenge utility function for evaluating sepsis prediction models beyond standard metrics like AUROC and recall.

## What is the Utility Function?

The utility function rewards **early, accurate sepsis detection** while penalizing false alarms and missed detections:

```
Utility Score = (Observed Utility - Inaction Utility) / (Best Utility - Inaction Utility)
```

### Per-Row Utility

For each prediction:
- **True Positive (TP)**: +1.0 at optimal time, ramps linearly based on how early you predict
  - 6 hours before onset: +1.0 (ideal)
  - At onset: +0.0
  - After onset: -0.05 (too late)
  
- **False Negative (FN)**: -2.0 penalty after optimal window (missing at-risk patient)

- **False Positive (FP)**: -0.05 per false alarm (alarm fatigue cost)

- **True Negative (TN)**: 0.0 (no cost for correct absence)

### Why This Matters

Unlike AUROC (which treats all thresholds equally), the utility function:
- **Prioritizes early detection**: Predicting 6 hours early is better than predicting 1 hour early
- **Penalizes delay**: Missing a septic patient gets progressively worse penalty
- **Accounts for alarm fatigue**: False alarms have a cost
- **Clinical realism**: Reflects the actual clinical value of early warning

## Using Utility in the Rebuild

### Basic Evaluation

```python
from utility import physionet_utility

hours_until_sepsis = df['hours_until_sepsis'].values  # NaN for non-septic
y_pred_binary = (y_proba >= 0.5).astype(int)

utility_score = physionet_utility(hours_until_sepsis, y_pred_binary)
print(f"Utility: {utility_score:.4f}")
```

### Finding Optimal Threshold

```python
from utility import find_optimal_threshold

optimal_threshold, max_utility = find_optimal_threshold(
    y_proba, 
    hours_until_sepsis,
    patient_ids=patient_ids  # Optional
)

print(f"Best threshold: {optimal_threshold:.2f}")
print(f"Max utility: {max_utility:.4f}")
```

### Per-Group Fairness

```python
from utility import evaluate_utility_per_group

per_group_utility = evaluate_utility_per_group(
    y_proba,
    hours_until_sepsis,
    group_column=gender,  # [0, 1, ...]
    patient_ids=patient_ids
)

# Example output:
# {0: 0.65, 1: 0.68}  # Group 0: 0.65, Group 1: 0.68
utility_gap = abs(per_group_utility[0] - per_group_utility[1])
```

### With BootstrapEvaluator

The utility function is automatically computed in bootstrap evaluation if `hours_until_sepsis_column` is available:

```python
from training import BootstrapEvaluator
from models import LogisticGLM

model = LogisticGLM()
evaluator = BootstrapEvaluator(
    model,
    train_df,
    label_column="SepsisLabel",
    hours_until_sepsis_column="hours_until_sepsis",  # ← Enable utility
    patient_id_column="patient_id",
)

# Evaluate on bootstrap samples
for i in range(n_iterations):
    pids, bootstrap_df = resampler.generate_iteration(i)
    metrics = evaluator.evaluate_iteration(bootstrap_df, i)
    # metrics includes "utility" key

# Aggregate
agg = evaluator.aggregate_bootstrap_metrics()
print(f"Utility: {agg['utility']['mean']:.4f} ± {agg['utility']['std']:.4f}")
```

## Classes and Functions

### `physionet_utility(hours_until_sepsis, y_pred_binary, patient_ids=None)`

Compute normalized utility score.

**Args:**
- `hours_until_sepsis`: Hours before sepsis onset (NaN for non-septic)
- `y_pred_binary`: Binary predictions
- `patient_ids`: Optional patient IDs for patient-level aggregation

**Returns:** Float utility score

**Example:**
```python
utility = physionet_utility(
    df['hours_until_sepsis'].values,
    (model.predict_proba(X)[:, 1] >= 0.5).astype(int),
    patient_ids=df['patient_id'].values
)
```

### `evaluate_utility_at_threshold(y_proba, hours_until_sepsis, threshold=0.5, patient_ids=None)`

Evaluate utility at a specific decision threshold.

**Args:**
- `y_proba`: Probability predictions
- `hours_until_sepsis`: Hours until sepsis
- `threshold`: Decision threshold (default 0.5)
- `patient_ids`: Optional patient IDs

**Returns:** Utility score at this threshold

**Example:**
```python
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    util = evaluate_utility_at_threshold(y_proba, hours, threshold)
    print(f"Threshold {threshold}: Utility {util:.4f}")
```

### `find_optimal_threshold(y_proba, hours_until_sepsis, patient_ids=None, thresholds=None)`

Find the threshold that maximizes utility.

**Args:**
- `y_proba`: Probability predictions
- `hours_until_sepsis`: Hours until sepsis
- `patient_ids`: Optional patient IDs
- `thresholds`: Thresholds to search (default: linspace 0.01-0.99)

**Returns:** (optimal_threshold, maximum_utility)

**Example:**
```python
best_thresh, best_util = find_optimal_threshold(y_proba, hours)
model_binary = (y_proba >= best_thresh).astype(int)
```

### `evaluate_utility_per_group(y_proba, hours_until_sepsis, group_column, patient_ids=None, threshold=0.5)`

Evaluate utility separately by group (gender, age bin, etc.).

**Args:**
- `y_proba`: Probability predictions
- `hours_until_sepsis`: Hours until sepsis
- `group_column`: Group membership values
- `patient_ids`: Optional patient IDs
- `threshold`: Decision threshold

**Returns:** {group_value: utility_score}

**Example:**
```python
per_group = evaluate_utility_per_group(
    y_proba, 
    hours,
    gender_column,
    threshold=0.5
)
# {0: 0.65, 1: 0.68}  (Female: 0.65, Male: 0.68)
```

### `UtilityEvaluator(y_proba, hours_until_sepsis, patient_ids=None, y_true=None, group_column=None)`

Main class for comprehensive utility evaluation.

**Methods:**
- `utility_at_threshold(threshold=0.5)` → float
- `find_optimal_threshold(thresholds=None)` → (threshold, utility)
- `utility_per_group(threshold=0.5)` → {group: utility}
- `utility_curve(thresholds=None)` → DataFrame
- `summary()` → Dict with utility_at_0.5, optimal_threshold, max_utility

**Example:**
```python
evaluator = UtilityEvaluator(
    y_proba,
    hours_until_sepsis,
    patient_ids=patient_ids,
    group_column=gender
)

# Get utility at default 0.5 threshold
util_0p5 = evaluator.utility_at_threshold(0.5)

# Find best threshold
best_thresh, best_util = evaluator.find_optimal_threshold()

# Utility curve (for plotting)
curve_df = evaluator.utility_curve()

# Per-group utilities
per_group = evaluator.utility_per_group(threshold=best_thresh)

# All at once
summary = evaluator.summary()
```

## Integration with Training

The `evaluate_model()` and `BootstrapEvaluator` classes automatically compute utility if `hours_until_sepsis_column` is provided:

```python
from training import evaluate_model

metrics = evaluate_model(
    model,
    bootstrap_df,
    label_column="SepsisLabel",
    hours_until_sepsis_column="hours_until_sepsis",  # ← Required for utility
    patient_id_column="patient_id",
)

if "utility" in metrics:
    print(f"Utility: {metrics['utility']:.4f}")
```

## When Utility Isn't Available

If `hours_until_sepsis` isn't in your data:

1. **Compute it from SepsisLabel**: Find the row index where SepsisLabel first becomes 1
   ```python
   def compute_hours_until_sepsis(df):
       df = df.sort_values('hour')
       hours = []
       for patient_id in df['patient_id'].unique():
           p_df = df[df['patient_id'] == patient_id]
           sepsis_rows = p_df[p_df['SepsisLabel'] == 1]
           if len(sepsis_rows) > 0:
               onset_hour = sepsis_rows.iloc[0]['hour']
               p_df['hours_until_sepsis'] = onset_hour - p_df['hour']
           else:
               p_df['hours_until_sepsis'] = np.nan
       return df
   ```

2. **Use the timeseries data loader**: Load pre-processed data with `hours_until_sepsis` already computed

3. **Skip utility**: Just use standard metrics (AUROC, recall, etc.)

## Interpreting Utility Scores

| Utility Range | Interpretation |
|---------------|-----------------|
| 0.8–1.0 | Excellent: Near-optimal early detection |
| 0.6–0.8 | Good: Early detection with occasional false alarms |
| 0.4–0.6 | Moderate: Reasonable detection but room for improvement |
| 0.0–0.4 | Poor: Late detection or high false alarm rate |
| < 0.0 | Worse than inaction (predicting nothing would be better) |

## Fairness Through Utility

Utility enables fairness evaluation based on **clinical value**, not just statistical parity:

```python
# Compute utility per group
utils_female = evaluate_utility_per_group(..., group_column=gender, ...)
utils_male = utils_female[1]
utils_female = utils_female[0]

# Fairness gaps
print(f"Female utility: {utils_female:.4f}")
print(f"Male utility: {utils_male:.4f}")
print(f"Gap: {abs(utils_female - utils_male):.4f}")

# Do both groups get early warnings?
early_female = evaluate_detection_lead(...group=0...)
early_male = evaluate_detection_lead(..., group=1...)
print(f"Female detection lead: {early_female:.1f} hours before onset")
print(f"Male detection lead: {early_male:.1f} hours before onset")
```

## Threshold Optimization

Different clinical settings may prefer different thresholds:

```python
# Find thresholds optimizing different objectives
thresholds = np.linspace(0.01, 0.99, 99)

results = []
for thresh in thresholds:
    util = evaluate_utility_at_threshold(y_proba, hours, thresh)
    recall = recall_at_threshold(y_true, y_proba, thresh)
    fpr = fpr_at_threshold(y_true, y_proba, thresh)
    
    results.append({
        'threshold': thresh,
        'utility': util,
        'recall': recall,
        'fpr': fpr
    })

df_thresh = pd.DataFrame(results)

# Plot threshold-performance curve
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(df_thresh['threshold'], df_thresh['utility'])
axes[0].set_title('Utility vs Threshold')
axes[0].set_xlabel('Threshold')
axes[0].set_ylabel('Utility')

axes[1].plot(df_thresh['threshold'], df_thresh['recall'])
axes[1].set_title('Recall vs Threshold')

axes[2].plot(df_thresh['threshold'], df_thresh['fpr'])
axes[2].set_title('False Positive Rate vs Threshold')

plt.tight_layout()
plt.savefig('threshold_analysis.png')
```

## Example: Complete Workflow

```python
from training import BootstrapEvaluator
from models import LogisticGLM
from utility import find_optimal_threshold, UtilityEvaluator
from data_loader import load_physionet_files, split_patients_by_status, get_rows_for_patients
from bootstrap import BootstrapResampler

# 1. Load and split data
df = load_physionet_files(Path("data/physionet_sepsis"))
train_pids, bootstrap_pids = split_patients_by_status(df, 100)
train_df = get_rows_for_patients(df, train_pids)

# 2. Create bootstrap resampler
resampler = BootstrapResampler(bootstrap_pids, df, n_iterations=100, 
                               bootstrap_sample_size=50)

# 3. Train and evaluate
model = LogisticGLM()
evaluator = BootstrapEvaluator(
    model, train_df,
    hours_until_sepsis_column="hours_until_sepsis"
)

# 4. Bootstrap evaluation
for i in range(100):
    pids, bootstrap_df = resampler.generate_iteration(i)
    evaluator.evaluate_iteration(bootstrap_df, i)

# 5. Aggregate results
agg = evaluator.aggregate_bootstrap_metrics()

print("Standard Metrics:")
print(f"  AUROC: {agg['auroc']['mean']:.4f} ± {agg['auroc']['std']:.4f}")
print(f"  Recall: {agg['recall']['mean']:.4f} ± {agg['recall']['std']:.4f}")

print("\nUtility Metrics:")
print(f"  Utility: {agg['utility']['mean']:.4f} ± {agg['utility']['std']:.4f}")
print(f"  Optimal threshold: {agg['utility']['max']:.4f}")
```

## Notes

- **Patient-level aggregation**: Utility can be computed at row level or patient level. Patient-level (using `patient_ids`) better reflects clinical practice where we make one decision per patient.

- **Pre-onset data**: The PhysioNet challenge uses pre-onset data only (`hours_until_sepsis >= 0`), which simplifies the utility calculation.

- **Threshold dependence**: Utility depends critically on the decision threshold. Always explore threshold optimization, not just 0.5.

- **Group fairness**: Check utility per demographic group to ensure your model provides equal value to all populations.

## References

- PhysioNet 2019 Challenge: https://physionet.org/content/challenge-2019/
- Challenge utility function: https://github.com/physionet/python-challenge/blob/master/evaluate_sepsis_score.py
- Reyna et al. (2019, Critical Care Medicine): "Early prediction of sepsis from clinical data"
