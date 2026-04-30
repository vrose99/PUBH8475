# Training and Evaluation Utilities

The `training.py` module provides utilities for:
- Extracting features and labels from DataFrames
- Training models on training data
- Evaluating models on test/bootstrap data
- Aggregating bootstrap results across iterations

## Core Functions

### `get_feature_columns(df)`

Extract feature column names (exclude metadata).

```python
from training import get_feature_columns

feat_cols = get_feature_columns(df)
# Returns list of feature names, excludes: patient_id, hour, target, Gender, etc.
```

Excluded columns (metadata):
- `patient_id`, `hour`, `SepsisLabel`, `target`
- `hours_until_sepsis`, `is_censored`
- `Gender`, `Unit1`, `Unit2`, `Age`, `HospAdmTime`

### `extract_Xy(df, label_column="target")`

Extract features and labels from a DataFrame.

```python
from training import extract_Xy

X, y = extract_Xy(df, label_column="SepsisLabel")
# X shape: (n_samples, n_features)
# y shape: (n_samples,) — binary labels (0 or 1)
```

Returns:
- `X`: Numpy array of features (float32)
- `y`: Numpy array of binary labels (int)

Handles:
- DataFrame or numpy input
- Automatic type conversion
- Feature column selection

### `train_model(model, train_df, label_column="target")`

Train a model on training data.

```python
from models import LogisticGLM
from training import train_model

model = LogisticGLM()
model, feat_cols = train_model(model, train_df, label_column="SepsisLabel")

# model is now fit and ready to predict
# feat_cols is the list of feature column names used
```

Returns:
- `model`: Fitted model object
- `feat_cols`: List of feature column names

### `evaluate_model(model, eval_df, label_column="target")`

Evaluate a fitted model on a test/evaluation set.

```python
from training import evaluate_model

metrics = evaluate_model(model, test_df, label_column="SepsisLabel")
print(f"AUROC: {metrics['auroc']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

Returns dictionary with keys:
- `auroc`: Area under ROC curve
- `accuracy`: Fraction correct
- `precision`: True positives / (True positives + False positives)
- `recall`: True positives / (True positives + False negatives)
- `f1`: Harmonic mean of precision and recall
- `n_samples`: Size of evaluation set
- `n_positive`: Number of positive samples
- `prevalence`: Fraction positive

### `evaluate_model_per_group(model, eval_df, group_column="Gender", label_column="target")`

Evaluate model separately for each group (e.g., by demographic).

```python
from training import evaluate_model_per_group

per_group = evaluate_model_per_group(
    model, 
    test_df, 
    group_column="Gender",
    label_column="SepsisLabel"
)

# per_group = {
#   0: {auroc: 0.75, accuracy: 0.82, ...},  # Female
#   1: {auroc: 0.78, accuracy: 0.84, ...},  # Male
# }
```

Returns nested dict: `{group_value: {metric: value}}`

## BootstrapEvaluator Class

Main class for evaluating a model across multiple bootstrap samples.

### Initialization

```python
from models import LogisticGLM
from training import BootstrapEvaluator

model = LogisticGLM()
evaluator = BootstrapEvaluator(
    model=model,
    train_df=train_df,
    label_column="SepsisLabel"
)
# Model is automatically fit on train_df during initialization
```

### Evaluating Each Iteration

```python
for i in range(n_bootstrap_iterations):
    pids, bootstrap_df = resampler.generate_iteration(i)
    metrics = evaluator.evaluate_iteration(
        bootstrap_df=bootstrap_df,
        iteration_idx=i,
        compute_per_group=False,
        group_column="Gender"
    )
```

Parameters:
- `bootstrap_df`: Bootstrap sample DataFrame
- `iteration_idx`: Iteration number (for logging)
- `compute_per_group` (bool): Also compute per-group metrics
- `group_column` (str): Column to group by (if per_group=True)

### Aggregating Results

```python
# After evaluating all iterations
agg = evaluator.aggregate_bootstrap_metrics()

print(agg['auroc'])
# {
#   'mean': 0.7894,
#   'std': 0.0684,
#   'median': 0.7704,
#   'min': 0.7147,
#   'max': 0.9161,
#   'n_iterations': 20
# }
```

Returns dict with aggregation for each metric:
- `mean`: Mean across iterations
- `std`: Standard deviation
- `median`: Median value
- `min` / `max`: Range
- `n_iterations`: Number of valid iterations

### Getting Results as DataFrame

```python
df = evaluator.summary()
# Returns DataFrame with one row per iteration:
# Columns: n_samples, n_positive, prevalence, auroc, accuracy, 
#          precision, recall, f1, iteration

print(df[['iteration', 'auroc', 'recall']])
```

## Full Workflow Example

```python
from pathlib import Path
from models import LogisticGLM
from training import BootstrapEvaluator
from data_loader import load_physionet_files, split_patients_by_status, get_rows_for_patients
from bootstrap import BootstrapResampler

# 1. Load and split data
df = load_physionet_files(Path("data/physionet_sepsis"))
train_pids, bootstrap_pids = split_patients_by_status(df, n_train_patients=100)
train_df = get_rows_for_patients(df, train_pids)

# 2. Initialize bootstrap resampler
resampler = BootstrapResampler(
    bootstrap_pool_patient_ids=bootstrap_pids,
    full_df=df,
    n_iterations=100,
    bootstrap_sample_size=50,
)

# 3. Create evaluator (which fits the model)
model = LogisticGLM()
evaluator = BootstrapEvaluator(model, train_df, label_column="SepsisLabel")

# 4. Evaluate on each bootstrap sample
for i in range(resampler.n_iterations):
    pids, bootstrap_df = resampler.generate_iteration(i)
    evaluator.evaluate_iteration(bootstrap_df, i)

# 5. Aggregate and display results
agg = evaluator.aggregate_bootstrap_metrics()
summary = evaluator.summary()

print(f"AUROC: {agg['auroc']['mean']:.4f} ± {agg['auroc']['std']:.4f}")
print(f"Recall: {agg['recall']['mean']:.4f} ± {agg['recall']['std']:.4f}")
```

## Handling Missing Labels

If a DataFrame is missing the specified `label_column`, the code falls back to `SepsisLabel`:

```python
# These are equivalent if df has 'SepsisLabel' but not 'target'
extract_Xy(df, label_column="target")
extract_Xy(df, label_column="SepsisLabel")
```

## Handling Class Imbalance

Metrics are computed even when:
- Only one class is present (AUROC returns NaN)
- Zero positive samples (precision/recall = 0)
- Zero negative samples (all positive)

No special handling is done here; the model itself (via `class_weight="balanced"`) handles imbalance.

## Output Interpretation

### AUROC (Area Under ROC Curve)

- Range: 0–1
- Higher is better
- Threshold-independent: evaluates all decision boundaries
- 0.5 = random, 1.0 = perfect

### Accuracy

- Fraction of correct predictions (at threshold 0.5)
- Range: 0–1
- Can be misleading on imbalanced data

### Recall (Sensitivity)

- True positive rate: TP / (TP + FN)
- **Critical for early detection**: high recall = catch most at-risk patients
- Trade-off: increasing recall usually decreases precision

### Precision (Positive Predictive Value)

- True positive rate: TP / (TP + FP)
- Answers: "Of predicted sepsis, how many actually have it?"
- Trade-off: increasing precision usually decreases recall

### F1 Score

- Harmonic mean: 2 × (precision × recall) / (precision + recall)
- Balances precision and recall
- Often used when you care about both false positives and false negatives

## Comparison: Multiple Models

```python
from models import get_model

models_to_test = ["glm", "xgboost"]
results = {}

for model_name in models_to_test:
    model = get_model(model_name)
    evaluator = BootstrapEvaluator(model, train_df, label_column="SepsisLabel")
    
    for i in range(n_iterations):
        pids, bootstrap_df = resampler.generate_iteration(i)
        evaluator.evaluate_iteration(bootstrap_df, i)
    
    results[model_name] = evaluator.aggregate_bootstrap_metrics()

# Print comparison
for model_name in models_to_test:
    agg = results[model_name]
    print(f"{model_name}: AUROC={agg['auroc']['mean']:.4f}")
```

## Debugging

### Check feature columns
```python
from training import get_feature_columns
print(get_feature_columns(df))
```

### Check extracted data
```python
from training import extract_Xy
X, y = extract_Xy(df, "SepsisLabel")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X missing values: {np.isnan(X).sum()}")
print(f"Class balance: {(y==1).sum()} / {len(y)}")
```

### Check metrics for one iteration
```python
pids, bootstrap_df = resampler.generate_iteration(0)
metrics = evaluate_model(model, bootstrap_df, "SepsisLabel")
print(metrics)
```

## Performance

Evaluating typical sizes (patient-hour data):
- 1000 rows: ~100 ms
- 5000 rows: ~200 ms  
- 20000 rows: ~500 ms

(Assumes model is already fit; metric computation only)
