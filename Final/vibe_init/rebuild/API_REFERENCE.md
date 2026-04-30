# Rebuild Module API Reference

This document describes the actual public API of the rebuild modules. Use this for writing notebooks or scripts that use the codebase.

## Data Loading (`data_loader.py`)

```python
from data_loader import (
    load_physionet_files,
    add_hours_until_sepsis,
    split_patients_by_status,
    get_rows_for_patients,
    get_patient_list,
    get_patient_sepsis_status,
    summarize_dataset,
)
```

### Key Functions

**`load_physionet_files(data_dir: Path) → pd.DataFrame`**
- Loads all PSV files from training sets
- Returns patient-hour level DataFrame
- Columns: patient_id, hour, vitals, labs, Gender, SepsisLabel, etc.

**`add_hours_until_sepsis(df: pd.DataFrame, keep_post_onset=True) → pd.DataFrame`**
- Computes hours_until_sepsis column
- For septic patients: h=0 at TRUE sepsis onset (with +6 hour offset)
- For non-septic patients: hours_until_sepsis = NaN
- Returns copy of df with new column

**`split_patients_by_status(df, n_train_patients, random_state=42, stratify_by_sepsis=True) → (train_pids, bootstrap_pids)`**
- Splits patient IDs into training and bootstrap pool
- Optionally stratifies by sepsis status
- Returns numpy arrays of patient IDs

**`get_rows_for_patients(df: pd.DataFrame, patient_ids: np.ndarray) → pd.DataFrame`**
- Filters DataFrame to rows belonging to specified patients
- Returns filtered DataFrame with reset index

---

## Bootstrap Resampling (`bootstrap.py`)

```python
from bootstrap import BootstrapResampler
```

### Key Class

**`BootstrapResampler`**

```python
resampler = BootstrapResampler(
    bootstrap_pool_patient_ids=np.ndarray,  # Patient IDs to sample from
    full_df=pd.DataFrame,                   # Full data (will be indexed by patient_id)
    n_iterations=100,                       # Number of bootstrap samples
    bootstrap_sample_size=50,               # Patients per sample
    random_state=42,
)

# Generate one iteration
sampled_pids, sample_df = resampler.generate_iteration(iteration_idx=0)

# Generate all at once
all_iterations = resampler.generate_all_iterations()  # List of (pids, df) tuples
```

**Key points:**
- Bootstrap operates at **patient level**, not row level
- Sampling **with replacement**
- Returns both sampled patient IDs and all their rows

---

## Models (`models.py`)

```python
from models import LogisticGLM, XGBoostModel, GRUModel, get_model
```

### Model Classes

All models inherit from `BaseModel` and implement:
- `fit(X, y)` → self
- `predict_proba(X)` → (n_samples, 2) array
- `predict(X)` → (n_samples,) binary predictions (threshold 0.5)

**`LogisticGLM(C=0.01, random_state=42)`**
- L1-regularized logistic regression (sparse GLM)
- Parameters:
  - `C`: Inverse regularization strength (smaller = more regularization)
  - `random_state`: Random seed
- Includes automatic imputation (median) and feature scaling

**`XGBoostModel(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)`**
- Gradient boosted trees (requires `pip install xgboost`)
- Parameters:
  - `n_estimators`: Number of boosting rounds
  - `max_depth`: Tree depth
  - `learning_rate`: Boosting step size
  - `random_state`: Random seed

**`GRUModel(hidden_size=128, num_layers=2, dropout=0.2, epochs=20, batch_size=32, learning_rate=5e-4, random_state=42)`**
- Recurrent neural network (requires PyTorch: `pip install torch`)
- Parameters:
  - `hidden_size`: GRU hidden dimension
  - `num_layers`: Stacked GRU layers
  - `dropout`: Dropout rate
  - `epochs`: Training epochs
  - `batch_size`: Batch size
  - `learning_rate`: Adam learning rate
  - `random_state`: Random seed

**`get_model(model_name: str, **kwargs) → BaseModel`**
- Factory function: `"glm"`, `"xgboost"`, or `"gru"`
- Returns unfitted model instance

---

## Training & Evaluation (`training.py`)

```python
from training import (
    BootstrapEvaluator,
    extract_Xy,
    train_model,
    evaluate_model,
    evaluate_model_per_group,
)
```

### Key Class

**`BootstrapEvaluator`**

```python
evaluator = BootstrapEvaluator(
    model=LogisticGLM(),                # Unfitted model
    train_df=pd.DataFrame,              # Training data
    label_column='SepsisLabel',         # Target column name
    patient_id_column='patient_id',     # Patient ID column (optional)
)

# Fit happens automatically on construction

# Evaluate on bootstrap sample
metrics = evaluator.evaluate_iteration(
    bootstrap_df=pd.DataFrame,          # Bootstrap sample DataFrame
    iteration_idx=0,                    # Iteration number (for logging)
    compute_per_group=True,             # Compute per-group metrics
    group_column='Gender',              # Column for grouping
)
# Returns dict: {'utility': float, 'auroc': float, 'recall': float, ...}

# Aggregate results
agg = evaluator.aggregate_bootstrap_metrics()
# Returns dict: {'utility': {'mean': float, 'std': float, 'median': float, ...}, ...}

# Get summary DataFrame
summary_df = evaluator.summary()  # One row per iteration
```

**Returns from `evaluate_iteration()`:**
```python
{
    'n_samples': int,
    'n_positive': int,
    'prevalence': float,
    'auroc': float,
    'accuracy': float,
    'precision': float,
    'recall': float,
    'f1': float,
    'utility': float,  # PhysioNet 2019 official utility
    'iteration': int,
    'per_group': {
        group_value: {
            'n_samples': int,
            'utility': float,
            'auroc': float,
            ...
        }
    }
}
```

### Key Functions

**`extract_Xy(df: pd.DataFrame, label_column: str) → (X, y)`**
- Extracts features and labels from DataFrame
- Excludes metadata columns automatically
- Returns (n_samples, n_features) array and (n_samples,) labels

**`train_model(model, train_df, label_column='target') → (fitted_model, feature_columns)`**
- Trains unfitted model on training data
- Returns fitted model and list of feature column names

**`evaluate_model(model, eval_df, label_column='SepsisLabel', patient_id_column='patient_id') → dict`**
- Evaluates fitted model on evaluation data
- Computes AUROC, recall, F1, **PhysioNet utility**
- Returns metrics dictionary

---

## Utility Function (`utility.py`)

```python
from utility import (
    physionet_utility,
    utility_curve,
    find_optimal_threshold,
    UtilityEvaluator,
)
```

### Key Function

**`physionet_utility(labels: np.ndarray, y_pred: np.ndarray, patient_ids: Optional[np.ndarray] = None) → float`**
- Computes **official PhysioNet 2019 Challenge utility score**
- **Inputs:**
  - `labels`: Raw SepsisLabel column (0/1 per patient-hour)
  - `y_pred`: Binary predictions (0/1)
  - `patient_ids`: Patient IDs for per-patient normalization (optional but recommended)
- **Returns:** Normalized utility score (-2 to +1)

**Key formula (piecewise linear with time windows):**
```
Time relative to true sepsis onset:
  [-12h, -6h]:  TP utility increases 0 → 1
  [-6h, +3h]:   TP utility decreases 1 → 0
  [+3h, ∞):     TP utility = -2 (too late)
  Any time:     FP utility = -0.05 (alarm fatigue)
  Any time:     FN utility = -2 (worst case)
  Any time:     TN utility = 0 (baseline)
```

**Normalization:**
```
Per-patient score = (observed - inaction) / (best - inaction)
Overall score = mean across patients
```

### Other Utility Functions

**`utility_curve(y_proba, labels, patient_ids, thresholds=np.linspace(0, 1, 101)) → pd.DataFrame`**
- Returns utility at each threshold
- Useful for threshold optimization

**`find_optimal_threshold(y_proba, labels, patient_ids) → (threshold, max_utility)`**
- Returns optimal decision threshold and corresponding utility

---

## Configuration (`config.py`)

```python
from config import Config, TrainingConfig, BootstrapConfig
```

(Not commonly used in notebooks; documented for completeness)

---

## Example Workflow

```python
from pathlib import Path
import numpy as np
from data_loader import load_physionet_files, add_hours_until_sepsis, split_patients_by_status, get_rows_for_patients
from bootstrap import BootstrapResampler
from models import LogisticGLM
from training import BootstrapEvaluator

# Load data
DATA_DIR = Path('../data/physionet_sepsis')
raw_df = load_physionet_files(DATA_DIR)
df = add_hours_until_sepsis(raw_df, keep_post_onset=True)

# Split patients
train_pids, boot_pids = split_patients_by_status(
    df, n_train_patients=100, random_state=42
)
train_df = get_rows_for_patients(df, train_pids)

# Create bootstrap resampler
resampler = BootstrapResampler(
    bootstrap_pool_patient_ids=boot_pids,
    full_df=df,
    n_iterations=10,
    bootstrap_sample_size=50,
    random_state=42
)

# Train model and evaluate
evaluator = BootstrapEvaluator(
    model=LogisticGLM(C=0.01),
    train_df=train_df,
    label_column='SepsisLabel',
    patient_id_column='patient_id'
)

# Evaluate on each bootstrap sample
for i in range(10):
    _, sample_df = resampler.generate_iteration(i)
    metrics = evaluator.evaluate_iteration(sample_df, i)
    print(f"Iteration {i}: Utility={metrics['utility']:.4f}")

# Aggregate
summary = evaluator.aggregate_bootstrap_metrics()
print(f"Mean utility: {summary['utility']['mean']:.4f} ± {summary['utility']['std']:.4f}")
```

---

## Common Pitfalls

1. **Don't use `hours_until_sepsis` for utility!** Use raw `SepsisLabel`
   ```python
   # ✗ WRONG
   utility = physionet_utility(df['hours_until_sepsis'], y_pred)
   
   # ✓ CORRECT
   utility = physionet_utility(df['SepsisLabel'], y_pred)
   ```

2. **Always pass `patient_ids` to utility function** for correct per-patient normalization
   ```python
   # ✓ RECOMMENDED
   utility = physionet_utility(labels, y_pred, patient_ids=pids)
   ```

3. **Use raw SepsisLabel in `label_column`**, not computed hours
   ```python
   # ✓ CORRECT
   evaluator = BootstrapEvaluator(..., label_column='SepsisLabel')
   ```

4. **Models expect 2D feature arrays**
   ```python
   X = df[features].values  # (n_samples, n_features) — correct
   y = df['SepsisLabel'].values  # (n_samples,) — correct
   model.fit(X, y)
   ```

---

## Return to Test Notebook

The corrected **`test_utility_models_minimal.ipynb`** now uses the correct API.

To run it:
```bash
cd /Users/vrose/ClaudeContainer/PUBH8475/Final/vibe_init
jupyter notebook rebuild/test_utility_models_minimal.ipynb
```

**Expected runtime:** 2-3 minutes for 12 evaluations (LogisticGLM + GRU, 2 training sizes, 3 iterations)
