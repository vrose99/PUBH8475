# Fundamental Bootstrap Rebuild

This directory contains a clean, fundamental implementation of the bootstrap mechanism for PhysioNet sepsis prediction, including model fitting and evaluation.

## Overview

The rebuild demonstrates a complete pipeline:

1. **Data Loading** — Load PhysioNet PSV files and extract features
2. **Patient-Level Splitting** — Training set vs. bootstrap pool (no data leakage)
3. **Model Fitting** — Fit a model once on training data
4. **Bootstrap Evaluation** — Evaluate the fitted model on multiple bootstrap samples
5. **Results Aggregation** — Compute mean, std, median metrics across iterations

## Data Structure

Each observation is a **patient-hour** (one row per patient per ICU hour). The key grouping unit is `patient_id`.

- **Full dataset**: 20,336 patients × ~790K patient-hours
- **Training set**: 100 patients × ~4K patient-hours (configurable)
- **Bootstrap pool**: ~20K patients × ~786K patient-hours (rest of data)

### Labels

- `SepsisLabel`: Raw sepsis indicator from PhysioNet dataset
- `target`: Binary early-detection label (1 if sepsis within lookahead_hours)

## Quick Start

### 1. Basic Setup Test
```bash
python rebuild/main.py
```
Output: Patient split summary, bootstrap pool statistics, example iteration.

### 2. Full Model Fit & Evaluation
```bash
python rebuild/test_full_pipeline.py
```
Output: Train GLM model, evaluate on 5 bootstrap samples, aggregate results.

### 3. Compare Multiple Models
```bash
python rebuild/fit_and_evaluate.py
```
Output: Fit GLM and XGBoost, evaluate each on 20 bootstrap samples, compare metrics.

## Configuration

Edit configuration in `config.py`:

```python
from config import Config, TrainingConfig, BootstrapConfig

cfg = Config(
    data_dir=Path("data/physionet_sepsis"),
    training=TrainingConfig(
        n_patients=100,              # Patients for model training
        random_state=42,
        stratify_by_sepsis=True,     # Maintain sepsis rate in split
    ),
    bootstrap=BootstrapConfig(
        n_iterations=100,            # Bootstrap samples
        bootstrap_sample_size=50,    # Patients per sample (with replacement)
        random_state=42,
    ),
)
```

## Modules

### Core Data & Bootstrap

| Module | Contents |
|--------|----------|
| `config.py` | Configuration classes (TrainingConfig, BootstrapConfig) |
| `data_loader.py` | PhysioNet loading, patient-level splitting |
| `bootstrap.py` | Bootstrap resampling (BootstrapResampler class) |

### Models & Training

| Module | Contents |
|--------|----------|
| `models.py` | Three model classes: LogisticGLM, XGBoostModel, GRUModel |
| `training.py` | Training utilities and BootstrapEvaluator class |

### Examples & Tests

| Module | Contents |
|--------|----------|
| `main.py` | Data loading and bootstrap setup only |
| `fit_and_evaluate.py` | Complete workflow: train + evaluate all models |
| `test_full_pipeline.py` | Test pipeline with GLM (fastest) |
| `example_usage.py` | Code patterns and usage examples |

## Models

Three model classes for sepsis risk prediction:

### LogisticGLM
- L1-regularized logistic regression
- **Best for**: Interpretability, sparse features
- **Speed**: Fast training and inference
- **Dependencies**: sklearn only

```python
from models import LogisticGLM
model = LogisticGLM(C=0.01)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)
```

### XGBoostModel
- Gradient boosted decision trees
- **Best for**: Predictive accuracy
- **Speed**: Medium
- **Dependencies**: `pip install xgboost`

```python
from models import XGBoostModel
model = XGBoostModel(n_estimators=300, max_depth=4)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)
```

### GRUModel
- Recurrent neural network (GRU)
- **Best for**: Deep learning baseline
- **Speed**: Slow on CPU, fast on GPU
- **Dependencies**: `pip install torch`

```python
from models import GRUModel
model = GRUModel(hidden_size=128, epochs=20)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)
```

See [MODELS.md](MODELS.md) for detailed model documentation.

## Training & Evaluation

The `BootstrapEvaluator` class handles training and bootstrapping:

```python
from models import LogisticGLM
from training import BootstrapEvaluator

# Initialize (automatically fits on training data)
model = LogisticGLM()
evaluator = BootstrapEvaluator(model, train_df, label_column="SepsisLabel")

# Evaluate on each bootstrap sample
for i in range(n_iterations):
    pids, bootstrap_df = resampler.generate_iteration(i)
    evaluator.evaluate_iteration(bootstrap_df, i)

# Aggregate results
agg = evaluator.aggregate_bootstrap_metrics()
summary = evaluator.summary()  # DataFrame with one row per iteration
```

See [TRAINING.md](TRAINING.md) for detailed training documentation.

## Complete Workflow Example

```python
from pathlib import Path
from config import Config, TrainingConfig, BootstrapConfig
from data_loader import load_physionet_files, split_patients_by_status, get_rows_for_patients
from bootstrap import BootstrapResampler
from models import LogisticGLM
from training import BootstrapEvaluator

# 1. Configuration
cfg = Config(
    training=TrainingConfig(n_patients=100),
    bootstrap=BootstrapConfig(n_iterations=50, bootstrap_sample_size=100),
)

# 2. Load and split data
df = load_physionet_files(cfg.data_dir)
train_pids, bootstrap_pids = split_patients_by_status(df, cfg.training.n_patients)
train_df = get_rows_for_patients(df, train_pids)

# 3. Initialize bootstrap
resampler = BootstrapResampler(
    bootstrap_pool_patient_ids=bootstrap_pids,
    full_df=df,
    n_iterations=cfg.bootstrap.n_iterations,
    bootstrap_sample_size=cfg.bootstrap.bootstrap_sample_size,
)

# 4. Train and evaluate
model = LogisticGLM()
evaluator = BootstrapEvaluator(model, train_df, label_column="SepsisLabel")

for i in range(cfg.bootstrap.n_iterations):
    pids, bootstrap_df = resampler.generate_iteration(i)
    evaluator.evaluate_iteration(bootstrap_df, i)

# 5. Results
agg = evaluator.aggregate_bootstrap_metrics()
print(f"AUROC: {agg['auroc']['mean']:.4f} ± {agg['auroc']['std']:.4f}")
print(f"Recall: {agg['recall']['mean']:.4f} ± {agg['recall']['std']:.4f}")
```

## Bootstrap Mechanics

### One Bootstrap Iteration

If `bootstrap_sample_size=50`:

1. Sample 50 patient IDs **with replacement** from the pool
2. Fetch all rows (patient-hours) for those 50 patients
3. Evaluate the fitted model on this sample

Result: ~1–2K patient-hours, varying by patient length.

### Why Patient-Level Resampling?

- **No data leakage**: Each patient's complete time series is together
- **Temporal integrity**: All hours from a patient preserved
- **Clinical realism**: Decisions about whole patients
- **Stratification**: Bootstrap maintains sepsis rate

### Expected Duplicates

With `bootstrap_sample_size=50` and sampling with replacement:
- Exactly 50 slots sampled
- Expected unique patients: 50 × (1 - 1/e) ≈ **31.6**
- Some patients sampled 0–3+ times, creating variation

## Output Metrics

Each evaluation returns:

- **AUROC**: Area under ROC curve (threshold-independent, 0–1)
- **Accuracy**: Fraction correct at 0.5 threshold
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN) — **critical for early detection**
- **F1**: Harmonic mean of precision and recall
- **Prevalence**: Fraction with sepsis in sample
- **n_samples, n_positive**: Sample size breakdown

Aggregated across iterations:
- `mean`, `std`, `median`: Central tendency and spread
- `min`, `max`: Range of values
- `n_iterations`: Count of valid iterations

## Files

```
rebuild/
├── config.py              # Configuration
├── data_loader.py         # Data loading & splitting
├── bootstrap.py           # Bootstrap resampling
├── models.py              # Three model classes
├── training.py            # Training & evaluation
├── main.py                # Basic setup demo
├── fit_and_evaluate.py    # Full workflow (all models)
├── test_full_pipeline.py  # Test with GLM
├── test_models.py         # Quick model test
├── example_usage.py       # Usage patterns
├── README.md              # This file
├── MODELS.md              # Model documentation
├── TRAINING.md            # Training documentation
└── __init__.py            # Package init
```

## Key Parameters

| Setting | Default | Meaning |
|---------|---------|---------|
| `n_patients` (training) | 100 | Patients used to fit model |
| `bootstrap_sample_size` | 50 | Patients per iteration (with replacement) |
| `n_iterations` | 100 | Number of bootstrap samples |
| `random_state` | 42 | Reproducibility seed |
| `stratify_by_sepsis` | True | Maintain sepsis rate in splits |

## Performance Notes

Training times (approximate, single CPU):
- GLM: 1–3 seconds for typical training set
- XGBoost: 5–20 seconds
- GRU: 30–120 seconds (CPU-bound)

Bootstrap evaluation (50 iterations):
- ~5 minutes total per model
- ~6 seconds per iteration (model evaluation only)

## Troubleshooting

### XGBoost not available
```
RuntimeError: XGBoost not available. Install with: pip install xgboost
```
Solution: Use GLM or install xgboost.

### PyTorch not available
```
ImportError: PyTorch required for GRU model.
```
Solution: Use GLM/XGBoost or install torch.

### Memory issues on large bootstrap samples
```python
# Reduce bootstrap_sample_size or n_iterations
cfg.bootstrap.bootstrap_sample_size = 25  # Fewer patients per sample
cfg.bootstrap.n_iterations = 50           # Fewer iterations
```

## Next Steps

### Fairness Evaluation
Add per-group metrics (by Gender, Age bins, etc.):
```python
per_group = evaluator.evaluate_model_per_group(
    eval_df, group_column="Gender"
)
```

### Mitigation Strategies
Test different approaches to improve fairness while maintaining accuracy.

### Threshold Optimization
Instead of fixed 0.5 threshold, optimize for clinical utility:
```python
from training import compute_utility_score
```

## Code Quality

- **Modular**: Clear separation of data, models, training
- **Transparent**: Logging at each step
- **Type hints**: For IDE support and documentation
- **Minimal dependencies**: Core uses only numpy/pandas/sklearn
- **Reproducible**: Random seeds throughout
- **Well-documented**: See MODELS.md and TRAINING.md
