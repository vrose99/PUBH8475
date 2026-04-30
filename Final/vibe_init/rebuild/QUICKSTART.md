# Quick Start Guide

## 30 Second Overview

```bash
cd /path/to/vibe_init
python rebuild/test_full_pipeline.py
```

This runs a complete pipeline:
1. Load PhysioNet data (20K patients)
2. Split: 50 for training, rest for bootstrap
3. Fit logistic regression model (GLM)
4. Evaluate on 5 bootstrap samples
5. Print aggregated metrics

Expected output:
```
AUROC: 0.7894 ± 0.0684
Recall: 0.6293 ± 0.2023
Accuracy: 0.8208 ± 0.0630
F1: 0.1142 ± 0.0381
```

## 5 Minute Setup

### 1. Verify data is loaded
```bash
python rebuild/main.py
```
Output should show patient counts and bootstrap initialization.

### 2. Run full pipeline with GLM
```bash
python rebuild/test_full_pipeline.py
```
Fits model and evaluates on 5 bootstrap samples.

### 3. Try all models (if dependencies installed)
```bash
pip install xgboost torch
python rebuild/fit_and_evaluate.py
```
Fits GLM, XGBoost, and GRU, compares results.

## 30 Minute Deep Dive

### See what the rebuild contains
```bash
ls -lh rebuild/
```

Core modules:
- `config.py` — Configuration
- `data_loader.py` — Data loading
- `bootstrap.py` — Bootstrap resampling
- `models.py` — Three model classes
- `training.py` — Training & evaluation

### Understand the data flow
```bash
python
```

```python
from pathlib import Path
from rebuild.config import Config, TrainingConfig, BootstrapConfig
from rebuild.data_loader import load_physionet_files, split_patients_by_status
from rebuild.bootstrap import BootstrapResampler

cfg = Config(
    training=TrainingConfig(n_patients=100),
    bootstrap=BootstrapConfig(n_iterations=50, bootstrap_sample_size=50),
)

# Load
df = load_physionet_files(cfg.data_dir)
print(f"Full dataset: {df['patient_id'].nunique()} patients")

# Split
train_pids, bootstrap_pids = split_patients_by_status(df, 100)
print(f"Training: {len(train_pids)} patients")
print(f"Bootstrap pool: {len(bootstrap_pids)} patients")

# Bootstrap resampler
resampler = BootstrapResampler(bootstrap_pids, df, n_iterations=50, bootstrap_sample_size=50)
pids, sample_df = resampler.generate_iteration(0)
print(f"Bootstrap sample 0: {len(set(pids))} unique patients, {len(sample_df)} rows")
```

### Fit a model and evaluate
```python
from rebuild.models import LogisticGLM
from rebuild.training import BootstrapEvaluator, extract_Xy

# Extract training data
train_df = df[df['patient_id'].isin(train_pids)]

# Train model (happens automatically in BootstrapEvaluator.__init__)
model = LogisticGLM()
evaluator = BootstrapEvaluator(model, train_df, label_column="SepsisLabel")

# Evaluate on one bootstrap sample
pids, bootstrap_df = resampler.generate_iteration(0)
metrics = evaluator.evaluate_iteration(bootstrap_df, iteration_idx=0)
print(f"AUROC: {metrics['auroc']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
```

## Common Tasks

### Change training set size
Edit `fit_and_evaluate.py` or create your own script:

```python
cfg = Config(
    training=TrainingConfig(n_patients=200),  # ← Change here
    bootstrap=BootstrapConfig(n_iterations=50, bootstrap_sample_size=100),
)
```

### Evaluate fairness by gender
```python
from rebuild.fairness_eval import aggregate_fairness_across_bootstrap, print_fairness_report

# During evaluation, enable per-group metrics:
evaluator.evaluate_iteration(
    bootstrap_df, 
    i, 
    compute_per_group=True,
    group_column="Gender"
)

# After all iterations:
agg_fairness = aggregate_fairness_across_bootstrap(evaluator)
print_fairness_report(agg_fairness, evaluator.aggregate_bootstrap_metrics())
```

### Compare GLM vs XGBoost
```bash
python rebuild/fit_and_evaluate.py
```

Output shows comparison table:
```
MODEL       AUROC              ACCURACY           RECALL
GLM         0.7894 ± 0.0684   0.8208 ± 0.0630   0.6293 ± 0.2023
XGBOOST     0.8120 ± 0.0650   0.8350 ± 0.0550   0.6500 ± 0.1900
```

### Export results to CSV
```python
from rebuild.training import BootstrapEvaluator

evaluator = BootstrapEvaluator(...)
# ... run iterations ...

df = evaluator.summary()  # Per-iteration DataFrame
df.to_csv("bootstrap_results.csv", index=False)

agg = evaluator.aggregate_bootstrap_metrics()
# Convert agg to DataFrame and save
```

### Increase bootstrap samples for stability
```python
cfg = Config(
    training=TrainingConfig(n_patients=100),
    bootstrap=BootstrapConfig(
        n_iterations=500,          # ← From 100
        bootstrap_sample_size=100,
    ),
)
```
Takes ~1 hour for GLM, ~3 hours for XGBoost.

## Documentation

- **README.md** — Full overview and architecture
- **MODELS.md** — Model classes (GLM, XGBoost, GRU), hyperparameters
- **TRAINING.md** — Training utilities, BootstrapEvaluator, metrics
- **SUMMARY.md** — Design decisions, data flow, performance notes
- **This file** — Quick start

## Troubleshooting

### "No PSV files found"
Check data directory:
```bash
ls data/physionet_sepsis/training_setA/training/ | head -5
```
Should show patient files like `p000001.psv`, `p000002.psv`.

### "XGBoost not available"
```bash
pip install xgboost
# On macOS, may need: brew install libomp
```

### "PyTorch required for GRU"
```bash
pip install torch
```

### "Model not fit" error
Always call fit before predict:
```python
model.fit(X_train, y_train)  # ← Required
model.predict(X_test)        # ← Now works
```

### Out of memory
Reduce bootstrap sample size:
```python
cfg.bootstrap.bootstrap_sample_size = 25  # From 50
```

## Next Steps

1. **Understand the modules**: Read README.md, MODELS.md, TRAINING.md
2. **Run examples**: `test_full_pipeline.py`, `fit_and_evaluate.py`
3. **Experiment**: Adjust config, try different training sizes
4. **Fairness**: Use `fairness_eval.py` to check per-group metrics
5. **Extend**: Add mitigation strategies, custom models, etc.

## File Structure

```
rebuild/
├── Core Modules
│   ├── config.py              Configuration
│   ├── data_loader.py         PhysioNet loading
│   ├── bootstrap.py           Bootstrap resampling
│   ├── models.py              GLM, XGBoost, GRU
│   ├── training.py            Training & evaluation
│   └── fairness_eval.py       Fairness metrics
│
├── Examples & Tests
│   ├── main.py                Basic demo
│   ├── fit_and_evaluate.py    Full workflow
│   ├── test_full_pipeline.py  Quick test
│   ├── example_usage.py       Usage patterns
│   └── fairness_eval.py       (runnable example)
│
└── Documentation
    ├── README.md              Main guide
    ├── MODELS.md              Model docs
    ├── TRAINING.md            Training docs
    ├── SUMMARY.md             Design notes
    └── QUICKSTART.md          This file
```

## Key Concepts

### Patient-Level Bootstrap
- Train set: fixed set of N patients
- Bootstrap pool: all remaining patients
- Each iteration: sample bootstrap_sample_size patient IDs with replacement
- Prevents data leakage: patient's hours stay together

### Models
- **LogisticGLM**: Sparse, interpretable, fast
- **XGBoostModel**: Accurate, handles interactions
- **GRUModel**: Deep learning baseline, slow

### Metrics
- **AUROC**: Threshold-independent performance (0–1, higher=better)
- **Recall**: True positive rate — critical for early detection
- **Precision**: Positive predictive value
- **F1**: Balance of precision/recall

## Tips

1. **Start small**: Test with n_patients=25, n_iterations=5 first
2. **Check fairness early**: Enable per_group=True to see gender gaps
3. **Compare models**: Always run multiple to see trade-offs
4. **Aggregate properly**: Bootstrap gives you uncertainty estimates
5. **Document your runs**: Save config and results together

## Getting Help

Read the docstrings:
```python
from rebuild.models import LogisticGLM
help(LogisticGLM)

from rebuild.training import BootstrapEvaluator
help(BootstrapEvaluator.evaluate_iteration)
```

Check examples:
```bash
python rebuild/example_usage.py
```

Inspect the source code (all well-commented).

---

**TL;DR**: Run `python rebuild/test_full_pipeline.py` to see it work end-to-end.
