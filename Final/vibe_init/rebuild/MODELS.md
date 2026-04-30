# Models for Sepsis Risk Prediction

This module provides three model classes for predicting hourly sepsis risk from patient-hour data.

## Overview

All models:
- Fit on training data (rows are patient-hours)
- Predict binary sepsis risk (0 or 1) and probabilities
- Handle missing values via imputation
- Support sklearn-style `fit()`, `predict()`, `predict_proba()` interface

### Model Classes

| Model | Type | Best for | Notes |
|-------|------|----------|-------|
| `LogisticGLM` | Linear | Interpretability, sparse features | L1 regularization for feature selection |
| `XGBoostModel` | Tree ensemble | Predictive accuracy, feature interactions | Requires `xgboost` package |
| `GRUModel` | Neural network | Complex patterns, non-linear relationships | Requires PyTorch |

## LogisticGLM

L1-regularized logistic regression (sparse generalized linear model).

### What It Does
- Fits a logistic regression with L1 (LASSO) penalty for feature selection
- Automatically handles missing values (median imputation)
- Scales features for numerical stability
- Balances class imbalance via `class_weight="balanced"`

### Usage

```python
from models import LogisticGLM

model = LogisticGLM(C=0.01)  # C controls regularization strength
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)  # shape (n_samples, 2)
y_pred = model.predict(X_test)          # shape (n_samples,)
```

### Parameters

- `C` (float): Inverse regularization strength
  - Smaller C → more regularization (sparser model)
  - Default: 0.01 (moderate regularization)
- `random_state` (int): Random seed for reproducibility

### Hyperparameters (Fixed)

- Solver: `saga` (supports L1 penalty)
- Penalty: L1 (LASSO, feature selection)
- Max iterations: 5000
- Class weighting: Balanced

### Output

- Coefficients stored in `model.model.coef_` after fitting
- Non-zero coefficients indicate selected features
- Interpretable: coefficient magnitude = feature importance

### Example

```python
model = LogisticGLM(C=0.01)
model.fit(X_train, y_train)

# Check which features are selected
n_nonzero = (model.model.coef_ != 0).sum()
print(f"Selected {n_nonzero} features")

# Predict on test set
metrics = evaluate_model(model, test_df)
```

## XGBoostModel

Gradient boosted decision trees.

### What It Does
- Gradient boosting with multiple tree iterations
- Handles missing values via imputation
- No feature scaling needed (tree-based)
- Regularization via max depth and L1/L2 penalties

### Usage

```python
from models import XGBoostModel

model = XGBoostModel(n_estimators=300, max_depth=4, learning_rate=0.05)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)
```

### Parameters

- `n_estimators` (int): Number of boosting rounds. Default: 300
- `max_depth` (int): Maximum tree depth. Default: 4
- `learning_rate` (float): Boosting learning rate. Default: 0.05
- `random_state` (int): Random seed

### Hyperparameters (Fixed)

- Subsampling: 0.8 (reduce overfitting)
- Column subsampling: 0.8 per tree and per level
- L1/L2 regularization: 1.0 each
- Objective: Binary logistic loss
- Eval metric: AUC

### Output

- Feature importances can be extracted
- Calibrated probabilities from logistic objective
- Fast predictions even on large datasets

### Example

```python
model = XGBoostModel()
model.fit(X_train, y_train)

# Feature importance
importances = model.model.feature_importances_

# Evaluate
metrics = evaluate_model(model, test_df)
print(f"AUROC: {metrics['auroc']:.4f}")
```

### Requirements

```bash
pip install xgboost
```

## GRUModel

Gated Recurrent Unit (GRU) neural network.

### What It Does
- Two-layer GRU with 128 hidden units per layer
- Dropout regularization (0.2)
- Sigmoid output for probability prediction
- Handles missing values via imputation and scaling

### Usage

```python
from models import GRUModel

model = GRUModel(
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    epochs=20,
    batch_size=32,
    learning_rate=5e-4
)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)
```

### Parameters

- `hidden_size` (int): GRU hidden dimension. Default: 128
- `num_layers` (int): Number of stacked GRU layers. Default: 2
- `dropout` (float): Dropout rate. Default: 0.2
- `epochs` (int): Training epochs. Default: 20
- `batch_size` (int): Batch size. Default: 32
- `learning_rate` (float): Adam optimizer learning rate. Default: 5e-4
- `random_state` (int): Random seed

### Architecture

For patient-hour data (one row per timestep):
- Input shape: (batch_size, 1, n_features)
- GRU layers: Bidirectional learning of feature patterns
- Output: Sigmoid → probability of sepsis

### Training

- Optimizer: Adam
- Loss: Binary cross-entropy
- Learning: Mini-batch stochastic gradient descent
- Seeds: Fixed for reproducibility

### Output

- Probability predictions (0 to 1)
- Threshold at 0.5 for binary classification

### Example

```python
model = GRUModel(epochs=20, batch_size=32)
model.fit(X_train, y_train)

# Evaluate
metrics = evaluate_model(model, test_df)
print(f"AUROC: {metrics['auroc']:.4f}")
```

### Requirements

```bash
pip install torch
```

## Choosing a Model

### Use LogisticGLM if:
- You need interpretability (feature importance is clear)
- You want a sparse, explainable model
- You have limited training data
- Speed is critical (very fast inference)

### Use XGBoostModel if:
- Predictive accuracy is the priority
- You can afford the computational cost
- You have sufficient training data (1000+ samples)
- You want to capture feature interactions

### Use GRUModel if:
- You want a neural network baseline
- You have GPU available (for speed)
- You want to explore deep learning approaches
- You have very large training sets (10K+ samples)

## Feature Preprocessing

All models handle preprocessing internally:

1. **Missing Values**: Median imputation per feature
2. **Scaling**: StandardScaler (GLM and GRU only; XGBoost doesn't need it)
3. **Feature Selection**: L1 regularization in GLM; automatic in trees

You can pass DataFrames or numpy arrays; conversion is automatic.

## Evaluating Models

Use the `evaluate_model()` function from `training.py`:

```python
from training import evaluate_model

metrics = evaluate_model(model, test_df, label_column="SepsisLabel")
print(f"AUROC: {metrics['auroc']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1: {metrics['f1']:.4f}")
```

### Returned Metrics

- `auroc`: Area under ROC curve (threshold-independent)
- `accuracy`: Fraction correct (threshold = 0.5)
- `precision`: TP / (TP + FP)
- `recall`: TP / (TP + FN) — important for early detection
- `f1`: Harmonic mean of precision and recall
- `n_samples`: Size of evaluation set
- `n_positive`: Number of positive class samples
- `prevalence`: Fraction with sepsis

## Example: Compare Models

```python
from models import get_model
from training import BootstrapEvaluator

models = ["glm", "xgboost"]
results = {}

for model_name in models:
    model = get_model(model_name)
    evaluator = BootstrapEvaluator(model, train_df)
    
    for i in range(10):
        pids, bootstrap_df = resampler.generate_iteration(i)
        evaluator.evaluate_iteration(bootstrap_df, i)
    
    results[model_name] = evaluator.aggregate_bootstrap_metrics()
```

## Troubleshooting

### XGBoost not available
```
RuntimeError: XGBoost not available. Install with: pip install xgboost
```
Solution: Install XGBoost or skip XGBoost-based experiments.

### PyTorch not available
```
ImportError: PyTorch required for GRU model.
```
Solution: Install PyTorch with `pip install torch` or use GLM/XGBoost.

### Model not fit error
```
ValueError: Model not fit. Call fit() first.
```
Solution: Always call `model.fit(X_train, y_train)` before prediction.

### Mismatched X and y lengths
```
ValueError: X and y have mismatched lengths
```
Solution: Ensure X and y have the same number of rows.

## Performance Notes

### Training Time (approximate)

| Model | 1000 rows | 5000 rows | 20000 rows |
|-------|-----------|-----------|------------|
| GLM | 1 s | 3 s | 10 s |
| XGBoost | 5 s | 15 s | 60 s |
| GRU (CPU) | 30 s | 120 s | 500 s |

### Inference Time (per 1000 samples)

| Model | Time |
|-------|------|
| GLM | 10 ms |
| XGBoost | 20 ms |
| GRU (CPU) | 50 ms |

## References

- **LogisticGLM**: L1-regularized logistic regression (elastic net with l1_ratio=1)
- **XGBoostModel**: Gradient boosting with tree-specific regularization
- **GRUModel**: Recurrent neural networks for sequence modeling
