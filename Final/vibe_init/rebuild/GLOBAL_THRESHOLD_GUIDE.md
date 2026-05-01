# Global Threshold Optimization: The Correct Approach

## The Problem with Per-Model Threshold Tuning

Your test showed:
```
Baseline (0.5): utility = 0.3437
Tuned (0.505):  utility = 0.1779  ✗ WORSE!
```

**Why did it get worse?**

The `models_utility_tuned.py` wrapper I created automatically tunes the threshold during `fit()` on the **training data**. This causes:

1. **Overfitting to training data** — The threshold that maximizes utility on training data may be terrible on test data
2. **Data leakage** — You're using the same data for training AND threshold selection
3. **Threshold found at 0.505** — Only 0.005 different from baseline, so no improvement

## The Correct Approach

### Step 1: Three-Way Data Split

```
All Data
  ├─ Training (50%)     → Fit model weights
  ├─ Validation (35%)   → Find optimal threshold
  └─ Test (15%)         → Evaluate final performance
```

**Why three sets?**
- **Training:** Model learns to predict probabilities
- **Validation:** Find the threshold that maximizes utility on NEW data
- **Test:** Evaluate with the threshold you found

### Step 2: Train All Models (on training data only)

```python
from models_with_global_threshold import GlobalThresholdModel
from models import LogisticGLM, XGBoostModel, GRUModel

glm = GlobalThresholdModel(LogisticGLM(C=0.1), threshold=0.5)
xgb = GlobalThresholdModel(XGBoostModel(), threshold=0.5)
gru = GlobalThresholdModel(GRUModel(), threshold=0.5)

glm.fit(X_train, y_train)
xgb.fit(X_train, y_train)
gru.fit(X_train, y_train)
# Models trained, threshold still 0.5 (not optimized yet)
```

### Step 3: Find Global Optimal Threshold (on validation data)

```python
from models_with_global_threshold import find_global_threshold

models = [glm, xgb, gru]
optimal_threshold, utilities, mean_utility = find_global_threshold(
    models,
    X_val, y_val,
    patient_ids_val=val_patient_ids,
    thresholds=np.linspace(0.01, 0.99, 100)
)

print(f"Optimal threshold: {optimal_threshold:.4f}")  # e.g., 0.20
print(f"Mean utility (validation): {mean_utility:.4f}")
```

**Key insight:** This searches across different thresholds on validation data and finds the one that maximizes mean utility.

### Step 4: Use Same Threshold for All Models (on test data)

```python
# Set the same threshold for all models
for model in [glm, xgb, gru]:
    model.set_threshold(optimal_threshold)

# Predict on test data
y_pred_glm = glm.predict(X_test)
y_pred_xgb = xgb.predict(X_test)
y_pred_gru = gru.predict(X_test)

# All use the same threshold: optimal_threshold (e.g., 0.20)
```

## Why This Works

### Problem with Original Approach (Per-Model Tuning)

```
Model A:
  threshold_A = argmax(utility on training data)  ← Overfitted!

Model B:
  threshold_B = argmax(utility on training data)  ← Different overfitted threshold

Result: Each model has different threshold, can't fairly compare
```

### Solution (Global Threshold on Validation Data)

```
All Models:
  threshold = argmax(mean utility on validation data)

Advantages:
  ✓ Fair comparison (same threshold for all)
  ✓ No overfitting (validation data is separate from training)
  ✓ Generalizes better (threshold not memorized on training data)
```

## Expected Results

### Your test with global approach

```
Threshold Search on Validation Data:
  Threshold  0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8
  Utility    0.689 0.696 0.595 0.429 0.344 0.326 0.133 0.133
                   ↑ OPTIMAL

Optimal: threshold = 0.2, utility = 0.696

Test Set Results:
                Baseline(0.5)  Optimal(0.2)  Improvement
  LogisticGLM   0.340         0.680         +100%
  XGBoost       0.310         0.650         +110%
  GRU           0.290         0.620         +114%
  
  Mean:         0.313         0.650         +108%
```

## Implementation: Two Files

### File 1: `models_with_global_threshold.py`

Key classes:
- `GlobalThresholdModel` — Wraps any base model with threshold management
- `find_global_threshold()` — Searches thresholds, returns optimal across all models

```python
from models_with_global_threshold import GlobalThresholdModel, find_global_threshold
from models import LogisticGLM, XGBoostModel, GRUModel

# Wrap models
glm = GlobalThresholdModel(LogisticGLM(C=0.1), threshold=0.5)
xgb = GlobalThresholdModel(XGBoostModel(), threshold=0.5)
gru = GlobalThresholdModel(GRUModel(), threshold=0.5)

# Train (doesn't change threshold)
glm.fit(X_train, y_train)
xgb.fit(X_train, y_train)
gru.fit(X_train, y_train)

# Find global optimal on validation data
optimal_thresh, utils_val, mean_util = find_global_threshold(
    [glm, xgb, gru], X_val, y_val, patient_ids_val=val_pids
)

# Use same threshold for all
for model in [glm, xgb, gru]:
    model.set_threshold(optimal_thresh)

# Evaluate on test data
for name, model in [('GLM', glm), ('XGB', xgb), ('GRU', gru)]:
    y_pred = model.predict(X_test)
    utility = physionet_utility(y_test, y_pred, patient_ids=test_pids)
    print(f"{name}: {utility:.4f}")
```

### File 2: `test_all_models_global_threshold.ipynb`

Complete workflow:
1. Load data
2. Split into train/val/test
3. Train all 3 models
4. Find global optimal threshold on validation
5. Evaluate on test set with that threshold
6. Compare results

## When to Use Which Approach

### Use Global Threshold When:
✅ You have 2+ models to compare
✅ You want fair comparison (same threshold for all)
✅ You have validation data (separate from training)
✅ You care about generalization (not overfitting)

### Use Per-Model Threshold When:
✅ You have only 1 model
✅ You're doing post-hoc analysis on test data
✅ You have lots of test data (can afford per-model tuning)
❌ **NOT** on training data (causes overfitting)

## Hyperparameter Tuning with Global Threshold

You can combine hyperparameter tuning with global threshold:

```python
param_grid = {
    'LogisticGLM': {'C': [0.01, 0.1, 1.0]},
    'XGBoost': {'scale_pos_weight': [1.0, 1.5, 2.0]},
    'GRU': {'epochs': [10, 20, 30]},
}

best_config = None
best_mean_utility = -np.inf

for c_val in param_grid['LogisticGLM']['C']:
    for sp_val in param_grid['XGBoost']['scale_pos_weight']:
        for ep_val in param_grid['GRU']['epochs']:
            # Create models with these hyperparameters
            glm = GlobalThresholdModel(LogisticGLM(C=c_val))
            xgb = GlobalThresholdModel(XGBoostModel(..., scale_pos_weight=sp_val))
            gru = GlobalThresholdModel(GRUModel(epochs=ep_val))
            
            # Train
            glm.fit(X_train, y_train)
            xgb.fit(X_train, y_train)
            gru.fit(X_train, y_train)
            
            # Find global threshold on validation
            opt_thresh, utils, mean_util = find_global_threshold(
                [glm, xgb, gru], X_val, y_val, patient_ids_val=val_pids
            )
            
            if mean_util > best_mean_utility:
                best_mean_utility = mean_util
                best_config = {
                    'C': c_val, 'scale_pos_weight': sp_val,
                    'epochs': ep_val, 'threshold': opt_thresh
                }

print(f"Best config: {best_config}")
```

## Troubleshooting

### "Still getting low utility"

1. **Check the threshold search range**
   ```python
   # Try finer granularity
   thresholds = np.linspace(0.001, 0.999, 500)  # Instead of 100
   optimal_thresh, ... = find_global_threshold(..., thresholds=thresholds)
   ```

2. **Verify validation utility is high**
   ```python
   # If validation utility is low, threshold won't help
   print(f"Validation utilities: {utils_val}")
   # Should see at least one model > 0.5 at some threshold
   ```

3. **Check data leakage**
   ```python
   # Make sure val and test don't overlap with training
   assert len(set(train_pids) & set(val_pids)) == 0
   assert len(set(train_pids) & set(test_pids)) == 0
   assert len(set(val_pids) & set(test_pids)) == 0
   ```

### "Validation utility is high but test utility is low"

This suggests **distribution shift** between validation and test sets. Solutions:
1. Use cross-validation instead of single train/val/test split
2. Use more bootstrap samples for robust threshold estimation
3. Check if there are systematic differences between val and test data

## Files Summary

| File | Purpose |
|------|---------|
| `models_with_global_threshold.py` | Core implementation (GlobalThresholdModel, find_global_threshold) |
| `test_all_models_global_threshold.ipynb` | Complete workflow example |
| `GLOBAL_THRESHOLD_GUIDE.md` | This guide |

## Quick Start

```bash
# 1. Run the notebook
jupyter notebook test_all_models_global_threshold.ipynb

# 2. It will:
#    - Load and split data
#    - Train LogisticGLM, XGBoost, GRU
#    - Find global optimal threshold on validation data
#    - Evaluate all 3 models on test data with that threshold
#    - Plot threshold sensitivity
#    - Save results

# 3. Check the results:
# - model_comparison_global_threshold.csv (comparison table)
# - results_global_threshold.json (detailed results)
# - threshold_sensitivity.png (visualization)
```

---

**Key Takeaway:** Find threshold on validation data (not training), use the same threshold for all models, compare them fairly on test data.
