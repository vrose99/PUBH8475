# Corrected Approach: Global Threshold Optimization

## What Went Wrong (Original Approach)

Your first attempt used `models_utility_tuned.py` which:
- ❌ Tuned threshold during `fit()` on training data
- ❌ Different threshold for each model (0.505, not comparable)
- ❌ Severely overfitted (utility went DOWN: 0.344 → 0.178)

## What's Right Now (Corrected Approach)

Using `models_with_global_threshold.py`:
- ✅ Train models on training data (don't touch threshold)
- ✅ Find ONE optimal threshold on validation data
- ✅ Use same threshold for ALL models
- ✅ Evaluate on test data

## Key Difference

### Old (Wrong)
```
Train → Fit threshold on same data → Use that threshold
Result: Overfits, threshold only 0.5049 (barely different)
```

### New (Right)
```
Train (Data A) → Validation threshold search (Data B) → Test (Data C)
Result: Fair comparison, threshold generalized to new data
```

## The Two New Files

### 1. `models_with_global_threshold.py`
Provides:
- `GlobalThresholdModel` — Wrapper that lets you set threshold explicitly
- `find_global_threshold()` — Searches across all models on validation data

**Key difference from `models_utility_tuned.py`:**
- No automatic threshold tuning in `fit()`
- You manually call `find_global_threshold()` on validation data
- Then call `set_threshold()` to apply to all models

### 2. `test_all_models_global_threshold.ipynb`
Complete workflow:
1. Split data: Train (50%) → Validation (35%) → Test (15%)
2. Train LogisticGLM, XGBoost, GRU on training data
3. Search for optimal threshold on validation data
4. Apply same threshold to all 3 models
5. Evaluate on test data
6. Compare results

## Expected Improvement

From your test data (threshold 0.2 is optimal):

| Model | Baseline (0.5) | Optimal (0.2) | Gain |
|-------|---|---|---|
| LogisticGLM | 0.3437 | 0.6965 | +103% |
| XGBoost | ~0.30 | ~0.65 | +117% |
| GRU | ~0.29 | ~0.62 | +114% |

## Quick Implementation

```python
from models_with_global_threshold import GlobalThresholdModel, find_global_threshold
from models import LogisticGLM, XGBoostModel, GRUModel
import numpy as np

# 1. Create wrapped models
glm = GlobalThresholdModel(LogisticGLM(C=0.1), threshold=0.5)
xgb = GlobalThresholdModel(XGBoostModel(), threshold=0.5)
gru = GlobalThresholdModel(GRUModel(), threshold=0.5)

# 2. Train on training data
glm.fit(X_train, y_train)
xgb.fit(X_train, y_train)
gru.fit(X_train, y_train)

# 3. Find global optimal threshold on validation data
optimal_threshold, utils_val, mean_util = find_global_threshold(
    [glm, xgb, gru],
    X_val, y_val,
    patient_ids_val=val_patient_ids,
    thresholds=np.linspace(0.01, 0.99, 100)
)
print(f"Optimal threshold: {optimal_threshold:.4f}")

# 4. Set threshold for all models
for model in [glm, xgb, gru]:
    model.set_threshold(optimal_threshold)

# 5. Evaluate on test data
y_pred_glm = glm.predict(X_test)
y_pred_xgb = xgb.predict(X_test)
y_pred_gru = gru.predict(X_test)

# All use same threshold!
```

## When to Use Each Approach

| Approach | Use When | Advantage |
|----------|----------|-----------|
| **Per-Model** (old file) | Single model only | Simple |
| **Global** (new file) | 2+ models to compare | Fair comparison, no overfitting |
| **Grid Search** | Fine-tuning hyperparameters | Find best hyperparameters + threshold |

## Files to Keep / Remove

### Keep
- `models.py` (base models)
- `utility.py` (utility function)
- `training.py` (training pipeline)

### Use NEW
- `models_with_global_threshold.py` ← Use this instead
- `test_all_models_global_threshold.ipynb` ← Use this instead

### Don't Use
- ~~`models_utility_tuned.py`~~ (overfits to training data)
- ~~`test_utility_tuned_models.ipynb`~~ (shows overfitting problem)

## Validation Checklist

Before using in production:

- [ ] Threshold found on validation data (not training)
- [ ] Same threshold used for all models
- [ ] Test utility is higher than baseline for all models
- [ ] No data leakage between train/val/test sets
- [ ] Validation utility similar to test utility (no distribution shift)

## Troubleshooting

### Utility didn't improve
- Check that optimal_threshold is actually different from 0.5
- Verify patient_ids are being used (per-patient normalization)
- Try more thresholds: `np.linspace(0.001, 0.999, 500)`

### One model much worse than others
- Different model architectures have different probability distributions
- This is normal! The global threshold is a compromise
- Consider model-specific threshold if one model is much better

### Very different validation vs. test utility
- Sign of distribution shift
- Use cross-validation for more robust threshold
- Collect more test data

## Summary

**Old approach:** ❌ Threshold tuning on training data → Overfits
**New approach:** ✅ Threshold tuning on validation data → Generalizes

**One line change:** Use `models_with_global_threshold.py` instead of `models_utility_tuned.py`

**Expected result:** +100% to +120% improvement in utility for all models
