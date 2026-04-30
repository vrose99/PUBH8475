# PhysioNet Utility Function Visualized

## What is the Utility Function?

The PhysioNet 2019 Challenge uses a **custom scoring function** that measures how well a sepsis prediction system performs in a clinical setting.

### The Problem It Solves

Standard metrics (accuracy, AUROC) treat all errors equally:
- ❌ Predicting sepsis 12 hours early = same as predicting 5 minutes early
- ❌ Missing sepsis by 1 minute = same as missing by 12 hours
- ❌ False alarm = same weight as missed sepsis

The PhysioNet utility function adds **context**:
- ✅ Early prediction (6 hours before sepsis) = most valuable
- ✅ Late prediction or missed diagnosis = heavily penalized
- ✅ False alarm = small penalty

---

## The Utility Score Breakdown

### For Septic Patients (labels contain 1s)

#### True Positive (Alarm when patient has sepsis)

```
Utility for a true positive prediction:

     +1.0 ┌──────────────────────┐
          │      OPTIMAL ZONE    │
    +0.5  │    (6h before       │
          │     sepsis)         │
      0.0 ├────────┬────────────┤────────────┐
          │        │            │            │
     -0.5 │        │            │            │
          │        │            │            │
     -2.0 └────────┘────────────┘────────────┘
          -12h    -6h         +3h
         (too    (best)      (too
          early)              late)

Time relative to sepsis onset (t_sepsis)
```

**Breakdown:**
- **-12h to -6h:** Linear ramp from 0 to +1 (getting better the closer to sepsis)
- **-6h to +3h:** Ramped down from +1 to 0 (still good, but utility decreases)
- **+3h onwards:** Drops to -2 (too late, severe penalty)

**Example utilities:**
- Predict 12h before: +0.0 (neutral, doesn't help yet)
- Predict 6h before: +1.0 (PERFECT — gives doctors time to prepare)
- Predict 2h before: +0.6 (good, still in treatment window)
- Predict 1h after: -0.4 (too late, should have caught earlier)
- Predict 6h after: -2.0 (completely useless, missed critical window)

---

#### False Negative (No alarm when patient has sepsis)

```
Utility for a false negative prediction:

     +1.0 │
          │
    +0.5  │
          │
      0.0 ├────────┬────────────┤────────────┐
          │        │            │            │
     -0.5 │        │            │            │
          │        │            │            │
     -2.0 └────────┴────────────┴────────────┘
          -12h    -6h         +3h
         (too    (critical   (too
          early)  window)     late)

Time relative to sepsis onset (t_sepsis)
```

**Breakdown:**
- **≤-6h:** 0 utility (too early to expect detection)
- **-6h to +3h:** Linear penalty from 0 to -2 (WORST CASE — missed critical window)
- **>+3h:** -2 utility (completely missed sepsis)

**Example utilities:**
- No alarm, sepsis at -12h: 0 (acceptable, no alarm needed yet)
- No alarm, sepsis at -2h: -0.4 (should have alarmed!)
- No alarm, sepsis at +2h: -1.2 (critical failure)
- No alarm, sepsis at +10h: -2.0 (catastrophic failure)

---

### For Non-Septic Patients (no sepsis)

#### False Positive (Alarm when patient is NOT septic)

```
Utility for a false positive prediction:
  ALWAYS: -0.05 (alarm fatigue)
```

**Why -0.05?**
- Alarms are costly (tie up resources, cause stress)
- But not as bad as missing sepsis
- Ratio of FP penalty to TP gain: 0.05 / 1.0 = 5% cost

---

#### True Negative (No alarm, patient is NOT septic)

```
Utility for a true negative prediction:
  ALWAYS: 0.0 (correct)
```

**Why 0?**
- No action is correct for non-septic patients
- Neutral outcome

---

## Putting It Together

```
SEPTIC PATIENT (POSITIVE CLASS):

Prediction=1 (ALARM):
  └─ u = max(-2, 1 - 0.167*|t-t_sep|)  if -12h < t < +3h else -2

Prediction=0 (NO ALARM):
  └─ u = 0                              if t ≤ -6h
  └─ u = -0.667*(t-t_sep) - 4           if -6h < t < +3h
  └─ u = -2                             if t ≥ +3h


NON-SEPTIC PATIENT (NEGATIVE CLASS):

Prediction=1 (ALARM):
  └─ u = -0.05  (always)

Prediction=0 (NO ALARM):
  └─ u = 0      (always)
```

---

## Why Threshold = 0.5 is Wrong

Standard models output probabilities (0 to 1). The natural threshold is 0.5.

**BUT:** The utility function is non-convex in threshold space!

```
Utility vs. Threshold:

+0.8 ┌─────────────────┐
     │                 │
+0.6 │     ╱╲          │
     │    ╱  ╲         │  Optimal threshold
+0.4 │   ╱    ╲   ╱╲   │  might be 0.15 or 0.35
     │  ╱      ╲ ╱  ╲  │  NOT 0.5!
+0.2 │ ╱        ╲     ╲ │
     │                  │
  0  ├──────────────────┤
     0.0  0.2  0.4  0.6  0.8  1.0
              Threshold
```

**Example from your tests:**
- Utility at threshold=0.5: 0.34
- Utility at threshold=0.2: 0.83  ← Better!
- Improvement: +144%

---

## The Key Insight

### Standard Metrics Assume Equal Cost

```
If you predict:

     Sepsis    Non-Sepsis
   ┌──────────┬────────────┐
   │ Correct  │ Cost: C_fp │
Predict ├──────────┼────────────┤
1  │          │            │
   └──────────┴────────────┘

   ┌──────────┬────────────┐
   │ Cost:_fn │ Correct    │
Predict ├──────────┼────────────┤
0  │          │            │
   └──────────┴────────────┘

Threshold = 0.5 → predict positive if P(sepsis) > 0.5
```

### PhysioNet Utility Uses Asymmetric, Time-Dependent Cost

```
Cost of FN (missing sepsis):
  ├─ At -6h: 0 (early, not critical yet)
  └─ At +2h: -1.2 (VERY expensive, missed critical window)

Cost of FP (false alarm):
  └─ Always -0.05 (relatively cheap)

Cost of TP (correct alarm):
  ├─ At -6h: +1.0 (VERY valuable)
  └─ At +5h: -2.0 (too late, worthless)
```

**This changes the optimal threshold!**

---

## How to Find the Optimal Threshold

```python
from utility import find_optimal_threshold
import numpy as np

# After training your model
y_proba = model.predict_proba(X_val)[:, 1]

# Search thresholds
thresholds = np.linspace(0.01, 0.99, 100)
optimal_threshold, max_utility = find_optimal_threshold(
    y_proba, y_val, 
    patient_ids=patient_ids,
    thresholds=thresholds
)

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"Maximum utility: {max_utility:.4f}")

# Use it
y_pred = (y_proba >= optimal_threshold).astype(int)
```

---

## Implications for Model Training

### Current Approach (Wrong)

```
1. Train model to maximize AUROC
2. Use threshold = 0.5
3. Hope utility is good (it's not!)
```

### Better Approach

```
1. Train model on classification
2. Find threshold that maximizes utility
3. Use that threshold for predictions
```

### Best Approach (for PhysioNet)

```
1. Train model with class weights favoring early detection
2. Adjust hyperparameters to increase sensitivity
3. Find threshold that maximizes utility
4. Use that threshold for predictions
```

---

## Example: Why Your Model Has Low Utility

### Scenario 1: High Specificity (Avoid False Alarms)

```
Model trained with:
  - High regularization (C=0.01)
  - Default threshold (0.5)

Result:
  - Few false alarms (-0.05 each is cheap)
  - Many false negatives (-2.0 each is EXPENSIVE)
  - Utility: LOW ❌

Fix:
  - Lower regularization (C=0.1) → more sensitive
  - Lower threshold (0.2) → alarm more often
  - Result: Utility increases significantly ✅
```

### Scenario 2: Wrong Threshold

```
Model actually has good probabilities:
  - P(sepsis|features) is well-calibrated
  - But threshold=0.5 is chosen arbitrarily

Result:
  - With threshold=0.5: utility=0.30
  - Optimal threshold=0.15: utility=0.70
  - Just by changing threshold: +133% improvement!

Fix:
  - Search for optimal threshold
  - Use find_optimal_threshold() function
```

---

## Summary: The Three Failures

| Failure | Example | Impact | Fix |
|---------|---------|--------|-----|
| **Wrong threshold** | Using 0.5 when 0.15 is optimal | Utility: 0.30 → 0.70 (+133%) | Search thresholds |
| **Wrong regularization** | C=0.01 (too low sensitivity) | Utility: 0.30 → 0.55 (+83%) | Tune C, reduce regularization |
| **Wrong hyperparameters** | max_depth=4, scale_pos_weight=1.0 | Utility: 0.30 → 0.65 (+117%) | Tune for utility, not accuracy |

**Your model likely fails on all three!**

The solution: Use the utility-tuned models that fix all three issues.
