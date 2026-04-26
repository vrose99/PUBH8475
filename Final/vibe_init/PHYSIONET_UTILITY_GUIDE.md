# PhysioNet 2019 Challenge Utility Scores

This document explains the PhysioNet 2019 Sepsis Challenge's utility function, how it's computed in this pipeline, and how to interpret the results.

## The Utility Function

The challenge uses a **time-dependent utility function** that rewards **early detection** of sepsis while penalizing missed or late predictions.

For each patient at each time step `t`, the utility is:

```
U(s,t) = 
  - If sepsis patient & positive prediction:
    U_TP(t) = 1 - (t_sepsis - t) / 144    (if t < t_sepsis)
    U_TP(t) = -2                            (if t ≥ t_sepsis, too late)
  
  - If sepsis patient & negative prediction:
    U_FN = -2                               (missed sepsis)
  
  - If non-sepsis patient & positive prediction:
    U_FP = -0.05                            (false alarm)
  
  - If non-sepsis patient & negative prediction:
    U_TN = 0                                (correct negative)
```

## Key Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `U_TP` base | 1.0 | Maximum reward (perfect early detection) |
| `U_TP` decay | 1/144 per hour | Reward decreases as you get closer to sepsis onset |
| `U_FN` | -2.0 | Most severe penalty (missing sepsis costs lives) |
| `U_FP` | -0.05 | Mild penalty (alarm fatigue, but less critical than missing sepsis) |
| `U_TN` | 0.0 | No reward for correct negatives (expected baseline) |

## What This Means

- **Early warning (6 hours before)**: U_TP ≈ 0.96 — large reward
- **Warning at onset (0 hours before)**: U_TP ≈ 1.0 — still good  
- **Late detection (after onset)**: U_TP ≈ -2.0 — same penalty as missed
- **False alarm (non-septic patient)**: U_FP ≈ -0.05 — mild cost

**Fairness lens:** If one gender gets systematically earlier warnings (higher U_TP), their `physionet_utility` score will be higher. A `physionet_utility_gap` of 0 means both genders received equally-timed predictions.

## Pipeline Output

When running in **time-series mode** (`USE_TIMESERIES = True`), the pipeline outputs:

### 1. Console Summary: Overall Utility

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║ OVERALL UTILITY — Baseline (no mitigation)                                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

model          overall_physionet_utility
liu_glm                            0.2543
liu_xgboost                        0.1876
liu_rnn                            0.1234
```

This shows: **Liu GLM achieved a utility score of 0.25 on the test set.**

### 2. Console Summary: Per-Group Utility

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║ PER-GROUP UTILITY — Females vs Males (baseline)                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

model          Female Utility  Male Utility  Gap (F - M)
liu_glm               0.2810        0.2145        0.0665
liu_xgboost           0.1995        0.1823       -0.0172
liu_rnn               0.1543        0.1089        0.0454
```

**How to read:**
- Liu GLM: females got utility 0.281, males got 0.215 → gap of +0.067 (females favored)
- Liu XGBoost: females got slightly lower utility → gap of -0.017 (males slightly favored)
- **Fairness:** gap close to 0 is fairer (equal early-warning for both groups)

### 3. Console Summary: Mitigation Impact

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║ MITIGATION IMPACT — Best model: liu_glm                                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝

mitigation          Overall Utility  Gap (F - M)
none                        0.2543       0.0665
fairness_penalty            0.2401       0.0241
reweighting                 0.2478       0.0182
smote                       0.2356       0.0095
```

**How to read:**
- `none`: baseline utility is 0.254, but gap is large (0.067)
- `fairness_penalty`: utility drops to 0.240, but gap shrinks to 0.024 (more fair!)
- `reweighting`: utility is 0.248, gap is 0.018 (good balance)
- **Trade-off:** fairness improvements sometimes cost overall utility

### 4. CSV Output: `utility_summary.csv`

Complete table with all metrics:
- `overall_physionet_utility` — average utility across all test rows
- `female_physionet_utility` — average utility for female patients
- `male_physionet_utility` — average utility for male patients
- `physionet_utility_gap` — female utility minus male utility
- `median_detection_lead_hours` — how many hours before onset was the first alarm?
- `pct_detected_before_onset` — % of septic patients detected early
- `pct_missed` — % of septic patients never detected

## Interpreting Results

### Good Utility, Bad Fairness
```
Overall utility: 0.30 (high)
Gender gap: 0.12 (large)
→ Model predicts well overall but favors one gender
→ Apply fairness mitigation (reweighting, fairness_penalty)
```

### Bad Utility, Good Fairness
```
Overall utility: 0.10 (low)
Gender gap: 0.01 (small)
→ Model fails to detect early but treats both groups equally
→ Try better model or more features
```

### Good Utility, Good Fairness
```
Overall utility: 0.25 (reasonable)
Gender gap: 0.02 (small)
→ Model is both accurate and fair — acceptable for deployment
```

## Key Metrics for Fair Comparison

| Metric | Target | Why |
|--------|--------|-----|
| `overall_physionet_utility` | Maximize | Higher = better early detection |
| `physionet_utility_gap` | Minimize | Gap close to 0 = equal service to both groups |
| `detection_lead_gap_hours` | Minimize | Both genders should get equal warning time |
| `missed_rate_gap` | Minimize | Neither group should have more missed cases |
| `alarm_fatigue_gap` | Minimize | False alarm rates should be equal |

## Running the Pipeline for Utility Output

### Time-Series Mode (Recommended)
```bash
cd /Users/vrose/ClaudeContainer/PUBH8475/Final/vibe_init

# Edit pipeline.py
# SET: USE_TIMESERIES = True

python pipeline.py                  # ~5–10 min (TEST=True, 300 patients)
python pipeline.py --no-test        # ~1–2 hours (full dataset)
```

Outputs:
- Console: PhysioNet utility summaries (as above)
- `outputs/tables/utility_summary.csv` — all utility metrics
- `outputs/tables/results_all.csv` — complete evaluation results

### Static Mode (Comparison)
```bash
# Edit pipeline.py
# SET: USE_TIMESERIES = False

python pipeline.py
```

This uses standard ML metrics (AUC, TPR, FPR) instead of time-dependent utility.

## References

- PhysioNet 2019 Challenge: https://physionet.org/content/challenge-2019/1.0.0/
- Reyna et al. (2019): "Early sepsis detection with machine learning techniques" (Nature Reviews Methods Primers and CinC challenge papers)
