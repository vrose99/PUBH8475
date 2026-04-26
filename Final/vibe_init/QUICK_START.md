# Quick Start: Running the Fairness Pipeline with PhysioNet Utility Scores

## 1. Enable Time-Series Mode (for PhysioNet Challenge Utility)

Edit `pipeline.py` line 37:
```python
USE_TIMESERIES = True   # ← change from False
TEST = True              # keep True for fast testing
```

## 2. Run the Pipeline

```bash
cd /Users/vrose/ClaudeContainer/PUBH8475/Final/vibe_init
python pipeline.py
```

### Expected Runtime

- **TEST mode** (`TEST = True`): 5–10 minutes
  - 300 patients
  - 3 models × 4 mitigations = 12 cells
  
- **Production mode** (`python pipeline.py --no-test`): 1–2 hours
  - 40,336 patients (both training_setA + training_setB)
  - 3 models × 4 mitigations = 12 cells

## 3. Read the Console Output

The pipeline will print three tables to the console:

### Table 1: Overall Utility (All Models)
Shows average PhysioNet utility score per model, ranked by best utility.

```
model              overall_physionet_utility
─────────────────────────────────────────────
liu_glm                            0.2543
liu_xgboost                        0.1876
liu_rnn                            0.1234
```

**Read as:** Liu GLM achieves 0.254 average utility on the test set.

### Table 2: Per-Group Fairness (All Models, Baseline Only)
Shows utility scores split by gender and the gap (fairness metric).

```
model          Female Utility  Male Utility  Gap (F - M)
───────────────────────────────────────────────────────────
liu_glm               0.2810        0.2145        0.0665
liu_xgboost           0.1995        0.1823       -0.0172
liu_rnn               0.1543        0.1089        0.0454
```

**Read as:** 
- Gap > 0 = females get higher utility (earlier warnings)
- Gap < 0 = males get higher utility
- Gap ≈ 0 = fair (equal early warning for both)

### Table 3: Mitigation Impact (Best Model Only)
Shows how different bias mitigation strategies affect utility and fairness.

```
mitigation          Overall Utility  Gap (F - M)
──────────────────────────────────────────────────
none                        0.2543       0.0665
fairness_penalty            0.2401       0.0241
reweighting                 0.2478       0.0182
smote                       0.2356       0.0095
```

**Read as:**
- `fairness_penalty` reduces gap most (0.067 → 0.024) but costs utility (0.254 → 0.240)
- `reweighting` balances utility + fairness (0.248 utility, 0.018 gap)
- **Choose based on your priorities:**
  - Prioritize accuracy? → `none` or `reweighting`
  - Prioritize fairness? → `smote` or `fairness_penalty`

## 4. Check CSV Files

**`outputs/tables/results_all.csv`** — Full results table with all columns:
- `model`, `mitigation` (metadata)
- `overall_physionet_utility`, `female_physionet_utility`, `male_physionet_utility`, `physionet_utility_gap`
- `median_detection_lead_hours`, `pct_detected_before_onset`, `pct_missed`
- `tpr`, `fpr`, `auroc` (for comparison with static metrics)

**`outputs/tables/utility_summary.csv`** — Utility-only columns (easier to read/analyze)

Load in Python:
```python
import pandas as pd

# All results
df = pd.read_csv("outputs/tables/results_all.csv")

# Filter for best model
best = df[df['model'] == 'liu_glm']

# See fairness gap across mitigations
fairness_gap = best.groupby('mitigation')['physionet_utility_gap'].mean()
print(fairness_gap.sort_values(key=abs))
```

## 5. Understand the Metrics

| Metric | Formula | Good Value | Meaning |
|--------|---------|------------|---------|
| `physionet_utility` | Σ U(t) / n | 0.2–0.3 | Average reward per prediction (higher is better early warning) |
| `physionet_utility_gap` | female_utility − male_utility | Close to 0 | Fairness: equal early warning for both genders |
| `median_detection_lead_hours` | median(t_sepsis − t_prediction) | 12–48 hours | How far in advance was sepsis detected? |
| `pct_detected_before_onset` | % of septic patients with ≥1 pre-onset alarm | >80% | Coverage: how many patients got warned early? |
| `pct_missed` | % of septic patients with 0 alarms | <20% | Miss rate: what % were never detected? |

## 6. Troubleshooting

### "FileNotFoundError: No PSV files found"
Check that data exists:
```bash
ls /Users/vrose/ClaudeContainer/PUBH8475/Final/vibe_init/data/physionet_sepsis/training_setA/training/*.psv | wc -l
ls /Users/vrose/ClaudeContainer/PUBH8475/Final/vibe_init/data/physionet_sepsis/training_setB/training_setB/*.psv | wc -l
```

Should print: `20336` and `20000` respectively.

### "ImportError: torch is not installed"
The RNN model requires PyTorch. To skip it:
```bash
python pipeline.py --models liu_glm liu_xgboost
```

### Utility scores all negative
This is normal! The challenge utility is zero-sum:
- One misdetection (FN) costs -2 points
- Must catch many patients early to overcome it
- Scores of -0.5 to +0.5 are typical depending on prevalence

## 7. Next Steps

1. **Understand fairness:** Read `PHYSIONET_UTILITY_GUIDE.md`
2. **Experiment:** Flip `TEST = False`, run on full dataset
3. **Compare modes:** Run with `USE_TIMESERIES = False` to see static metrics
4. **Mitigate:** If gap is large, try `--mitigations reweighting fairness_penalty`

## Files Modified

- `pipeline.py` — Added time-series mode toggle + utility output
- `fairness_timeseries.py` — Added `physionet_2019_utility()` function
- `data_loader.py` — Both setA and setB auto-discovered via rglob
- `PHYSIONET_UTILITY_GUIDE.md` — Detailed explanation of utility scoring
- `QUICK_START.md` — This file
