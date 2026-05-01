"""
Quick smoke-test for the fixed XGBoost and GRU models.
Run from rebuild/:  python test_xgb_gru_quick.py

Tests train sizes 200 and 300, bootstrap size 50, 3 iterations each.
Prints utility at threshold 0.5 AND at the optimal threshold.
Target: utility > 0.15 for both models.
"""

import sys, logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.WARNING)

from data_loader import load_physionet_files, add_hours_until_sepsis, split_patients_by_status, get_rows_for_patients
from bootstrap import BootstrapResampler
from models import XGBoostModel, GRUModel
from training import extract_Xy
from utility import physionet_utility, find_optimal_threshold

DATA_DIR = Path("../data/physionet_sepsis")
TRAIN_SIZE = 300
BOOT_SIZE = 50
N_ITER = 3
RANDOM_STATE = 42

print("Loading data …")
raw_df = load_physionet_files(DATA_DIR)
df = add_hours_until_sepsis(raw_df, keep_post_onset=True)
print(f"  {raw_df['patient_id'].nunique():,} patients, {len(raw_df):,} rows")

train_pids, boot_pids = split_patients_by_status(
    df, n_train_patients=TRAIN_SIZE,
    random_state=RANDOM_STATE, stratify_by_sepsis=True,
)
train_df = get_rows_for_patients(df, train_pids)
X_train, y_train = extract_Xy(train_df, "SepsisLabel")

print(f"\nTraining on {TRAIN_SIZE} patients ({len(X_train)} rows)")
print(f"  Class ratio: {(y_train==0).sum()} neg / {(y_train==1).sum()} pos "
      f"= {(y_train==0).sum()/(y_train==1).sum():.1f}:1")

resampler = BootstrapResampler(
    boot_pids, df,
    n_iterations=N_ITER,
    bootstrap_sample_size=BOOT_SIZE,
    random_state=RANDOM_STATE,
)

for model_name, ModelClass, kwargs in [
    ("XGBoost", XGBoostModel, {}),
    ("GRU",     GRUModel,     {}),
]:
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    try:
        model = ModelClass(**kwargs)
        model.fit(X_train, y_train)
    except Exception as exc:
        print(f"  FIT FAILED: {exc}")
        continue

    utilities_05, utilities_opt = [], []

    for i in range(N_ITER):
        _, boot_df = resampler.generate_iteration(i)
        X_eval, y_eval = extract_Xy(boot_df, "SepsisLabel")
        pids = boot_df["patient_id"].values

        y_proba = model.predict_proba(X_eval)[:, 1]

        # Utility at default threshold 0.5
        y_pred_05 = (y_proba >= 0.5).astype(int)
        u_05 = physionet_utility(y_eval, y_pred_05, patient_ids=pids)

        # Utility at optimal threshold (searched on this eval set as upper bound)
        opt_thresh, u_opt = find_optimal_threshold(
            y_proba, y_eval, patient_ids=pids,
            thresholds=np.linspace(0.01, 0.99, 99),
        )

        utilities_05.append(u_05)
        utilities_opt.append(u_opt)

        n_pos_pred = y_pred_05.sum()
        print(f"  iter {i+1}: threshold=0.50 → utility={u_05:.4f}  "
              f"(n_alarms={n_pos_pred})  |  "
              f"optimal threshold={opt_thresh:.2f} → utility={u_opt:.4f}")

    print(f"\n  Summary (threshold=0.5):")
    print(f"    mean={np.mean(utilities_05):.4f}  std={np.std(utilities_05):.4f}  "
          f"min={np.min(utilities_05):.4f}")
    print(f"  Summary (optimal threshold):")
    print(f"    mean={np.mean(utilities_opt):.4f}  std={np.std(utilities_opt):.4f}  "
          f"min={np.min(utilities_opt):.4f}")

    target = 0.15
    passed = np.mean(utilities_05) > target
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}: mean utility at 0.5 {'>' if passed else '<='} {target}")
