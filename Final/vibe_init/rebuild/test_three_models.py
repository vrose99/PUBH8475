"""
Unified test for all three models with configurable bootstrap parameters.

Usage:
    python test_three_models.py

Configuration:
    TRAIN_SIZE = 300 patients (fixed)
    THRESHOLD = 0.4 (fixed)
    BOOTSTRAP_POOL_SIZE = 100 patients available for bootstrap sampling
    BOOTSTRAP_EVAL_SIZE = 50 patients per bootstrap iteration (sampled with replacement)
    N_BOOTSTRAP_ITERATIONS = 5 iterations

Outputs utility score for each model at threshold 0.4.
"""

import sys, logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.WARNING)

from data_loader import (
    load_physionet_files,
    add_hours_until_sepsis,
    split_patients_by_status,
    get_rows_for_patients,
)
from bootstrap import BootstrapResampler
from models import LogisticGLM, XGBoostModel, GRUModel
from training import extract_Xy
from utility import physionet_utility

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_SIZE = 20000
THRESHOLD = 0.4
BOOTSTRAP_POOL_SIZE = 2000
BOOTSTRAP_EVAL_SIZE = 500
N_BOOTSTRAP_ITERATIONS = 5000
RANDOM_STATE = 1

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("THREE-MODEL TEST")
print("=" * 70)
print(f"Config:")
print(f"  Train size:        {TRAIN_SIZE} patients")
print(f"  Threshold:         {THRESHOLD}")
print(f"  Bootstrap pool:    {BOOTSTRAP_POOL_SIZE} patients")
print(f"  Bootstrap eval:    {BOOTSTRAP_EVAL_SIZE} patients per iteration")
print(f"  Iterations:        {N_BOOTSTRAP_ITERATIONS}")

# Load and split data
print(f"\nLoading data...")
DATA_DIR = Path("../data/physionet_sepsis")
raw_df = load_physionet_files(DATA_DIR)
df = add_hours_until_sepsis(raw_df, keep_post_onset=True)
print(f"  {raw_df['patient_id'].nunique():,} patients, {len(raw_df):,} rows")

# Split into training and bootstrap pool
train_pids, boot_pids = split_patients_by_status(
    df,
    n_train_patients=TRAIN_SIZE,
    random_state=RANDOM_STATE,
    stratify_by_sepsis=True,
)
train_df = get_rows_for_patients(df, train_pids)
X_train, y_train = extract_Xy(train_df, "SepsisLabel")

print(f"\nTraining data:")
print(f"  {len(X_train)} rows, {X_train.shape[1]} features")
print(f"  Class ratio: {(y_train==0).sum()}/{(y_train==1).sum()} "
      f"(neg/pos) = {(y_train==0).sum()/(y_train==1).sum():.1f}:1")
print(f"  Bootstrap pool: {len(boot_pids)} patients available")

# Fit models
models = {
    "LogisticGLM": LogisticGLM(C=0.1),
    "XGBoost": XGBoostModel(),
    "GRU": GRUModel(),
}

print(f"\nFitting models...")
for name, model in tqdm(models.items(), desc="Fitting", unit="model"):
    try:
        model.fit(X_train, y_train)
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")

# Bootstrap evaluation
print(f"\nBootstrap evaluation (threshold={THRESHOLD}):")
print(f"{'Model':<15} {'Iter':<6} {'Utility':<10} {'N_Alarms':<10}")
print("-" * 45)

results = {name: [] for name in models.keys()}

resampler = BootstrapResampler(
    bootstrap_pool_patient_ids=boot_pids,
    full_df=df,
    n_iterations=N_BOOTSTRAP_ITERATIONS,
    bootstrap_sample_size=BOOTSTRAP_EVAL_SIZE,
    random_state=RANDOM_STATE,
)

for iter_idx in tqdm(range(N_BOOTSTRAP_ITERATIONS), desc="Bootstrap iterations", unit="iteration"):
    _, boot_df = resampler.generate_iteration(iter_idx)
    X_eval, y_eval = extract_Xy(boot_df, "SepsisLabel")
    pids = boot_df["patient_id"].values

    for name, model in models.items():
        try:
            y_proba = model.predict_proba(X_eval)[:, 1]
            y_pred = (y_proba >= THRESHOLD).astype(int)
            utility = physionet_utility(y_eval, y_pred, patient_ids=pids)
            n_alarms = y_pred.sum()

            results[name].append(utility)
            print(f"{name:<15} {iter_idx+1:<6} {utility:<10.4f} {n_alarms:<10}")

        except Exception as e:
            print(f"{name:<15} {iter_idx+1:<6} ERROR: {str(e)[:30]}")
            results[name].append(np.nan)

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for name in models.keys():
    utils = np.array(results[name])
    valid = utils[~np.isnan(utils)]
    if len(valid) > 0:
        mean_u = valid.mean()
        std_u = valid.std()
        passed = "✓" if mean_u > 0.15 else "✗"
        print(f"{name:<15} mean={mean_u:.4f}  std={std_u:.4f}  "
              f"[{valid.min():.4f}, {valid.max():.4f}]  {passed}")
    else:
        print(f"{name:<15} ALL NaN")

print()
