"""
Unified test for three models with mitigations evaluated on perturbed datasets.

Evaluates 36 model-mitigation-perturbation combinations per bootstrap iteration:
  - 3 models: LogisticGLM, XGBoost, GRU
  - 4 mitigations: none, reweighting, smote, threshold_optimization
  - 3 perturbations: D0 (original), D1A (row removal), D2A (MAR)

Each bootstrap iteration:
  1. Generate ONE evaluation set
  2. Create three dataset variants from training data
  3. Train each of the 12 model-mitigation combos on each dataset variant
  4. All 36 combos evaluated on the SAME evaluation set
  5. Compute fairness metrics for each

Configuration:
    TRAIN_SIZE = 300 patients
    THRESHOLD = 0.4
    BOOTSTRAP_POOL_SIZE = 100 patients
    BOOTSTRAP_EVAL_SIZE = 50 patients per iteration
    N_BOOTSTRAP_ITERATIONS = 5

Outputs three fairness metrics per combination per iteration:
  1. PhysioNet 2019 Utility Score (ideal = 1.0)
  2. Disparate Impact P(alarm|F)/P(alarm|M) (ideal = 1.0)
  3. Equal Opportunity TPR(F)−TPR(M) (ideal = 0)
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
from fairness_metrics import compute_fairness_metrics
from mitigation import get_mitigation
from perturbations import build_all_datasets

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

TRAIN_SIZE = 300
THRESHOLD = 0.4
BOOTSTRAP_POOL_SIZE = 1000
BOOTSTRAP_EVAL_SIZE = 200
N_BOOTSTRAP_ITERATIONS = 10
RANDOM_STATE = 1

MODEL_NAMES = ["LogisticGLM", "XGBoost", "GRU"]
MITIGATION_NAMES = ["none", "reweighting", "smote", "fairness_penalty"]
PERTURBATION_NAMES = ["D0", "D1A", "D2A"]

# Output directory for results
OUTPUT_DIR = Path("./rebuild_outputs")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 100)
print("THREE-MODEL × FOUR-MITIGATION × THREE-PERTURBATION BOOTSTRAP EVALUATION")
print("=" * 100)
print(f"Config:")
print(f"  Train size:        {TRAIN_SIZE} patients")
print(f"  Threshold:         {THRESHOLD}")
print(f"  Bootstrap pool:    {BOOTSTRAP_POOL_SIZE} patients")
print(f"  Bootstrap eval:    {BOOTSTRAP_EVAL_SIZE} patients per iteration")
print(f"  Iterations:        {N_BOOTSTRAP_ITERATIONS}")
print(f"  Models:            {MODEL_NAMES}")
print(f"  Mitigations:       {MITIGATION_NAMES}")
print(f"  Perturbations:     {PERTURBATION_NAMES}")
print(f"  Total combinations: {len(MODEL_NAMES)} × {len(MITIGATION_NAMES)} × {len(PERTURBATION_NAMES)} = {len(MODEL_NAMES) * len(MITIGATION_NAMES) * len(PERTURBATION_NAMES)}")

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

print(f"\nTraining data:")
print(f"  {len(train_df)} rows, {train_df.shape[1]} features")
print(f"  {train_df['patient_id'].nunique()} patients")
print(f"  Bootstrap pool: {len(boot_pids)} patients available")

# Pre-compute perturbation variants (reused across all bootstrap iterations)
print(f"\nBuilding dataset perturbations...")
train_df_perturbed = build_all_datasets(train_df, random_state=RANDOM_STATE)
print(f"  D0 (original):      {len(train_df_perturbed['D0'])} rows, {train_df_perturbed['D0']['patient_id'].nunique()} patients")
print(f"  D1A (row removal):  {len(train_df_perturbed['D1A'])} rows, {train_df_perturbed['D1A']['patient_id'].nunique()} patients")
print(f"  D2A (MAR):          {len(train_df_perturbed['D2A'])} rows, {train_df_perturbed['D2A']['patient_id'].nunique()} patients")

# Initialize results storage: {(model, mitigation, perturbation): {...}}
results = {}
for model_name in MODEL_NAMES:
    for mitigation_name in MITIGATION_NAMES:
        for perturbation_name in PERTURBATION_NAMES:
            key = (model_name, mitigation_name, perturbation_name)
            results[key] = {
                'overall_physionet_utility': [],
                'disparate_impact': [],
                'equal_opportunity': [],
                'n_alarms': []
            }

# Bootstrap evaluation
print(f"\nBootstrap evaluation (threshold={THRESHOLD}):")
print(f"{'Model':<15} {'Mitigation':<20} {'Perturbation':<12} {'Iter':<5} {'PhysioNet_U':<12} {'Disp_Impact':<12} {'Eq_Opp':<12}")
print("-" * 120)

resampler = BootstrapResampler(
    bootstrap_pool_patient_ids=boot_pids,
    full_df=df,
    n_iterations=N_BOOTSTRAP_ITERATIONS,
    bootstrap_sample_size=BOOTSTRAP_EVAL_SIZE,
    random_state=RANDOM_STATE,
)

for iter_idx in tqdm(range(N_BOOTSTRAP_ITERATIONS), desc="Bootstrap iterations", unit="iteration"):
    # Generate ONE evaluation set for this iteration
    _, boot_df = resampler.generate_iteration(iter_idx)
    X_eval, y_eval = extract_Xy(boot_df, "SepsisLabel")
    pids = boot_df["patient_id"].values
    gender = boot_df["Gender"].values
    hours_until_sepsis = boot_df["hours_until_sepsis"].values if "hours_until_sepsis" in boot_df.columns else np.full(len(boot_df), np.nan)

    # For each model-mitigation-perturbation combination
    for model_name in MODEL_NAMES:
        for mitigation_name in MITIGATION_NAMES:
            for perturbation_name in PERTURBATION_NAMES:
                try:
                    # Get perturbed training data
                    train_df_variant = train_df_perturbed[perturbation_name]
                    X_train, y_train = extract_Xy(train_df_variant, "SepsisLabel")
                    s_train = train_df_variant["Gender"].values

                    # Create fresh model instance
                    if model_name == "LogisticGLM":
                        model = LogisticGLM(C=0.1)
                    elif model_name == "XGBoost":
                        model = XGBoostModel()
                    elif model_name == "GRU":
                        model = GRUModel()
                    else:
                        raise ValueError(f"Unknown model: {model_name}")

                    # Apply mitigation to training data
                    mitigation_fn = get_mitigation(mitigation_name)
                    mitigated_model = None

                    if mitigation_name == 'fairness_penalty':
                        # Fairness penalty returns a fitted model
                        X_train_mit, y_train_mit, sample_weights, mitigated_model = mitigation_fn(
                            X_train, y_train, s_train, model=model
                        )
                    else:
                        # Other mitigations return modified data/weights
                        X_train_mit, y_train_mit, sample_weights = mitigation_fn(X_train, y_train, s_train)

                    # Fit model on mitigated training data with optional sample weights
                    # (unless using fairness_penalty which returns a pre-fitted model)
                    if mitigated_model is None:
                        model.fit(X_train_mit, y_train_mit, sample_weight=sample_weights,
                                 mitigation=mitigation_name)
                    else:
                        model = mitigated_model

                    # Evaluate on (same) bootstrap evaluation set
                    y_proba = model.predict_proba(X_eval)[:, 1]
                    y_pred = (y_proba >= THRESHOLD).astype(int)

                    # Compute fairness metrics
                    fairness = compute_fairness_metrics(
                        y_true=y_eval,
                        y_prob=y_proba,
                        y_pred=y_pred,
                        sensitive=gender,
                        patient_ids=pids,
                        hours_until_sepsis=hours_until_sepsis,
                        threshold=THRESHOLD,
                    )

                    n_alarms = y_pred.sum()

                    # Store results
                    key = (model_name, mitigation_name, perturbation_name)
                    results[key]['overall_physionet_utility'].append(fairness['overall_physionet_utility'])
                    results[key]['disparate_impact'].append(fairness['disparate_impact'])
                    results[key]['equal_opportunity'].append(fairness['equal_opportunity'])
                    results[key]['n_alarms'].append(n_alarms)

                    print(f"{model_name:<15} {mitigation_name:<20} {perturbation_name:<12} {iter_idx+1:<5} {fairness['overall_physionet_utility']:<12.4f} {fairness['disparate_impact']:<12.4f} {fairness['equal_opportunity']:<12.4f}")

                except Exception as e:
                    print(f"{model_name:<15} {mitigation_name:<20} {perturbation_name:<12} {iter_idx+1:<5} ERROR: {str(e)[:30]}")

# Summary
print("\n" + "=" * 120)
print("SUMMARY STATISTICS")
print("=" * 120)

metrics_to_print = [
    ('overall_physionet_utility', 'PhysioNet Utility (ideal=1.0)'),
    ('disparate_impact', 'Disparate Impact (ideal=1.0)'),
    ('equal_opportunity', 'Equal Opportunity (ideal=0.0)'),
]

for model_name in MODEL_NAMES:
    print(f"\n{model_name}:")
    print(f"  {'Mitigation':<20} {'Perturbation':<12} {'Metric':<40} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print(f"  {'-'*100}")

    for mitigation_name in MITIGATION_NAMES:
        for perturbation_name in PERTURBATION_NAMES:
            key = (model_name, mitigation_name, perturbation_name)

            for metric_key, metric_label in metrics_to_print:
                values = np.array(results[key][metric_key])
                valid = values[~np.isnan(values)]

                if len(valid) > 0:
                    mean_val = valid.mean()
                    std_val = valid.std()
                    min_val = valid.min()
                    max_val = valid.max()
                    print(f"  {mitigation_name:<20} {perturbation_name:<12} {metric_label:<40} {mean_val:<10.4f} {std_val:<10.4f} {min_val:<10.4f} {max_val:<10.4f}")
                else:
                    print(f"  {mitigation_name:<20} {perturbation_name:<12} {metric_label:<40} ALL NaN")

            # Pass/fail check on utility
            utils = np.array(results[key]['overall_physionet_utility'])
            valid_utils = utils[~np.isnan(utils)]
            if len(valid_utils) > 0:
                mean_u = valid_utils.mean()
                passed = "✓" if mean_u > 0.15 else "✗"
                print(f"  {mitigation_name:<20} {perturbation_name:<12} Utility threshold (>0.15): {passed} (mean={mean_u:.4f})")

print("\n" + "=" * 120)
print("SAVING RESULTS")
print("=" * 120)

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

# Convert results to long format for CSV export
import pickle
results_long = []

for (model_name, mitigation_name, perturbation_name), metrics in results.items():
    for i in range(N_BOOTSTRAP_ITERATIONS):
        results_long.append({
            'iteration': i + 1,
            'model': model_name,
            'mitigation': mitigation_name,
            'perturbation': perturbation_name,
            'overall_physionet_utility': metrics['overall_physionet_utility'][i] if i < len(metrics['overall_physionet_utility']) else np.nan,
            'disparate_impact': metrics['disparate_impact'][i] if i < len(metrics['disparate_impact']) else np.nan,
            'equal_opportunity': metrics['equal_opportunity'][i] if i < len(metrics['equal_opportunity']) else np.nan,
            'n_alarms': metrics['n_alarms'][i] if i < len(metrics['n_alarms']) else np.nan,
        })

# Save as CSV
import pandas as pd
df_results = pd.DataFrame(results_long)
csv_path = OUTPUT_DIR / "tables" / "results_all.csv"
df_results.to_csv(csv_path, index=False)
print(f"✓ Saved results CSV: {csv_path}")

# Save as pickle for notebook import
pickle_path = OUTPUT_DIR / "results.pkl"
with open(pickle_path, 'wb') as f:
    pickle.dump(results, f)
print(f"✓ Saved results pickle: {pickle_path}")

# Create summary CSV (mean stats per combination)
summary_data = []
for (model_name, mitigation_name, perturbation_name), metrics in results.items():
    utils = np.array(metrics['overall_physionet_utility'])
    valid_utils = utils[~np.isnan(utils)]

    di = np.array(metrics['disparate_impact'])
    valid_di = di[~np.isnan(di)]

    eo = np.array(metrics['equal_opportunity'])
    valid_eo = eo[~np.isnan(eo)]

    summary_data.append({
        'model': model_name,
        'mitigation': mitigation_name,
        'perturbation': perturbation_name,
        'utility_mean': valid_utils.mean() if len(valid_utils) > 0 else np.nan,
        'utility_std': valid_utils.std() if len(valid_utils) > 0 else np.nan,
        'disparate_impact_mean': valid_di.mean() if len(valid_di) > 0 else np.nan,
        'disparate_impact_std': valid_di.std() if len(valid_di) > 0 else np.nan,
        'equal_opportunity_mean': valid_eo.mean() if len(valid_eo) > 0 else np.nan,
        'equal_opportunity_std': valid_eo.std() if len(valid_eo) > 0 else np.nan,
    })

df_summary = pd.DataFrame(summary_data)
summary_path = OUTPUT_DIR / "tables" / "summary_statistics.csv"
df_summary.to_csv(summary_path, index=False)
print(f"✓ Saved summary CSV: {summary_path}")

print(f"\nOutput directory: {OUTPUT_DIR.resolve()}")
print()
