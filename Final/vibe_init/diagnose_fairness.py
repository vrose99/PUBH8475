#!/usr/bin/env python
"""Quick fairness diagnostics."""
from __future__ import annotations
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_loader_timeseries import load_timeseries_dataset, patient_level_split, split_Xy_sensitive, get_feature_columns, LABEL_COL_TS
from models import get_model
from preprocessing import build_preprocessor
from fairness_timeseries import physionet_2019_utility

print("\n" + "="*72)
print("  FAIRNESS DIAGNOSTICS")
print("="*72)

cfg = Config()
rng = np.random.default_rng(42)

print("\n[1/7] Loading dataset (100 patients)...")
df = load_timeseries_dataset(cfg, max_patients=100, cache=True)

print("[2/7] Data composition...")
n_patients = df["patient_id"].nunique()
f_df = df[df["Gender"] == 0]
m_df = df[df["Gender"] == 1]
pat_sep_f = df[df["Gender"] == 0].groupby("patient_id")[LABEL_COL_TS].max().mean()
pat_sep_m = df[df["Gender"] == 1].groupby("patient_id")[LABEL_COL_TS].max().mean()

print(f"  Patients: {n_patients} total ({f_df['patient_id'].nunique()}F / {m_df['patient_id'].nunique()}M)")
print(f"  Patient-level sepsis rate: Female {pat_sep_f*100:.1f}% | Male {pat_sep_m*100:.1f}%")
print(f"  Row-level positive rate: {df[LABEL_COL_TS].mean()*100:.1f}%")

print("\n[3/7] Feature set check...")
feat_cols = get_feature_columns(df)
gender_in = "Gender" in feat_cols
age_in = "Age" in feat_cols
print(f"  Total features: {len(feat_cols)}")
print(f"  Gender in features: {gender_in}  {'⚠️  LEAKS ATTRIBUTE' if gender_in else '✓'}")
print(f"  Age in features: {age_in}  {'(may correlate with Gender)' if age_in else ''}")

print("\n[4/7] Train/test split...")
train_df, test_df = patient_level_split(df, cfg, rng)
X_train, y_train, s_train, _, feat_names = split_Xy_sensitive(train_df, cfg)
X_test, y_test, s_test, times_test, _ = split_Xy_sensitive(test_df, cfg)

preprocessor = build_preprocessor(cfg)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

f_mask = s_test == 0
m_mask = s_test == 1
print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")
print(f"  Test: {f_mask.sum()} female rows, {m_mask.sum()} male rows")

print("\n[5/7] Model predictions (liu_glm)...")
try:
    model = get_model("liu_glm")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train_t, y_train)
        y_prob = model.predict_proba(X_test_t)[:, 1]

    print(f"  Mean prob: {y_prob.mean():.4f}")
    print(f"  Female mean prob: {y_prob[f_mask].mean():.4f} | Male: {y_prob[m_mask].mean():.4f}")

    print("\n[6/7] Threshold sweep...")
    print(f"  {'Thresh':>7s}  {'PPR':>6s}  {'DI':>6s}  {'PN_util':>8s}")
    print(f"  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*8}")

    best_thresh = 0.5
    best_util = -999

    for thresh in [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
        y_bin = (y_prob >= thresh).astype(int)
        ppr = y_bin.mean()
        ppr_f = y_bin[f_mask].mean() if f_mask.any() else 0
        ppr_m = y_bin[m_mask].mean() if m_mask.any() else 0
        di = ppr_f / ppr_m if ppr_m > 0 else float("nan")
        util = physionet_2019_utility(times_test, y_bin)

        marker = " ←" if util > best_util else ""
        if util > best_util:
            best_util = util
            best_thresh = thresh

        print(f"  {thresh:>7.2f}  {ppr:>5.1%}  {di:>6.3f}  {util:>8.3f}{marker}")

    print(f"\n  Best utility threshold: {best_thresh:.2f} (utility={best_util:.3f})")

    print("\n[7/7] Feature importance...")
    if hasattr(model, 'coef_'):
        imp = np.abs(model.coef_[0])
        top_idx = np.argsort(imp)[-10:][::-1]
        print(f"  Top 10 features:")
        for i, idx in enumerate(top_idx, 1):
            print(f"    {i:2d}. {feat_names[idx]:40s}  {model.coef_[0][idx]:+.4f}")

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*72)
print("  RECOMMENDATIONS")
print("="*72)
print("""
  If Disparate Impact (DI) < 0.8:
    A. Check if Gender is in features → remove it
    B. Try threshold tuning: use best_util_threshold from above
    C. Use per-group thresholds to equalize alarm rates
    D. With small test set, run with larger sample (--max-patients 2000)
""")
print("="*72 + "\n")
