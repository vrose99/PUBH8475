#!/usr/bin/env python
"""
Utility score diagnostic — run this to understand WHY scores are negative.

Usage:
  /Users/vrose/ClaudeContainer/venv311/bin/python diagnose_utility.py

Explains where utility is won and lost by showing alarm distribution vs
the reward/penalty windows of the PhysioNet 2019 scoring function.
"""
import logging
import sys

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, ".")
from config import Config
from data_loader_timeseries import load_timeseries_dataset, patient_level_split, split_Xy_sensitive
from models import get_model
from preprocessing import build_preprocessor
import warnings

cfg = Config()
rng = np.random.default_rng(42)


def utility_row(dt, y_pred, is_septic,
                dt_early=-12, dt_opt=-6, dt_late=3,
                u_tp=1.0, u_fn=-2.0, u_fp=-0.05, u_tn=0.0):
    """Single-row utility (mirrors physionet_2019_utility internals)."""
    m1 = u_tp  / (dt_opt - dt_early)
    b1 = -m1 * dt_early
    m2 = -u_tp / (dt_late  - dt_opt)
    b2 = -m2 * dt_late

    if not is_septic:
        return u_fp if y_pred else u_tn
    if dt > dt_late:
        return 0.0
    if y_pred:
        if dt <= dt_opt:
            return max(m1 * dt + b1, u_fp)
        else:
            return m2 * dt + b2
    else:
        if dt <= dt_opt:
            return 0.0
        else:
            m3 = u_fn / (dt_late - dt_opt)
            b3 = -m3 * dt_opt
            return m3 * dt + b3


def analyse(y_prob, hours_until_sepsis, threshold=0.5, label="model"):
    y_pred = (y_prob >= threshold).astype(int)
    is_sep = np.isfinite(hours_until_sepsis)
    h_safe = np.where(is_sep, hours_until_sepsis, 0.0)
    dt     = np.where(is_sep, -(h_safe + 6), np.inf)

    # ── Bucket rows ──────────────────────────────────────────────────────────
    buckets = {
        "Non-septic (FP/TN)":        ~is_sep,
        "Septic h>12  (dt<-18)":     is_sep & (h_safe > 12),
        "Septic h∈(6,12] (dt∈[-18,-12])": is_sep & (h_safe > 6)  & (h_safe <= 12),
        "Septic h∈(0, 6] (dt∈[-12, -6])": is_sep & (h_safe > 0)  & (h_safe <= 6),
        "Septic h=0  (onset, dt=-6)": is_sep & (h_safe == 0),
    }

    log.info("\n" + "="*70)
    log.info(f"  UTILITY BREAKDOWN — {label} @ threshold={threshold}")
    log.info("="*70)
    log.info(f"  {'Bucket':<40} {'Rows':>6} {'Alarms':>7} {'Alarm%':>7} {'Unit utility (per alarm)':>24}")
    log.info("  " + "-"*68)

    obs_total  = 0.0
    best_total = 0.0
    rows_total = 0

    for bname, mask in buckets.items():
        n = mask.sum()
        if n == 0:
            continue
        alarms    = y_pred[mask].sum()
        alarm_pct = alarms / n * 100

        # Per-row utility for each alarmed row in this bucket
        u_per_alarm = [
            utility_row(dt[i], 1, is_sep[i])
            for i in np.where(mask & (y_pred == 1))[0]
        ]
        u_per_no_alarm = [
            utility_row(dt[i], 0, is_sep[i])
            for i in np.where(mask & (y_pred == 0))[0]
        ]

        obs_bucket  = sum(u_per_alarm) + sum(u_per_no_alarm)
        obs_total  += obs_bucket

        # Best oracle: predict 1 only when dt ∈ [-12, 3]
        best_bucket = sum(
            utility_row(dt[i], 1, is_sep[i])
            for i in np.where(mask)[0]
            if -12 <= dt[i] <= 3
        ) + sum(
            utility_row(dt[i], 0, is_sep[i])
            for i in np.where(mask)[0]
            if not (-12 <= dt[i] <= 3)
        )
        best_total += best_bucket
        rows_total += n

        avg_u = sum(u_per_alarm) / max(alarms, 1)
        log.info(f"  {bname:<40} {n:>6} {alarms:>7} {alarm_pct:>6.1f}%  avg_u_per_alarm={avg_u:+.3f}   bucket_obs={obs_bucket:+.3f}")

    log.info("  " + "-"*68)
    log.info(f"  {'TOTALS':<40} {rows_total:>6}")
    log.info(f"  Observed utility:  {obs_total:+.3f}")
    log.info(f"  Best utility:      {best_total:+.3f}")
    log.info(f"  Inaction utility:  0.000  (pre-onset only → no FN penalty)")
    denom = best_total
    score = obs_total / denom if denom else float("nan")
    log.info(f"  Normalised score:  {score:+.4f}")
    log.info("")
    log.info("  *** KEY INSIGHT ***")
    log.info("  Rows with h > 6 (dt < dt_early = -12) are OUTSIDE the reward window.")
    log.info("  Any alarm there earns the same penalty as a false positive: -0.05/row.")
    log.info("  If the model alarms too many times outside this window,")
    log.info("  the accumulated penalties exceed the TP reward → negative score.")


# ── Load data ─────────────────────────────────────────────────────────────────
log.info("Loading dataset…")
df_ts = load_timeseries_dataset(cfg, max_patients=300)
train_df, test_df = patient_level_split(df_ts, cfg, rng)
X_train, y_train, s_train, t_train, feat_names = split_Xy_sensitive(train_df, cfg)
X_test,  y_test,  s_test,  t_test,  _          = split_Xy_sensitive(test_df,  cfg)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from preprocessing import build_preprocessor
    pp = build_preprocessor(cfg)
    X_train = pp.fit_transform(X_train)
    X_test  = pp.transform(X_test)

# ── Row-level summary ─────────────────────────────────────────────────────────
h = t_test
is_sep = np.isfinite(h)
log.info("\n" + "="*70)
log.info("  TEST DATA SUMMARY")
log.info("="*70)
log.info(f"  Total rows:           {len(h)}")
log.info(f"  Septic rows:          {is_sep.sum()} ({is_sep.mean()*100:.1f}%)")
log.info(f"    of which h ≤ 6:     {(is_sep & (h <= 6)).sum()} — reward window (u ∈ [0,1])")
log.info(f"    of which 6 < h ≤ 12:{(is_sep & (h > 6) & (h <= 12)).sum()} — partial reward (u ∈ [-0.05, 0])")
log.info(f"    of which h > 12:    {(is_sep & (h > 12)).sum()} — penalty zone (u = -0.05 per alarm)")
log.info(f"  Non-septic rows:      {(~is_sep).sum()} — FP zone (u = -0.05 per alarm)")
log.info(f"  Label prevalence:     {y_test.mean()*100:.1f}%  (rows with target=1 = h ≤ 6)")

# ── GLM analysis ──────────────────────────────────────────────────────────────
log.info("\nFitting liu_glm…")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm = get_model("liu_glm")
    glm.fit(X_train, y_train)
    y_prob_glm = glm.predict_proba(X_test)[:, 1]

analyse(y_prob_glm, t_test, threshold=0.5, label="liu_glm @ 0.5")
analyse(y_prob_glm, t_test, threshold=0.3, label="liu_glm @ 0.3  (lower threshold — more alarms)")
analyse(y_prob_glm, t_test, threshold=0.7, label="liu_glm @ 0.7  (higher threshold — fewer alarms)")
