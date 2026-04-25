"""
Main pipeline orchestrator.

Usage:
  python pipeline.py                          # run with defaults
  python pipeline.py --max-patients 2000      # quick smoke test
  python pipeline.py --sweep                  # also run parameter sweep
  python pipeline.py --bootstrap              # add CI columns (slow)
  python pipeline.py --data-dir path/to/psv   # custom data location

Outputs land in outputs/figures/ and outputs/tables/.
"""

# ── Quick-run flag ────────────────────────────────────────────────────────────
# Set TEST = True to run a fast end-to-end smoke-test (~5–10 minutes).
# Uses a stratified sample of 2 000 patients, only the liu_glm model, and
# skips EDA, post-hoc analysis, parameter sweep, and the markdown report.
# Flip to False (or use --no-test) for the full production run.
TEST = True

# ── Time-series mode ──────────────────────────────────────────────────────────
# Set USE_TIMESERIES = True to switch from static (aggregated) patient-level
# classification to time-series early-detection classification aligned with the
# PhysioNet 2019 challenge utility function.
#
# Static mode (False):
#   - Aggregates features across entire ICU stay (mean, std, min, max, last)
#   - One prediction per patient: "did/will this patient develop sepsis?"
#   - Fairness: per-group TPR, FPR, AUC gaps
#
# Time-series mode (True):
#   - Builds hourly sequences with rolling statistics (6-hour windows)
#   - One prediction per hour: "will sepsis occur in the next 6 hours?"
#   - Fairness: per-group early-warning-time gaps, alarm-fatigue gaps
#   - Aligns with PhysioNet 2019 challenge's early-detection utility function
#
USE_TIMESERIES = False
# ─────────────────────────────────────────────────────────────────────────────

# ── macOS fork-safety fix ─────────────────────────────────────────────────────
# Must happen before any C extension (numpy, sklearn, joblib) is imported.
# Forking after Objective-C runtime initialisation causes SIGSEGV in worker
# threads (EXC_BAD_ACCESS / KERN_INVALID_ADDRESS at 0x0, thread 20+).
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def parse_args():
    p = argparse.ArgumentParser(description="Fairness pipeline for PhysioNet sepsis data")
    p.add_argument("--data-dir",      type=Path, default=None)
    p.add_argument("--output-dir",    type=Path, default=Path("outputs"))
    p.add_argument("--max-patients",  type=int,  default=None,
                   help="Cap patient count for smoke-testing")
    p.add_argument("--sweep",         action="store_true",
                   help="Run parameter sweep (appendix figures)")
    p.add_argument("--bootstrap",     action="store_true",
                   help="Add bootstrap confidence intervals to results (slow)")
    p.add_argument("--no-eda",        action="store_true",
                   help="Skip EDA figures")
    p.add_argument("--no-analysis",   action="store_true",
                   help="Skip post-hoc analysis figures")
    p.add_argument("--no-report",     action="store_true",
                   help="Skip markdown report generation")
    p.add_argument("--no-cache",      action="store_true",
                   help="Ignore cached aggregated dataset")
    p.add_argument("--models",        nargs="+", default=None,
                   help="Override model list, e.g. --models logistic_regression xgboost")
    p.add_argument("--mitigations",   nargs="+", default=None)
    p.add_argument("--seed",          type=int,  default=42)
    p.add_argument("--no-test",       action="store_true",
                   help="Override the TEST flag and run the full production pipeline")
    return p.parse_args()


_TEST_MAX_PATIENTS  = 300   # patients to load in test mode
_TEST_MODELS        = ["liu_glm", "liu_xgboost","liu_rnn"]
_TEST_MITIGATIONS   = ["none", "reweighting","smote","fairness_penalty"]  # skip slow fairness_penalty/smote

def main():
    args = parse_args()

    # ── Resolve TEST mode ─────────────────────────────────────────────────────
    run_test = TEST and not args.no_test
    if run_test:
        logger.info(
            "TEST MODE — %d patients, models=%s, mitigations=%s. "
            "Set TEST=False in pipeline.py or pass --no-test for a full run.",
            _TEST_MAX_PATIENTS, _TEST_MODELS, _TEST_MITIGATIONS,
        )

    # ── Config ────────────────────────────────────────────────────────────────
    from config import Config

    cfg = Config()
    cfg.random_state = args.seed
    cfg.output_dir   = args.output_dir
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.models:
        cfg.model.models = args.models
    elif run_test:
        cfg.model.models = _TEST_MODELS
    if args.mitigations:
        cfg.mitigation.strategies = args.mitigations
    elif run_test:
        cfg.mitigation.strategies = _TEST_MITIGATIONS
    if args.sweep:
        cfg.sweep.run_sweep = True
    if args.bootstrap:
        cfg.bootstrap.enabled = True
    # In test mode disable slow optional steps unless explicitly requested.
    if run_test:
        cfg.run_eda      = False
        cfg.run_analysis = False
        cfg.run_report   = False
    if args.no_eda:
        cfg.run_eda = False
    if args.no_analysis:
        cfg.run_analysis = False
    if args.no_report:
        cfg.run_report = False

    # trigger directory creation
    cfg.__post_init__()

    rng = np.random.default_rng(cfg.random_state)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("=== Step 1/5 — Loading data ===")
    max_patients = args.max_patients or (_TEST_MAX_PATIENTS if run_test else None)
    try:
        if USE_TIMESERIES:
            from data_loader_timeseries import load_timeseries_dataset
            logger.info("TIME-SERIES MODE: hourly early-detection predictions")
            df = load_timeseries_dataset(cfg, max_patients=max_patients, cache=not args.no_cache)
        else:
            from data_loader import load_dataset
            logger.info("STATIC MODE: patient-level aggregated predictions")
            df = load_dataset(cfg, max_patients=max_patients, cache=not args.no_cache)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    # ── Build dataset variants (static mode only) ─────────────────────────────
    if not USE_TIMESERIES:
        from perturbations import build_all_datasets

        logger.info("=== Step 2/5 — Building dataset variants ===")
        datasets = build_all_datasets(df, cfg, rng)

        # ── EDA (static mode only) ────────────────────────────────────────────
        if cfg.run_eda:
            from eda import run_eda
            logger.info("=== Step 2b — EDA ===")
            run_eda(df, datasets, cfg)
    else:
        # Time-series mode: no perturbations, just use the raw dataset
        datasets = {"original": df}
        logger.info("=== Step 2/5 — Skipped (time-series mode) ===")

    # ── Run evaluation ────────────────────────────────────────────────────────
    from joblib import parallel_backend

    logger.info(
        "=== Step 3/5 — Evaluation (%d cells) ===",
        len(datasets) * len(cfg.model.models) * len(cfg.mitigation.strategies),
    )

    # Force sequential joblib backend — prevents all forked worker threads that
    # cause SIGSEGV on macOS (EXC_BAD_ACCESS in loky/OpenMP worker threads).
    with parallel_backend("sequential"):
        if USE_TIMESERIES:
            from evaluation_timeseries import run_timeseries_evaluation
            results = run_timeseries_evaluation(df, cfg, rng)
        else:
            from evaluation import run_evaluation, run_parameter_sweep
            results = run_evaluation(datasets, cfg, rng)

    # ── Optional: bootstrap CIs ───────────────────────────────────────────────
    if cfg.bootstrap.enabled:
        logger.info("=== Step 3b — Bootstrap confidence intervals ===")
        _add_bootstrap_cis(results, df, datasets, cfg, rng)

    results_path = cfg.output_dir / "tables" / "results_all.csv"
    results.to_csv(results_path, index=False)
    logger.info("Raw results saved to %s", results_path)

    sweep_results: pd.DataFrame = pd.DataFrame()
    if cfg.sweep.run_sweep and not USE_TIMESERIES:
        logger.info("=== Running parameter sweep (appendix) ===")
        from evaluation import run_parameter_sweep
        with parallel_backend("sequential"):
            sweep_results = run_parameter_sweep(df, cfg, rng)
        sweep_path = cfg.output_dir / "tables" / "results_sweep.csv"
        sweep_results.to_csv(sweep_path, index=False)
        logger.info("Sweep results saved to %s", sweep_path)
    elif cfg.sweep.run_sweep and USE_TIMESERIES:
        logger.info("Parameter sweep not implemented for time-series mode — skipping")

    # ── Visualise ─────────────────────────────────────────────────────────────
    from visualization import render_all

    logger.info("=== Step 4/5 — Figures & tables ===")
    render_all(
        results,
        cfg,
        sweep_results=sweep_results if not sweep_results.empty else None,
    )

    # ── Post-hoc analysis ─────────────────────────────────────────────────────
    analysis_summary: dict = {}
    if cfg.run_analysis:
        from analysis import run_analysis
        logger.info("=== Step 4b — Post-hoc analysis ===")
        analysis_summary = run_analysis(results, cfg)

    # ── Report ────────────────────────────────────────────────────────────────
    if cfg.run_report:
        from report import generate_report
        logger.info("=== Step 5/5 — Generating report ===")
        report_path = generate_report(results, cfg, analysis_summary=analysis_summary)
        logger.info("Report: %s", report_path)

    # ── Quick console summary ─────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("SUMMARY — equalized_odds_diff (baseline, no mitigation)")
    logger.info("%s", "=" * 60)
    baseline = results[results["mitigation"] == "none"]
    if "equalized_odds_diff" in baseline.columns:
        summary = (
            baseline.groupby(["dataset_id", "model"])["equalized_odds_diff"]
            .mean()
            .unstack("model")
            .round(3)
        )
        logger.info("\n%s\n", summary.to_string())

    if "worst_group_auroc" in results.columns:
        logger.info("\n%s", "=" * 60)
        logger.info("WORST-GROUP AUROC (baseline) — illusion-of-fairness check")
        logger.info("%s", "=" * 60)
        wg = (
            baseline.groupby(["dataset_id", "model"])["worst_group_auroc"]
            .mean()
            .unstack("model")
            .round(3)
        )
        logger.info("\n%s\n", wg.to_string())

    logger.info("Pipeline complete. Outputs: %s", cfg.output_dir.resolve())


def _add_bootstrap_cis(
    results: pd.DataFrame,
    df_clean: pd.DataFrame,
    datasets: dict,
    cfg,
    rng: np.random.Generator,
):
    """
    Re-compute fairness metrics with bootstrap CIs and merge into results in-place.
    This is expensive: it runs n_samples resamplings per (dataset, model, mitigation) cell.
    """
    from fairness import bootstrap_fairness_report
    from preprocessing import build_preprocessor, split_features_label, train_test_split_stratified
    from data_loader import LABEL_COL
    import warnings

    logger.info(
        "Bootstrap: %d samples, %.0f%% CI across %d cells",
        cfg.bootstrap.n_samples, cfg.bootstrap.ci_level * 100, len(results),
    )

    ci_records = []
    for _, row in results.iterrows():
        did   = row["dataset_id"]
        model = row["model"]
        mit   = row["mitigation"]

        df = datasets.get(did)
        if df is None:
            continue

        feature_cols = [
            c for c in df.columns
            if c not in {LABEL_COL, "dataset_id"}
        ]
        _, test_df = train_test_split_stratified(df, cfg, rng)
        X_test, y_test, s_test, _ = split_features_label(test_df, cfg, feature_cols)

        preprocessor = build_preprocessor(cfg)
        train_df, _ = train_test_split_stratified(df, cfg, rng)
        X_train, y_train, _, _ = split_features_label(train_df, cfg, feature_cols)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preprocessor.fit(X_train)
            X_test_t = preprocessor.transform(X_test)

        # We need the stored predictions — re-generate them from the fitted model.
        # This is a best-effort approach: we just bootstrap the test-set predictions
        # that were already computed, stored indirectly via the metric values.
        # For a full bootstrap we'd need to store y_prob, which we skip here to keep
        # the evaluation loop simple.  Instead we bootstrap the per-group metrics
        # using the overall point estimates already in the row as proxies.
        # Full implementation would require caching (y_prob, y_true, sensitive) per cell.
        ci_records.append({"dataset_id": did, "model": model, "mitigation": mit})

    logger.info(
        "Bootstrap CI generation requires caching predictions per cell — "
        "set cfg.bootstrap.enabled=True with a modified evaluation loop "
        "that stores y_prob for post-hoc bootstrapping.  Skipping for now."
    )


if __name__ == "__main__":
    main()
