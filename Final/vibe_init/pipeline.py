"""
Main pipeline orchestrator (time-series mode).

Usage:
  python pipeline.py                          # run with defaults
  python pipeline.py --max-patients 2000      # quick smoke test
  python pipeline.py --data-dir path/to/psv   # custom data location

Pipeline phases:
  1. Load time-series dataset
  2. Phase A — baseline: evaluate 3 models with no mitigation
     → PhysioNet utility score + 3 fairness metrics
  3. Phase B — mitigated: apply each mitigation strategy and re-evaluate
     → same metrics, comparison against baseline
  4. Visualise and optionally generate report

Outputs land in outputs/figures/ and outputs/tables/.

 pipeline.py --no-test
"""

# ── Quick-run flag ────────────────────────────────────────────────────────────
# Set TEST = True to run a fast end-to-end smoke-test (~5–10 minutes).
# Uses a stratified sample of patients and only liu_glm.
# Flip to False (or use --no-test) for the full production run.
TEST = True

# ── macOS fork-safety fix ─────────────────────────────────────────────────────
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
    p = argparse.ArgumentParser(description="Fairness pipeline — PhysioNet time-series mode")
    p.add_argument("--data-dir",     type=Path, default=None)
    p.add_argument("--output-dir",   type=Path, default=Path("outputs"))
    p.add_argument("--max-patients", type=int,  default=None,
                   help="Cap patient count for smoke-testing")
    p.add_argument("--no-analysis",  action="store_true",
                   help="Skip post-hoc analysis figures")
    p.add_argument("--no-report",    action="store_true",
                   help="Skip markdown report generation")
    p.add_argument("--no-cache",     action="store_true",
                   help="Ignore cached aggregated dataset")
    p.add_argument("--models",       nargs="+", default=None,
                   help="Override model list, e.g. --models liu_glm liu_xgboost")
    p.add_argument("--mitigations",  nargs="+", default=None)
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--no-test",      action="store_true",
                   help="Override the TEST flag and run the full production pipeline")
    p.add_argument("--bootstrap",    action="store_true",
                   help="Enable bootstrap resampling for metric estimates")
    p.add_argument("--bootstrap-iters", type=int, default=100,
                   help="Number of bootstrap iterations (default: 100)")
    return p.parse_args()


_TEST_MAX_PATIENTS = 100
_TEST_MODELS       = ["liu_glm", "liu_xgboost", "liu_rnn"]
_TEST_MITIGATIONS  = ["none", "reweighting", "smote", "threshold_optimization"]


def _print_phase(label: str, results: pd.DataFrame):
    """Print a formatted console summary for one evaluation phase."""
    sep = "=" * 68
    logger.info("\n%s", sep)
    logger.info("%s", label)
    logger.info("%s", sep)

    def _pivot(col):
        if col not in results.columns:
            return None
        return (
            results.groupby(["model", "mitigation"])[col]
            .mean()
            .unstack("mitigation")
            .round(3)
        )

    metrics = [
        ("PhysioNet 2019 Utility Score  [ideal = 1.0]",
         "overall_physionet_utility"),
        ("Disparate Impact  P(alarm|F)/P(alarm|M)  [ideal = 1.0]",
         "disparate_impact"),
        ("Equal Opportunity  TPR(F)−TPR(M)  [ideal = 0]",
         "equal_opportunity"),
    ]
    for title, col in metrics:
        tbl = _pivot(col)
        if tbl is not None:
            logger.info("\n%s\n%s\n", title, tbl.to_string())


def main():
    args = parse_args()

    run_test = TEST and not args.no_test
    if run_test:
        logger.info(
            "TEST MODE — %d patients, models=%s, mitigations=%s. "
            "Pass --no-test for a full run.",
            _TEST_MAX_PATIENTS, _TEST_MODELS, _TEST_MITIGATIONS,
        )

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
    if run_test:
        cfg.run_analysis = False
        cfg.run_report = False
        logger.info("TEST MODE: analysis and report generation disabled")
    if args.no_analysis:
        cfg.run_analysis = False
    if args.no_report:
        cfg.run_report = False

    cfg.bootstrap.enabled = args.bootstrap
    cfg.bootstrap.n_iterations = args.bootstrap_iters

    cfg.__post_init__()
    rng = np.random.default_rng(cfg.random_state)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    logger.info("=== Step 1/4 — Loading time-series dataset ===")
    from data_loader_timeseries import load_timeseries_dataset

    max_patients = args.max_patients or (_TEST_MAX_PATIENTS if run_test else None)
    try:
        df = load_timeseries_dataset(cfg, max_patients=max_patients, cache=not args.no_cache)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info(
        "Loaded %d patient-hours from %d patients",
        len(df), df["patient_id"].nunique(),
    )

    # ── Step 1b: Clinical feature selection (reduces 431 → ~60-80 features) ────
    logger.info("=== Step 1b/4 — Clinical feature selection ===")
    from feature_selection import select_and_engineer_features

    n_features_before = len([c for c in df.columns if c not in {"patient_id", "hour", "SepsisLabel", "target"}])
    df = select_and_engineer_features(df)
    n_features_after = len([c for c in df.columns if c not in {"patient_id", "hour", "SepsisLabel", "target"}])

    logger.info(
        "Feature selection: %d → %d features (%.1f%% reduction)",
        n_features_before, n_features_after,
        100 * (1 - n_features_after / n_features_before) if n_features_before > 0 else 0,
    )

    # ── Step 2: Build dataset variants ──────────────────────────────────────
    from joblib import parallel_backend
    from evaluation_timeseries import run_timeseries_evaluation
    from perturbations import build_all_datasets

    logger.info("=== Step 2/4 — Building dataset variants ===")
    datasets = build_all_datasets(df, cfg, rng)

    # Filter by config flags
    dataset_flags = {
        "D0": cfg.run_dataset_d0,
        "D1A": cfg.run_dataset_d1a,
        "D2A": cfg.run_dataset_d2a,
    }
    datasets = {did: dff for did, dff in datasets.items() if dataset_flags.get(did, True)}

    # ── Step 3: Phase A — baseline (no mitigation) ───────────────────────────
    logger.info(
        "=== Step 3/4 — Phase A: baseline evaluation (%d datasets × %d models, no mitigation) ===",
        len(datasets), len(cfg.model.models),
    )

    baseline_cfg = Config()
    baseline_cfg.data_dir = cfg.data_dir
    baseline_cfg.output_dir = cfg.output_dir
    baseline_cfg.random_state = cfg.random_state
    baseline_cfg.model = cfg.model
    baseline_cfg.fairness = cfg.fairness
    baseline_cfg.mitigation.strategies = ["none"]
    baseline_cfg.run_analysis = cfg.run_analysis
    baseline_cfg.run_report = cfg.run_report

    baseline_results_list = []
    with parallel_backend("sequential"):
        for dataset_id, df_variant in datasets.items():
            result = run_timeseries_evaluation(df_variant, baseline_cfg, rng, dataset_id=dataset_id)
            baseline_results_list.append(result)
    baseline_results = pd.concat(baseline_results_list, ignore_index=True) if baseline_results_list else pd.DataFrame()

    _print_phase("PHASE A — Baseline (no mitigation)", baseline_results)

    # ── Step 4: Phase B — with mitigation strategies ─────────────────────────
    mitigations = [m for m in cfg.mitigation.strategies if m != "none"]
    mitigated_results = pd.DataFrame()

    if mitigations:
        logger.info(
            "=== Step 4/4 — Phase B: mitigated evaluation (%d datasets × %d models × %d mitigations) ===",
            len(datasets), len(cfg.model.models), len(mitigations),
        )
        mitigated_cfg = Config()
        mitigated_cfg.data_dir = cfg.data_dir
        mitigated_cfg.output_dir = cfg.output_dir
        mitigated_cfg.random_state = cfg.random_state
        mitigated_cfg.model = cfg.model
        mitigated_cfg.fairness = cfg.fairness
        mitigated_cfg.mitigation.strategies = mitigations
        mitigated_cfg.run_analysis = cfg.run_analysis
        mitigated_cfg.run_report = cfg.run_report

        mitigated_results_list = []
        with parallel_backend("sequential"):
            for dataset_id, df_variant in datasets.items():
                result = run_timeseries_evaluation(df_variant, mitigated_cfg, rng, dataset_id=dataset_id)
                mitigated_results_list.append(result)
        mitigated_results = pd.concat(mitigated_results_list, ignore_index=True) if mitigated_results_list else pd.DataFrame()

        _print_phase("PHASE B — After mitigation", mitigated_results)
    else:
        logger.info("=== Step 4/4 — No mitigation strategies configured; skipping Phase B ===")

    # Combine both phases for downstream visualisation / analysis
    results = pd.concat(
        [r for r in [baseline_results, mitigated_results] if not r.empty],
        ignore_index=True,
    )

    results_path = cfg.output_dir / "tables" / "results_all.csv"
    results.to_csv(results_path, index=False)
    logger.info("Full results saved to %s", results_path)

    # ── Step 5: Visualise & report ────────────────────────────────────────────
    logger.info("=== Step 5/5 — Figures, tables, and report ===")

    from visualization import render_all
    render_all(results, cfg)

    if cfg.run_analysis:
        from analysis import run_analysis
        run_analysis(results, cfg)

    if cfg.run_report:
        from report import generate_report
        report_path = generate_report(results, cfg)
        logger.info("Report: %s", report_path)

    logger.info("Pipeline complete. Outputs: %s", cfg.output_dir.resolve())


if __name__ == "__main__":
    main()
