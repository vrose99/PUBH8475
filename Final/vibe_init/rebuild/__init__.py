"""
Fundamental bootstrap rebuild for PhysioNet sepsis analysis.

Core modules:
  config -- Configuration classes (TrainingConfig, BootstrapConfig, Config)
  data_loader -- PhysioNet data loading and patient-level splitting
  bootstrap -- Bootstrap resampling mechanism (BootstrapResampler)
  models -- Model classes (LogisticGLM, XGBoostModel, GRUModel)
  training -- Training and evaluation utilities (BootstrapEvaluator)
  utility -- PhysioNet 2019 utility function and evaluation
  fairness_eval -- Per-group fairness metrics and reporting

Examples & tests:
  main -- Data loading and bootstrap setup
  fit_and_evaluate -- Complete workflow: train and evaluate all models
  test_full_pipeline -- Test with GLM
  test_utility_evaluation -- Test with utility metrics
  example_usage -- Usage patterns
  fairness_eval -- Fairness evaluation with per-group metrics
"""

from config import Config, TrainingConfig, BootstrapConfig
from data_loader import (
    load_physionet_files,
    add_hours_until_sepsis,
    split_patients_by_status,
    get_rows_for_patients,
)
from bootstrap import BootstrapResampler, summarize_bootstrap_pool
from models import LogisticGLM, XGBoostModel, GRUModel, get_model
from training import BootstrapEvaluator, extract_Xy, evaluate_model
from utility import physionet_utility, evaluate_utility_at_threshold, find_optimal_threshold, UtilityEvaluator
from fairness_eval import (
    evaluate_fairness_gap,
    aggregate_fairness_across_bootstrap,
    print_fairness_report,
    fairness_summary_table,
)

__all__ = [
    "Config",
    "TrainingConfig",
    "BootstrapConfig",
    "load_physionet_files",
    "split_patients_by_status",
    "get_rows_for_patients",
    "BootstrapResampler",
    "summarize_bootstrap_pool",
    "LogisticGLM",
    "XGBoostModel",
    "GRUModel",
    "get_model",
    "BootstrapEvaluator",
    "extract_Xy",
    "evaluate_model",
    "evaluate_fairness_gap",
    "aggregate_fairness_across_bootstrap",
    "print_fairness_report",
    "fairness_summary_table",
]
