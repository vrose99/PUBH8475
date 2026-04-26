"""
Central configuration for the fairness pipeline (time-series mode).
Edit values here to tune the analysis; all other modules read from this.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelConfig:
    # Keys must match entries in models.MODEL_REGISTRY.
    models: List[str] = field(default_factory=lambda: [
        "liu_glm",
        "liu_xgboost",
        "liu_rnn",
    ])
    random_state: int = 42
    test_size: float = 0.20
    imputation_strategy: str = "median"   # "median" | "mean" | "knn"
    scale_features: bool = True


@dataclass
class FairnessConfig:
    sensitive_column: str = "Gender"
    # PhysioNet 2019 encoding: 0 = female, 1 = male
    female_value: int = 0
    male_value: int = 1
    decision_threshold: float = 0.50
    # Clinical utility weights (PhysioNet 2019 challenge reward structure)
    utility_w_tp: float =  1.0
    utility_w_fp: float = -0.05
    utility_w_fn: float = -2.0
    utility_w_tn: float =  0.0


@dataclass
class MitigationConfig:
    # "none" must be included as the baseline
    strategies: List[str] = field(default_factory=lambda: [
        "none",
        "reweighting",
        "smote",
        "fairness_penalty",
    ])


@dataclass
class Config:
    # --- Paths ---
    data_dir: Path = Path("data/physionet_sepsis")
    output_dir: Path = Path("outputs")

    # --- Sub-configs ---
    model: ModelConfig = field(default_factory=ModelConfig)
    fairness: FairnessConfig = field(default_factory=FairnessConfig)
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)

    # --- Optional pipeline steps ---
    run_analysis: bool = True
    run_report: bool = True

    # --- Dataset variants to run ---
    run_dataset_d0:  bool = True   # parent (forced parity)
    run_dataset_d1a: bool = True   # 75% female rows removed
    run_dataset_d1b: bool = True   # 75% male rows removed
    run_dataset_d2a: bool = True   # 25% MAR for female rows
    run_dataset_d2b: bool = True   # 25% MAR for male rows
    run_dataset_d3a: bool = True   # 25% noise for female rows
    run_dataset_d3b: bool = True   # 25% noise for male rows

    random_state: int = 42

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "tables").mkdir(parents=True, exist_ok=True)
