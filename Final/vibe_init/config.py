"""
Central configuration for the fairness pipeline.
Edit values here to tune the analysis; all other modules read from this.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PerturbationConfig:
    # --- Dataset 1: Row removal ---
    # Fraction of the target subgroup to drop (0.8 = keep 20 % of that group).
    # Both 1-A (remove women) and 1-B (remove men) are run by default because the
    # symmetric result — whichever group is underrepresented suffers — strengthens
    # the argument that the issue is structural, not group-specific.
    row_removal_fraction: float = 0.80

    # --- Dataset 2: Missingness-at-random (MAR) ---
    # Number of clinical columns to blank out for the target subgroup.
    mar_n_columns: int = 5
    # Fraction of that subgroup's rows that receive NaN in those columns.
    mar_missing_fraction: float = 0.60
    # Explicit column list; None = auto-select the top-N most-complete numeric cols.
    mar_columns: Optional[List[str]] = None

    # --- Dataset 3: Gaussian noise (optional) ---
    noise_n_columns: int = 5
    # Scale factor applied to each column's empirical std before adding noise.
    noise_std_multiplier: float = 2.0
    noise_columns: Optional[List[str]] = None


@dataclass
class ModelConfig:
    # Keys must match entries in models.MODEL_REGISTRY.
    # Liu et al. (2019) model family used as the primary comparison set.
    models: List[str] = field(default_factory=lambda: [
        "liu_glm",
        "liu_xgboost",   # requires `brew install libomp` on macOS if xgboost fails
        "liu_rnn",
    ])
    random_state: int = 42
    test_size: float = 0.20
    # Imputation strategy for missing values before model fitting
    imputation_strategy: str = "median"   # "median" | "mean" | "knn"
    scale_features: bool = True


@dataclass
class FairnessConfig:
    sensitive_column: str = "Gender"
    # PhysioNet 2019 encoding: 0 = female, 1 = male
    female_value: int = 0
    male_value: int = 1
    # Fairness metrics to compute (see fairness.py for full list)
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy",
        "auroc",
        "tpr",          # recall / sensitivity
        "fpr",          # 1 - specificity
        "precision",
        "f1",
        "tpr_gap",
        "fpr_gap",
        "auroc_gap",
        "precision_gap",
        "demographic_parity_diff",
        "equalized_odds_diff",
        "equal_opportunity_diff",
    ])
    # Positive threshold for binary predictions
    decision_threshold: float = 0.50
    # Clinical utility weights (PhysioNet 2019 challenge reward structure)
    utility_w_tp: float =  1.0   # correctly detected sepsis (life-saving)
    utility_w_fp: float = -0.05  # false alarm (alarm fatigue)
    utility_w_fn: float = -2.0   # missed sepsis (most harmful)
    utility_w_tn: float =  0.0   # correctly ruled out


@dataclass
class BootstrapConfig:
    enabled: bool = False          # set True to add CI columns to results
    n_samples: int = 500
    ci_level: float = 0.95


@dataclass
class MitigationConfig:
    # "none" is always included as the baseline
    strategies: List[str] = field(default_factory=lambda: [
        "none",
        "reweighting",
        "smote",
        "fairness_penalty",
        "robust_model",
    ])


@dataclass
class SweepConfig:
    """
    Parameter sweep for the appendix — vary one axis at a time.
    Set run_sweep=True in Config to activate.
    """
    run_sweep: bool = False
    row_removal_fractions: List[float] = field(
        default_factory=lambda: [0.20, 0.40, 0.60, 0.80]
    )
    mar_missing_fractions: List[float] = field(
        default_factory=lambda: [0.20, 0.40, 0.60, 0.80]
    )
    mar_n_columns_values: List[int] = field(
        default_factory=lambda: [2, 5, 8, 12]
    )


@dataclass
class Config:
    # --- Paths ---
    # Parent folder containing training_setA/ and training_setB/.
    # PSV files are discovered recursively so both sets are pooled automatically.
    data_dir: Path = Path("data/physionet_sepsis")
    output_dir: Path = Path("outputs")

    # --- Sub-configs ---
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fairness: FairnessConfig = field(default_factory=FairnessConfig)
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)

    # --- Optional pipeline steps ---
    run_eda: bool = True       # produce EDA figures
    run_analysis: bool = True  # produce post-hoc analysis figures + tables
    run_report: bool = True    # auto-generate markdown report

    # --- Which perturbation variants to run ---
    run_dataset_1a: bool = True   # remove women
    run_dataset_1b: bool = True   # remove men
    run_dataset_2a: bool = True   # MAR for women
    run_dataset_2b: bool = True   # MAR for men
    run_dataset_3a: bool = True   # noise for women
    run_dataset_3b: bool = True   # noise for men

    random_state: int = 42

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "tables").mkdir(parents=True, exist_ok=True)
