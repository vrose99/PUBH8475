from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List


ALL_FEATURES: List[str] = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime", "ICULOS",
]

TARGET_COL = "SepsisLabel"


@dataclass
class ExperimentConfig:
    seed: int = 1
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./results_modular")
    test_size: float = 0.30

    # Liu et al. train on a 1-hour window spanning 2 to 1 hours before septic shock onset.
    # With the challenge data we usually only observe SepsisLabel, which turns positive 6 hours
    # before sepsis onset, so event onset is commonly proxied as label_start + 6.
    label_lead_hours: int = 6
    positive_window: Tuple[int, int] = (-2, -1)

    # Sequence settings for the Liu-like RNN wrapper.
    seq_len: int = 8
    epochs: int = 12
    batch_size: int = 128
    learning_rate: float = 1e-3
    rnn_hidden_size: int = 64
    rnn_num_layers: int = 1
    rnn_dropout: float = 0.1

    # Challenge-specific option. Patients with fewer than 8 hours were excluded in the challenge.
    min_patient_hours: int = 8

    recursive: bool = True
    create_missingness_flags: bool = True
    create_delta_features: bool = True

    model_names: List[str] = field(default_factory=lambda: ["glm", "xgboost", "gru"])

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
