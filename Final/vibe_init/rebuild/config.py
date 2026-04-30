"""
Fundamental bootstrap configuration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """Training set configuration (for model fitting)."""
    n_patients: int = 100
    random_state: int = 42
    stratify_by_sepsis: bool = True


@dataclass
class BootstrapConfig:
    """Bootstrap resampling configuration."""
    n_iterations: int = 100
    bootstrap_sample_size: int = 50
    random_state: int = 42


@dataclass
class Config:
    """Root configuration."""
    data_dir: Path = Path("data/physionet_sepsis")

    training: TrainingConfig = None
    bootstrap: BootstrapConfig = None

    def __post_init__(self):
        if self.training is None:
            self.training = TrainingConfig()
        if self.bootstrap is None:
            self.bootstrap = BootstrapConfig()
