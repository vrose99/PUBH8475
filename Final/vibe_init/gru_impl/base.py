from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import numpy as np


class BaseSepsisModel(ABC):
    name: str = "base"
    is_sequence_model: bool = False

    @abstractmethod
    def fit(self, X, y, **kwargs) -> "BaseSepsisModel":
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path, **kwargs) -> "BaseSepsisModel":
        raise NotImplementedError
