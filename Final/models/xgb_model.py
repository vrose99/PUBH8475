from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from models.base import BaseSepsisModel

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


class LiuLikeXGBoost(BaseSepsisModel):
    """XGBoost wrapper aligned with the paper's model family.

    The paper reports using XGBoost as one of the three risk models. The exact tuned parameter
    set is not in the main paper text, so this wrapper uses conservative tree settings suitable
    for hourly EHR data while keeping the interface identical to the other models.
    """

    name = "xgboost"
    is_sequence_model = False

    def __init__(self, random_state: int = 1):
        self.random_state = random_state
        self.pipeline: Pipeline | None = None

    def fit(self, X, y, **kwargs) -> "LiuLikeXGBoost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed.")

        pos = max(1, int(np.sum(y == 1)))
        neg = max(1, int(np.sum(y == 0)))
        scale_pos_weight = neg / pos

        self.pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="auc",
                        random_state=self.random_state,
                        scale_pos_weight=scale_pos_weight,
                        n_jobs=4,
                    ),
                ),
            ]
        )
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model has not been fit.")
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        if self.pipeline is None:
            raise RuntimeError("Model has not been fit.")
        joblib.dump({"random_state": self.random_state, "pipeline": self.pipeline}, path)

    @classmethod
    def load(cls, path: Path, **kwargs) -> "LiuLikeXGBoost":
        payload = joblib.load(path)
        model = cls(random_state=payload["random_state"])
        model.pipeline = payload["pipeline"]
        return model
