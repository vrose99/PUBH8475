from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.base import BaseSepsisModel


class LiuLikeGLM(BaseSepsisModel):
    """L1-regularized logistic regression.

    Liu et al. report a GLM with lasso-based feature selection and 10-fold cross-validation.
    This wrapper uses `LogisticRegressionCV` with L1 penalty, liblinear solver, balanced class
    weights, standardization, and 10-fold CV to stay close to that description.
    """

    name = "glm"
    is_sequence_model = False

    def __init__(self, random_state: int = 1):
        self.random_state = random_state
        self.pipeline: Pipeline | None = None

    def fit(self, X, y, **kwargs) -> "LiuLikeGLM":
        self.pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegressionCV(
                        Cs=10,
                        cv=10,
                        penalty="l1",
                        solver="liblinear",
                        scoring="roc_auc",
                        class_weight="balanced",
                        max_iter=4000,
                        random_state=self.random_state,
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
    def load(cls, path: Path, **kwargs) -> "LiuLikeGLM":
        payload = joblib.load(path)
        model = cls(random_state=payload["random_state"])
        model.pipeline = payload["pipeline"]
        return model
