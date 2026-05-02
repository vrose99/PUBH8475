"""
Model classes for sepsis risk prediction.

Each model:
- Fits on training data (one row per patient-hour)
- Predicts hourly sepsis risk
- Handles missing values via imputation
- Returns probabilities and binary predictions

Models:
  LogisticGLM -- L1-regularized logistic regression (sparse features)
  XGBoostModel -- Gradient boosted trees
  GRUModel -- Recurrent neural network (if PyTorch available)
"""

import logging
from typing import Tuple
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    XGBClassifier = None

try:
    import torch
    from torch import nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class BaseModel:
    """
    Abstract base for all model wrappers.

    Subclasses must implement:
      fit(X_train, y_train) → self
      predict_proba(X) → (n_samples, 2) probability array
      predict(X) → (n_samples,) binary predictions
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_names = None
        self.n_features = None

    def _preprocess_data(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle missing values and scale features.

        X can be array-like or DataFrame.
        y is optional (only required at fit time).
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values

        X = np.asarray(X, dtype=float)

        if y is not None:
            y = np.asarray(y, dtype=int)
            if len(X) != len(y):
                raise ValueError(f"X and y have mismatched lengths: {len(X)} vs {len(y)}")

        self.n_features = X.shape[1]

        return X, y

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        """Binary prediction (threshold at 0.5)."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class LogisticGLM(BaseModel):
    """
    L1-regularized logistic regression (sparse GLM).

    Handles:
    - Missing value imputation (median per feature)
    - Feature scaling
    - L1 regularization for feature selection
    - Class imbalance via balanced class weights
    """

    def __init__(self, C: float = 0.01, random_state: int = 42):
        """
        Args:
            C: Inverse regularization strength (smaller = more regularization)
            random_state: Random seed
        """
        super().__init__(random_state)
        self.C = C
        self.model = LogisticRegression(
            C=C,
            penalty="l1",
            solver="saga",
            class_weight="balanced",
            max_iter=10000,
            tol=1e-3,
            random_state=random_state,
            n_jobs=1,
        )
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self._is_fit = False

    def fit(self, X, y, sample_weight=None):
        """Fit on training data."""
        X, y = self._preprocess_data(X, y)

        # Impute missing values
        X = self.imputer.fit_transform(X)

        # Scale features
        X = self.scaler.fit_transform(X)

        # Fit logistic regression with optional sample weights
        self.model.fit(X, y, sample_weight=sample_weight)
        self._is_fit = True

        logger.info(
            "LogisticGLM fit: %d samples, %d features, "
            "%d non-zero coefficients",
            len(X),
            X.shape[1],
            np.count_nonzero(self.model.coef_),
        )

        return self

    def predict_proba(self, X):
        """Return probability of sepsis (column 1)."""
        if not self._is_fit:
            raise ValueError("Model not fit. Call fit() first.")

        X, _ = self._preprocess_data(X)
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)

        return self.model.predict_proba(X)


class XGBoostModel(BaseModel):
    """
    XGBoost classifier for sepsis risk prediction.

    Handles:
    - Missing value imputation (median)
    - Tree-based feature importance
    - Gradient boosting on imbalanced data
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        """
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            random_state: Random seed
        """
        if not _XGB_AVAILABLE:
            raise RuntimeError(
                "XGBoost not available. Install with: pip install xgboost"
            )

        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            reg_alpha=0.1,
            reg_lambda=0.5,
            max_delta_step=1,       # XGBoost recommendation for imbalanced data
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=1,  # overridden in fit() from actual training labels
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        )
        self.imputer = SimpleImputer(strategy="median")
        self._is_fit = False

    def fit(self, X, y, sample_weight=None):
        """Fit on training data."""
        X, y = self._preprocess_data(X, y)

        # Impute missing values
        X = self.imputer.fit_transform(X)

        # Set scale_pos_weight from actual class ratio in this training split.
        # Using the full ratio keeps models sensitive to rare sepsis events.
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        self.model.set_params(scale_pos_weight=spw)

        # Fit XGBoost with optional sample weights
        self.model.fit(X, y, sample_weight=sample_weight)
        self._is_fit = True

        logger.info(
            "XGBoostModel fit: %d samples, %d features",
            len(X),
            X.shape[1],
        )

        return self

    def predict_proba(self, X):
        """Return probability of sepsis."""
        if not self._is_fit:
            raise ValueError("Model not fit. Call fit() first.")

        X, _ = self._preprocess_data(X)
        X = self.imputer.transform(X)

        return self.model.predict_proba(X)


class GRUModel(BaseModel):
    """
    Simple GRU-based neural network for sepsis risk prediction.

    Architecture:
    - Two-layer GRU (128 hidden units per layer)
    - Dropout regularization
    - Binary cross-entropy loss with class weighting

    Input shape: (batch_size, sequence_length, n_features)
    For patient-hour data, sequence_length=1 (single timestep per row).
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 5e-4,
        random_state: int = 42,
    ):
        """
        Args:
            hidden_size: GRU hidden dimension
            num_layers: Number of stacked GRU layers
            dropout: Dropout rate
            epochs: Training epochs
            batch_size: Batch size for training
            learning_rate: Adam learning rate
            random_state: Random seed
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch required for GRU model. Install with: pip install torch"
            )

        super().__init__(random_state)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.gru = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self._is_fit = False
        self._device = torch.device("cpu")

        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def _build_gru(self, n_features: int):
        """Build GRU architecture returning raw logits (no sigmoid)."""

        class _GRUNet(nn.Module):
            def __init__(self, n_features, hidden_size, num_layers, dropout):
                super().__init__()
                self.gru = nn.GRU(
                    input_size=n_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                # x: (batch, seq_len, features) — returns logits, not probabilities
                gru_out, _ = self.gru(x)
                last_out = gru_out[:, -1, :]
                return self.fc(last_out).squeeze(-1)

        return _GRUNet(n_features, self.hidden_size, self.num_layers, self.dropout).to(
            self._device
        )

    def fit(self, X, y, sample_weight=None):
        """Fit on training data."""
        X, y = self._preprocess_data(X, y)

        # Impute and scale
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)

        # Build network
        self.gru = self._build_gru(X.shape[1])
        optimizer = optim.Adam(self.gru.parameters(), lr=self.learning_rate)

        # Weighted loss using full class ratio to keep the GRU sensitive to
        # rare sepsis events. The PhysioNet utility function penalises missed
        # sepsis (-2) much more than false alarms (-0.05), so liberal alarming
        # is correct behaviour.
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        pw = n_neg / n_pos if n_pos > 0 else 1.0
        pos_weight = torch.tensor([pw], dtype=torch.float32, device=self._device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

        n_samples = len(X)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        for epoch in range(self.epochs):
            self.gru.train()
            epoch_loss = 0

            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = min((batch_idx + 1) * self.batch_size, n_samples)

                batch_indices = indices[start:end]

                X_batch = torch.tensor(
                    X[batch_indices],
                    dtype=torch.float32,
                    device=self._device
                ).unsqueeze(1)

                y_batch = torch.tensor(
                    y[batch_indices],
                    dtype=torch.float32,
                    device=self._device
                )

                optimizer.zero_grad()
                y_pred = self.gru(X_batch)
                loss_unreduced = criterion(y_pred, y_batch)

                if sample_weight is not None:
                    batch_weights = torch.tensor(
                        sample_weight[batch_indices],
                        dtype=torch.float32,
                        device=self._device
                    )
                    loss = (loss_unreduced * batch_weights).mean()
                else:
                    loss = loss_unreduced.mean()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        self._is_fit = True

        logger.info(
            "GRUModel fit: %d samples, %d features, %d epochs",
            len(X),
            X.shape[1],
            self.epochs,
        )

        return self

    def predict_proba(self, X):
        """Return probability of sepsis (column 1)."""
        if not self._is_fit:
            raise ValueError("Model not fit. Call fit() first.")

        X, _ = self._preprocess_data(X)
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)

        self.gru.eval()
        probs = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                end = min(start + self.batch_size, len(X))
                X_batch = torch.tensor(
                    X[start:end],
                    dtype=torch.float32,
                    device=self._device
                ).unsqueeze(1)

                # Network returns logits; apply sigmoid to get probabilities
                logits = self.gru(X_batch)
                proba_batch = torch.sigmoid(logits).cpu().numpy()
                probs.append(proba_batch)

        proba_pos = np.concatenate(probs)
        proba_neg = 1 - proba_pos
        return np.column_stack([proba_neg, proba_pos])


def get_model(model_name: str, **kwargs) -> BaseModel:
    """
    Factory function to get a model by name.

    Args:
        model_name: "glm", "xgboost", or "gru"
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Initialized model (not fit)
    """
    if model_name == "glm":
        return LogisticGLM(**kwargs)
    elif model_name == "xgboost":
        return XGBoostModel(**kwargs)
    elif model_name == "gru":
        return GRUModel(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")