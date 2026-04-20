from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from models.base import BaseSepsisModel

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    Dataset = object


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class GRUClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :]
        return self.fc(h).squeeze(-1)


class LiuLikeGRU(BaseSepsisModel):
    """GRU-based recurrent model.

    Liu et al. report an RNN as the best-performing family and note that additional RNN training
    details are in the supplement. Because those exact architecture details are not in the main
    paper, this wrapper uses a single-layer GRU with median imputation, z-scoring, class-weighted
    BCE loss, and validation-AUC-based checkpointing.
    """

    name = "gru"
    is_sequence_model = True

    def __init__(
        self,
        random_state: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        epochs: int = 12,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        validation_size: float = 0.15,
    ):
        self.random_state = random_state
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_size = validation_size

        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[GRUClassifier] = None
        self.input_size: Optional[int] = None

    @staticmethod
    def _check_torch() -> None:
        if torch is None:
            raise ImportError("torch is not installed.")

    def _impute_and_scale(self, X_train: np.ndarray, X_other: np.ndarray | None = None):
        n_train, seq_len, n_feat = X_train.shape
        tr2 = X_train.reshape(-1, n_feat)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        tr2 = self.imputer.fit_transform(tr2)
        tr2 = self.scaler.fit_transform(tr2)
        X_train_scaled = tr2.reshape(n_train, seq_len, n_feat)

        if X_other is None:
            return X_train_scaled, None

        n_other = X_other.shape[0]
        other2 = X_other.reshape(-1, n_feat)
        other2 = self.imputer.transform(other2)
        other2 = self.scaler.transform(other2)
        X_other_scaled = other2.reshape(n_other, seq_len, n_feat)
        return X_train_scaled, X_other_scaled

    def fit(self, X, y, **kwargs) -> "LiuLikeGRU":
        self._check_torch()
        idx = np.arange(len(X))
        tr_idx, val_idx = train_test_split(
            idx,
            test_size=self.validation_size,
            random_state=self.random_state,
            stratify=y if len(set(y)) > 1 else None,
        )
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        X_tr, X_val = self._impute_and_scale(X_tr, X_val)
        self.input_size = X_tr.shape[-1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GRUClassifier(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(device)

        train_dl = DataLoader(SequenceDataset(X_tr, y_tr), batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(SequenceDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False)

        cls_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_tr.astype(int))
        pos_weight = torch.tensor([cls_weights[1] / cls_weights[0]], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_state = None
        best_auc = -np.inf

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in train_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            self.model.eval()
            val_probs = []
            val_true = []
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb = xb.to(device)
                    logits = self.model(xb)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_probs.extend(probs.tolist())
                    val_true.extend(yb.numpy().tolist())
            val_auc = roc_auc_score(val_true, val_probs) if len(set(val_true)) > 1 else np.nan
            if np.isfinite(val_auc) and val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def _transform_sequences(self, X: np.ndarray) -> np.ndarray:
        if self.imputer is None or self.scaler is None:
            raise RuntimeError("Preprocessors are not fit.")
        n, seq_len, n_feat = X.shape
        X2 = X.reshape(-1, n_feat)
        X2 = self.imputer.transform(X2)
        X2 = self.scaler.transform(X2)
        return X2.reshape(n, seq_len, n_feat)

    def predict_proba(self, X) -> np.ndarray:
        self._check_torch()
        if self.model is None:
            raise RuntimeError("Model has not been fit.")
        X = self._transform_sequences(X)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device)
        loader = DataLoader(SequenceDataset(X, np.zeros(len(X))), batch_size=256, shuffle=False)
        probs = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                logits = self.model(xb)
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        return np.asarray(probs)

    def save(self, path: Path) -> None:
        self._check_torch()
        if self.model is None:
            raise RuntimeError("Model has not been fit.")
        payload = {
            "random_state": self.random_state,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "validation_size": self.validation_size,
            "input_size": self.input_size,
            "state_dict": self.model.state_dict(),
            "imputer": self.imputer,
            "scaler": self.scaler,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, **kwargs) -> "LiuLikeGRU":
        if torch is None:
            raise ImportError("torch is not installed.")
        payload = torch.load(path, map_location="cpu")
        model = cls(
            random_state=payload["random_state"],
            hidden_size=payload["hidden_size"],
            num_layers=payload["num_layers"],
            dropout=payload["dropout"],
            epochs=payload["epochs"],
            batch_size=payload["batch_size"],
            learning_rate=payload["learning_rate"],
            validation_size=payload["validation_size"],
        )
        model.input_size = payload["input_size"]
        model.imputer = payload["imputer"]
        model.scaler = payload["scaler"]
        model.model = GRUClassifier(
            input_size=model.input_size,
            hidden_size=model.hidden_size,
            num_layers=model.num_layers,
            dropout=model.dropout,
        )
        model.model.load_state_dict(payload["state_dict"])
        return model
