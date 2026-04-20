from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import ALL_FEATURES, TARGET_COL, ExperimentConfig


@dataclass
class PatientSummary:
    patient_id: str
    n_rows: int
    label_start_idx: Optional[int]
    became_positive: bool


@dataclass
class StaticDataset:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    feature_cols: List[str]
    train_rows: pd.DataFrame
    test_rows: pd.DataFrame


@dataclass
class SequenceDatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_cols: List[str]
    imputer: object | None = None
    scaler: object | None = None


class DataModule:
    """Data preparation for PhysioNet 2019 style hourly patient files.

    This module is intentionally Liu-like rather than a literal reproduction of Liu et al.
    For Challenge data, the available outcome is `SepsisLabel`, which becomes positive 6 hours
    before the sepsis onset time. We therefore expose a proxy event time equal to the first
    positive label time plus `label_lead_hours`.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

    @staticmethod
    def set_global_seed(seed: int = 42) -> None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    def find_patient_files(self) -> List[Path]:
        data_dir = self.config.data_dir
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

        if data_dir.is_file():
            if data_dir.suffix.lower() in {".psv", ".csv"}:
                return [data_dir]
            raise ValueError(f"Unsupported data file: {data_dir}")

        pattern_func = data_dir.rglob if self.config.recursive else data_dir.glob
        files = list(pattern_func("*.psv")) + list(pattern_func("*.csv"))
        files = sorted(f for f in files if f.is_file() and f.stem.startswith("p"))
        if not files:
            raise FileNotFoundError(f"No patient .psv or .csv files found under {data_dir}")
        return files

    @staticmethod
    def _load_single_patient(fp: Path) -> pd.DataFrame:
        if fp.suffix.lower() == ".psv":
            df = pd.read_csv(fp, sep="|")
        elif fp.suffix.lower() == ".csv":
            df = pd.read_csv(fp)
        else:
            raise ValueError(f"Unsupported file type: {fp}")

        if TARGET_COL not in df.columns:
            raise ValueError(f"Missing required target column {TARGET_COL!r} in {fp}")

        for col in ALL_FEATURES:
            if col not in df.columns:
                df[col] = np.nan

        df = df[ALL_FEATURES + [TARGET_COL]].copy()
        df["patient_id"] = fp.stem
        df["t"] = np.arange(len(df), dtype=int)
        return df

    def load_all_patients(self) -> Tuple[pd.DataFrame, List[PatientSummary]]:
        frames: List[pd.DataFrame] = []
        summaries: List[PatientSummary] = []
        for fp in self.find_patient_files():
            df = self._load_single_patient(fp)
            if len(df) < self.config.min_patient_hours:
                continue

            pos = np.where(df[TARGET_COL].fillna(0).to_numpy(dtype=int) == 1)[0]
            label_start_idx = int(pos[0]) if len(pos) else None
            summaries.append(
                PatientSummary(
                    patient_id=fp.stem,
                    n_rows=len(df),
                    label_start_idx=label_start_idx,
                    became_positive=label_start_idx is not None,
                )
            )
            frames.append(df)

        if not frames:
            raise RuntimeError("No patients remained after filtering.")
        return pd.concat(frames, ignore_index=True), summaries

    @staticmethod
    def split_patients(
        summaries: Sequence[PatientSummary],
        test_size: float,
        seed: int,
    ) -> Tuple[List[str], List[str]]:
        patient_ids = [s.patient_id for s in summaries]
        labels = [int(s.became_positive) for s in summaries]
        train_ids, test_ids = train_test_split(
            patient_ids,
            test_size=test_size,
            random_state=seed,
            stratify=labels if len(set(labels)) > 1 else None,
        )
        return sorted(train_ids), sorted(test_ids)

    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().sort_values(["patient_id", "t"]).reset_index(drop=True)

        if self.config.create_missingness_flags:
            for col in ALL_FEATURES:
                out[f"{col}_missing"] = out[col].isna().astype(int)

        if self.config.create_delta_features:
            groups: List[pd.DataFrame] = []
            for _, g in out.groupby("patient_id", sort=False):
                ff = g[ALL_FEATURES].ffill()
                deltas = ff.diff().fillna(0.0)
                deltas.columns = [f"{c}_delta" for c in ALL_FEATURES]
                groups.append(pd.concat([g, deltas], axis=1))
            out = pd.concat(groups, ignore_index=True)

        return out

    @staticmethod
    def choose_feature_columns(df: pd.DataFrame) -> List[str]:
        base = [c for c in ALL_FEATURES if c in df.columns]
        missing = sorted(c for c in df.columns if c.endswith("_missing"))
        delta = sorted(c for c in df.columns if c.endswith("_delta"))
        return base + missing + delta

    @staticmethod
    def get_label_start_and_proxy_event_time(
        g: pd.DataFrame,
        label_lead_hours: int,
        target_col: str = TARGET_COL,
    ) -> Tuple[Optional[int], Optional[int]]:
        g = g.sort_values("t").reset_index(drop=True)
        pos = np.where(g[target_col].fillna(0).to_numpy(dtype=int) == 1)[0]
        if len(pos) == 0:
            return None, None
        label_start_t = int(g.loc[pos[0], "t"])
        proxy_event_t = label_start_t + label_lead_hours
        return label_start_t, proxy_event_t

    def build_proxy_training_rows(
        self,
        df: pd.DataFrame,
        patient_ids: Sequence[str],
    ) -> pd.DataFrame:
        lo, hi = self.config.positive_window
        out_frames: List[pd.DataFrame] = []
        sub = df[df["patient_id"].isin(patient_ids)].copy()

        for _, g in sub.groupby("patient_id", sort=False):
            g = g.sort_values("t").reset_index(drop=True)
            _, proxy_event_t = self.get_label_start_and_proxy_event_time(
                g,
                label_lead_hours=self.config.label_lead_hours,
            )

            if proxy_event_t is None:
                gg = g.copy()
                gg["proxy_label"] = 0
                out_frames.append(gg)
                continue

            start = proxy_event_t + lo
            end = proxy_event_t + hi
            mask = (g["t"] >= start) & (g["t"] <= end)
            gg = g.loc[mask].copy()
            if gg.empty:
                continue
            gg["proxy_label"] = 1
            out_frames.append(gg)

        if not out_frames:
            raise RuntimeError("No proxy rows were generated.")
        return pd.concat(out_frames, ignore_index=True)

    def build_static_dataset(
        self,
        df_feat: pd.DataFrame,
        train_ids: Sequence[str],
        test_ids: Sequence[str],
    ) -> StaticDataset:
        feature_cols = self.choose_feature_columns(df_feat)
        train_rows = self.build_proxy_training_rows(df_feat, train_ids)
        test_rows = self.build_proxy_training_rows(df_feat, test_ids)
        return StaticDataset(
            X_train=train_rows[feature_cols],
            y_train=train_rows["proxy_label"].to_numpy(dtype=int),
            X_test=test_rows[feature_cols],
            y_test=test_rows["proxy_label"].to_numpy(dtype=int),
            feature_cols=feature_cols,
            train_rows=train_rows,
            test_rows=test_rows,
        )

    def build_patient_sequences(
        self,
        df: pd.DataFrame,
        patient_ids: Sequence[str],
        feature_cols: Sequence[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        sub = df[df["patient_id"].isin(patient_ids)].copy()
        sequences: List[np.ndarray] = []
        labels: List[int] = []
        seq_len = self.config.seq_len
        lo, hi = self.config.positive_window

        for _, g in sub.groupby("patient_id", sort=False):
            g = g.sort_values("t").reset_index(drop=True)
            _, proxy_event_t = self.get_label_start_and_proxy_event_time(
                g,
                label_lead_hours=self.config.label_lead_hours,
            )

            if proxy_event_t is None:
                for end_idx in range(seq_len - 1, len(g)):
                    block = g.iloc[end_idx - seq_len + 1 : end_idx + 1]
                    sequences.append(block[list(feature_cols)].to_numpy())
                    labels.append(0)
                continue

            start = proxy_event_t + lo
            end = proxy_event_t + hi
            for end_idx in range(seq_len - 1, len(g)):
                current_t = int(g.loc[end_idx, "t"])
                if start <= current_t <= end:
                    block = g.iloc[end_idx - seq_len + 1 : end_idx + 1]
                    sequences.append(block[list(feature_cols)].to_numpy())
                    labels.append(1)

        if not sequences:
            raise RuntimeError("No sequences were generated.")
        return np.stack(sequences), np.asarray(labels, dtype=int)
