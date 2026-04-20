from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, roc_curve

from config import TARGET_COL
from data import DataModule


@dataclass
class RowLevelMetrics:
    auc: float
    auprc: float
    accuracy_at_optimal_threshold: float
    optimal_threshold: float
    sensitivity_at_optimal_threshold: float
    specificity_at_optimal_threshold: float


class EvaluationModule:
    @staticmethod
    def optimal_threshold_from_roc(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        dist = np.sqrt((1.0 - tpr) ** 2 + fpr**2)
        idx = int(np.argmin(dist))
        return {
            "threshold": float(thr[idx]),
            "tpr": float(tpr[idx]),
            "fpr": float(fpr[idx]),
        }

    @classmethod
    def evaluate_binary_probabilities(cls, y_true: np.ndarray, y_prob: np.ndarray) -> RowLevelMetrics:
        auc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        thr = cls.optimal_threshold_from_roc(y_true, y_prob)
        y_pred = (y_prob >= thr["threshold"]).astype(int)
        acc = accuracy_score(y_true, y_pred)
        return RowLevelMetrics(
            auc=float(auc),
            auprc=float(auprc),
            accuracy_at_optimal_threshold=float(acc),
            optimal_threshold=float(thr["threshold"]),
            sensitivity_at_optimal_threshold=float(thr["tpr"]),
            specificity_at_optimal_threshold=float(1.0 - thr["fpr"]),
        )

    @staticmethod
    def build_full_patient_time_scores_static(model, df_feat: pd.DataFrame, patient_ids, feature_cols) -> pd.DataFrame:
        sub = df_feat[df_feat["patient_id"].isin(patient_ids)].copy()
        sub = sub.sort_values(["patient_id", "t"]).reset_index(drop=True)
        sub["risk_score"] = model.predict_proba(sub[feature_cols])
        return sub

    @staticmethod
    def build_full_patient_time_scores_sequence(model, df_feat, patient_ids, feature_cols, seq_len: int) -> pd.DataFrame:
        rows = []
        sub = df_feat[df_feat["patient_id"].isin(patient_ids)].copy()
        sub = sub.sort_values(["patient_id", "t"]).reset_index(drop=True)
        for pid, g in sub.groupby("patient_id", sort=False):
            g = g.sort_values("t").reset_index(drop=True)
            if len(g) < seq_len:
                continue
            for end_idx in range(seq_len - 1, len(g)):
                block = g.iloc[end_idx - seq_len + 1 : end_idx + 1]
                rows.append(
                    {
                        "patient_id": pid,
                        "t": int(g.loc[end_idx, "t"]),
                        TARGET_COL: int(g.loc[end_idx, TARGET_COL]),
                        "seq": block[list(feature_cols)].to_numpy(),
                    }
                )

        if not rows:
            return pd.DataFrame(columns=["patient_id", "t", TARGET_COL, "risk_score"])

        eval_df = pd.DataFrame(rows)
        X = np.stack(eval_df["seq"].values)
        eval_df["risk_score"] = model.predict_proba(X)
        return eval_df[["patient_id", "t", TARGET_COL, "risk_score"]]

    @staticmethod
    def compute_early_warning_metrics(
        score_df: pd.DataFrame,
        threshold: float,
        label_lead_hours: int,
    ) -> Dict[str, float]:
        ewt_list = []
        n_positive_patients = 0
        n_detected_positive_patients = 0
        n_negative_patients = 0
        false_alert_patients = 0

        for _, g in score_df.groupby("patient_id", sort=False):
            g = g.sort_values("t").reset_index(drop=True)
            pos = np.where(g[TARGET_COL].fillna(0).to_numpy(dtype=int) == 1)[0]
            detected_times = g.loc[g["risk_score"] >= threshold, "t"].tolist()

            if len(pos) > 0:
                n_positive_patients += 1
                label_start_t = int(g.loc[pos[0], "t"])
                proxy_event_t = label_start_t + label_lead_hours
                valid_detections = [t for t in detected_times if t < proxy_event_t]
                if valid_detections:
                    det_t = int(valid_detections[0])
                    ewt_list.append(proxy_event_t - det_t)
                    n_detected_positive_patients += 1
            else:
                n_negative_patients += 1
                if detected_times:
                    false_alert_patients += 1

        return {
            "n_positive_patients": int(n_positive_patients),
            "n_detected_positive_patients": int(n_detected_positive_patients),
            "patient_sensitivity": float(n_detected_positive_patients / n_positive_patients) if n_positive_patients else float("nan"),
            "n_negative_patients": int(n_negative_patients),
            "false_alert_patient_rate": float(false_alert_patients / n_negative_patients) if n_negative_patients else float("nan"),
            "median_early_warning_time_hours": float(np.median(ewt_list)) if ewt_list else float("nan"),
            "mean_early_warning_time_hours": float(np.mean(ewt_list)) if ewt_list else float("nan"),
        }
