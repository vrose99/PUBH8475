from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import ExperimentConfig
from data import DataModule
from evaluation import EvaluationModule
from models.glm_model import LiuLikeGLM
from models.xgb_model import LiuLikeXGBoost
from models.gru_model import LiuLikeGRU


def main() -> None:
    cfg = ExperimentConfig()
    cfg.ensure_output_dir()

    dm = DataModule(cfg)
    dm.set_global_seed(cfg.seed)
    ev = EvaluationModule()

    df, summaries = dm.load_all_patients()
    train_ids, test_ids = dm.split_patients(summaries, test_size=cfg.test_size, seed=cfg.seed)
    df_feat = dm.add_engineered_features(df)

    static_ds = dm.build_static_dataset(df_feat, train_ids, test_ids)

    results = {}

    glm = LiuLikeGLM(random_state=cfg.seed).fit(static_ds.X_train, static_ds.y_train)
    glm_metrics = ev.evaluate_binary_probabilities(static_ds.y_test, glm.predict_proba(static_ds.X_test))
    glm.save(cfg.output_dir / "glm_model.joblib")
    glm_scores = ev.build_full_patient_time_scores_static(glm, df_feat, test_ids, static_ds.feature_cols)
    glm_ewt = ev.compute_early_warning_metrics(glm_scores, glm_metrics.optimal_threshold, cfg.label_lead_hours)
    glm_scores.to_csv(cfg.output_dir / "glm_test_time_scores.csv", index=False)
    results["glm_row"] = glm_metrics.__dict__
    results["glm_patient_time"] = glm_ewt

    try:
        xgb = LiuLikeXGBoost(random_state=cfg.seed).fit(static_ds.X_train, static_ds.y_train)
        xgb_metrics = ev.evaluate_binary_probabilities(static_ds.y_test, xgb.predict_proba(static_ds.X_test))
        xgb.save(cfg.output_dir / "xgboost_model.joblib")
        xgb_scores = ev.build_full_patient_time_scores_static(xgb, df_feat, test_ids, static_ds.feature_cols)
        xgb_ewt = ev.compute_early_warning_metrics(xgb_scores, xgb_metrics.optimal_threshold, cfg.label_lead_hours)
        xgb_scores.to_csv(cfg.output_dir / "xgb_test_time_scores.csv", index=False)
        results["xgb_row"] = xgb_metrics.__dict__
        results["xgb_patient_time"] = xgb_ewt
    except ImportError:
        pass

    try:
        train_seq_X, train_seq_y = dm.build_patient_sequences(df_feat, train_ids, static_ds.feature_cols)
        test_seq_X, test_seq_y = dm.build_patient_sequences(df_feat, test_ids, static_ds.feature_cols)
        gru = LiuLikeGRU(
            random_state=cfg.seed,
            hidden_size=cfg.rnn_hidden_size,
            num_layers=cfg.rnn_num_layers,
            dropout=cfg.rnn_dropout,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
        ).fit(train_seq_X, train_seq_y)
        gru_metrics = ev.evaluate_binary_probabilities(test_seq_y, gru.predict_proba(test_seq_X))
        gru.save(cfg.output_dir / "gru_model.pt")
        gru_scores = ev.build_full_patient_time_scores_sequence(gru, df_feat, test_ids, static_ds.feature_cols, cfg.seq_len)
        gru_ewt = ev.compute_early_warning_metrics(gru_scores, gru_metrics.optimal_threshold, cfg.label_lead_hours)
        gru_scores.to_csv(cfg.output_dir / "gru_test_time_scores.csv", index=False)
        results["gru_row"] = gru_metrics.__dict__
        results["gru_patient_time"] = gru_ewt
    except ImportError:
        pass

    with open(cfg.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(cfg.output_dir / "patient_split.json", "w", encoding="utf-8") as f:
        json.dump({"train_patient_ids": train_ids, "test_patient_ids": test_ids}, f, indent=2)

    summary_rows = []
    for model_name in ("glm", "xgb", "gru"):
        row_key = f"{model_name}_row"
        time_key = f"{model_name}_patient_time"
        if row_key in results and time_key in results:
            summary_rows.append(
                {
                    "model": model_name,
                    "auc": results[row_key]["auc"],
                    "auprc": results[row_key]["auprc"],
                    "sensitivity": results[row_key]["sensitivity_at_optimal_threshold"],
                    "specificity": results[row_key]["specificity_at_optimal_threshold"],
                    "median_ewt_hours": results[time_key]["median_early_warning_time_hours"],
                }
            )
    print(pd.DataFrame(summary_rows))


if __name__ == "__main__":
    main()
