# Liu-like modular sepsis modeling package

Files:
- `config.py`: global configuration and feature list.
- `data.py`: patient file loading, feature engineering, patient-level split, Liu-like proxy label construction, and sequence generation.
- `evaluation.py`: row-level ROC/AUPRC evaluation, threshold selection, and patient-level early-warning summaries.
- `models/base.py`: shared model interface.
- `models/glm_model.py`: L1-regularized GLM wrapper.
- `models/xgb_model.py`: XGBoost wrapper.
- `models/gru_model.py`: recurrent sequence wrapper using a GRU.
- `run_experiment.py`: end-to-end training and evaluation script.

## Important methodological note
This code is **Liu-inspired** and **Challenge-compatible**, but it is not a literal reproduction of the paper. The main paper trains on:
1. sepsis patients who never progress to septic shock, and
2. a 1-hour window 2 to 1 hours before **septic shock onset** for patients who do progress.

The PhysioNet Challenge files expose `SepsisLabel`, which turns positive 6 hours before sepsis onset, not septic shock onset. In this package, the event time is therefore proxied as:

`proxy_event_time = first_positive_label_time + label_lead_hours`

That makes the training target compatible with your notebook and with the Challenge data, while staying as close as possible to the Liu framing.
