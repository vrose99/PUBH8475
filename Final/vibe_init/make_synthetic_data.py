"""
Generate a small synthetic dataset that mimics the PhysioNet 2019 format.

Useful for smoke-testing the full pipeline without downloading the real data.

Usage:
  python make_synthetic_data.py                    # 500 patients
  python make_synthetic_data.py --n-patients 2000  # larger dataset
  python make_synthetic_data.py --out data/synthetic_sepsis

Creates one PSV file per patient under the output directory.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader_timeseries import DEMO_COLS, LAB_COLS, VITAL_COLS
LABEL_COL = "SepsisLabel"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Approximate ranges (mean, std) from PhysioNet challenge overview
VITAL_PARAMS = {
    "HR":    (83, 18),   "O2Sat": (97, 3),   "Temp": (37.0, 0.8),
    "SBP":   (122, 24),  "MAP":   (82, 16),  "DBP":  (64, 14),
    "Resp":  (18, 5),    "EtCO2": (32, 8),
}
LAB_PARAMS = {c: (1.0, 0.5) for c in LAB_COLS}  # placeholder ranges

SEPSIS_RATE   = 0.10   # 10 % positive outcome
FEMALE_FRAC   = 0.45   # 45 % female


def _make_patient(patient_id: int, rng: np.random.Generator) -> pd.DataFrame:
    n_hours = int(rng.integers(4, 72))
    is_female = int(rng.random() < FEMALE_FRAC)
    is_sepsis  = int(rng.random() < SEPSIS_RATE)
    age        = rng.normal(65, 15)
    age        = float(np.clip(age, 18, 95))

    rows = []
    for t in range(n_hours):
        row: dict = {}
        for col, (mu, sigma) in VITAL_PARAMS.items():
            row[col] = float(np.clip(rng.normal(mu, sigma), 0, None)) if rng.random() > 0.1 else np.nan
        for col, (mu, sigma) in LAB_PARAMS.items():
            row[col] = float(np.clip(rng.normal(mu, sigma), 0, None)) if rng.random() > 0.5 else np.nan
        row["Age"]         = age
        row["Gender"]      = is_female
        row["Unit1"]       = int(rng.random() > 0.5)
        row["Unit2"]       = int(rng.random() > 0.5)
        row["HospAdmTime"] = -float(rng.integers(0, 48))
        row["ICULOS"]      = t + 1
        row["SepsisLabel"] = 1 if (is_sepsis and t >= n_hours - 2) else 0
        rows.append(row)

    col_order = VITAL_COLS + LAB_COLS + DEMO_COLS + [LABEL_COL]
    return pd.DataFrame(rows)[col_order]


def make_synthetic_dataset(
    n_patients: int = 500,
    out_dir: Path = Path("data/synthetic_sepsis"),
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_patients):
        patient_df = _make_patient(i, rng)
        path = out_dir / f"p{i:06d}.psv"
        patient_df.to_csv(path, sep="|", index=False)

    logger.info("Created %d synthetic patient PSVs in %s", n_patients, out_dir)
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-patients", type=int, default=500)
    p.add_argument("--out", type=Path, default=Path("data/synthetic_sepsis"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    make_synthetic_dataset(args.n_patients, args.out, args.seed)
