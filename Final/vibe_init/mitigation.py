"""
Bias mitigation wrappers.

Each strategy is a function with signature:
    apply_<strategy>(X_train, y_train, sensitive_train, model, cfg)
        -> (X_train_out, y_train_out, sensitive_out, sample_weight, fitted_model_or_None)

The evaluation loop calls the appropriate function and then either:
  - uses the returned (X, y, w) to fit the base model, OR
  - uses the returned fitted_model directly (for fairness_penalty which wraps
    the model internally via fairlearn's ExponentiatedGradient).

Strategy "none" is the unmodified baseline.

Strategies
----------
none              — no modification
reweighting       — inverse-frequency sample weights per (group, label) cell
smote             — SMOTE applied to balance the minority sensitive group
fairness_penalty  — fairlearn ExponentiatedGradient with EqualizedOdds
robust_model      — swap in a calibrated LR with balanced class weights
"""

import logging
from typing import Optional

import numpy as np

from config import Config

logger = logging.getLogger(__name__)

MITIGATION_REGISTRY: dict[str, callable] = {}


def register(name: str):
    def decorator(fn):
        MITIGATION_REGISTRY[name] = fn
        return fn
    return decorator


# ── Helper: inverse-frequency weights ────────────────────────────────────────

def _compute_reweighting(
    y: np.ndarray,
    sensitive: np.ndarray,
    female_val, male_val,
) -> np.ndarray:
    """
    Assign weight = 1 / P(group) / P(label|group) to each sample.
    Normalised so weights sum to n_samples.
    """
    weights = np.ones(len(y), dtype=float)
    for g in [female_val, male_val]:
        for lbl in [0, 1]:
            mask = (sensitive == g) & (y == lbl)
            if mask.sum() > 0:
                weights[mask] = len(y) / (4 * mask.sum())
    return weights


# ── Strategies ────────────────────────────────────────────────────────────────

@register("none")
def apply_none(X, y, sensitive, model, cfg: Config):
    """Baseline: no modification."""
    return X, y, sensitive, None, None


@register("reweighting")
def apply_reweighting(X, y, sensitive, model, cfg: Config):
    """
    Reweight samples so each (group × label) cell contributes equally.

    This corrects for both class imbalance and group imbalance without
    altering the training data itself.
    """
    weights = _compute_reweighting(
        y, sensitive,
        cfg.fairness.female_value,
        cfg.fairness.male_value,
    )
    logger.debug(
        "Reweighting: weight range [%.3f, %.3f]", weights.min(), weights.max()
    )
    return X, y, sensitive, weights, None


@register("smote")
def apply_smote(X, y, sensitive, model, cfg: Config):
    """
    Oversample the underrepresented sensitive group to match the majority
    group's size.

    Strategy (in descending preference):
      1. SMOTE — generates interpolated synthetic samples (requires ≥ 6
         samples in each class within the group).
      2. RandomOverSampler — duplicates existing rows; always valid as a
         fallback.

    Both preserve the existing class (outcome) ratio within the group.

    Requires: imbalanced-learn (`pip install imbalanced-learn`)
    """
    try:
        from imblearn.over_sampling import SMOTE, RandomOverSampler
    except ImportError:
        raise ImportError(
            "imbalanced-learn not installed — `pip install imbalanced-learn`"
        )

    rng = np.random.default_rng(cfg.random_state)

    f_mask = sensitive == cfg.fairness.female_value
    m_mask = sensitive == cfg.fairness.male_value
    n_female, n_male = f_mask.sum(), m_mask.sum()
    target = max(n_female, n_male)

    parts_X, parts_y, parts_s = [X], [y], [sensitive]

    for g_mask, g_val, n_g in [
        (f_mask, cfg.fairness.female_value, n_female),
        (m_mask, cfg.fairness.male_value,   n_male),
    ]:
        if n_g >= target:
            continue   # already the majority group

        X_g, y_g = X[g_mask], y[g_mask]
        n_synth = target - n_g

        # Build per-class target counts: current + proportional share of n_synth.
        # All values must be >= current count (required by imbalanced-learn).
        class_counts = {int(lbl): int((y_g == lbl).sum()) for lbl in np.unique(y_g)}
        total = sum(class_counts.values())
        new_counts = {
            lbl: cnt + max(1, int(n_synth * cnt / total))
            for lbl, cnt in class_counts.items()
        }

        min_class = min(class_counts.values())
        k_neighbors = min(5, min_class - 1)

        if k_neighbors >= 1:
            try:
                oversampler = SMOTE(
                    k_neighbors=k_neighbors,
                    sampling_strategy=new_counts,
                    random_state=cfg.random_state,
                )
                X_res, y_res = oversampler.fit_resample(X_g, y_g)
                method = "SMOTE"
            except Exception as exc:
                logger.debug("SMOTE fell back to RandomOverSampler: %s", exc)
                oversampler = RandomOverSampler(
                    sampling_strategy=new_counts,
                    random_state=cfg.random_state,
                )
                X_res, y_res = oversampler.fit_resample(X_g, y_g)
                method = "RandomOverSampler"
        else:
            oversampler = RandomOverSampler(
                sampling_strategy=new_counts,
                random_state=cfg.random_state,
            )
            X_res, y_res = oversampler.fit_resample(X_g, y_g)
            method = "RandomOverSampler"

        n_new = len(X_res) - n_g
        if n_new > 0:
            parts_X.append(X_res[n_g:])
            parts_y.append(y_res[n_g:])
            parts_s.append(np.full(n_new, g_val))
            logger.debug(
                "%s: generated %d synthetic samples for group=%s", method, n_new, g_val
            )

    X_out = np.vstack(parts_X)
    y_out = np.concatenate(parts_y)
    s_out = np.concatenate(parts_s)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle], s_out[shuffle], None, None


@register("fairness_penalty")
def apply_fairness_penalty(X, y, sensitive, model, cfg: Config):
    """
    Use fairlearn's ExponentiatedGradient with EqualizedOdds constraint.

    This wraps the base model in a constrained optimisation loop that
    explicitly penalises disparities in TPR and FPR across groups.
    Returns a fitted fairlearn estimator directly; the evaluation loop
    detects this and skips the standard fit() call.

    Requires: fairlearn (`pip install fairlearn`)
    """
    try:
        from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
    except ImportError:
        raise ImportError(
            "fairlearn not installed — `pip install fairlearn`"
        )

    from sklearn.base import clone

    # Clone the passed model so ExponentiatedGradient gets a fresh, unfitted estimator.
    # clone() copies hyperparameters without copying fitted state.
    base = clone(model)
    mitigator = ExponentiatedGradient(
        estimator=base,
        constraints=EqualizedOdds(),
        eps=0.01,
    )
    mitigator.fit(X, y, sensitive_features=sensitive)
    logger.debug("fairness_penalty: ExponentiatedGradient fitted.")
    return X, y, sensitive, None, mitigator   # return fitted model


@register("robust_model")
def apply_robust_model(X, y, sensitive, model, cfg: Config):
    """
    Replace the base model with a calibrated logistic regression that has
    balanced class weights — a conservative, interpretable choice that often
    has better out-of-distribution behaviour than complex trees.

    Useful as a "does a simpler model fix the problem?" baseline.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    robust = CalibratedClassifierCV(
        LogisticRegression(
            max_iter=2000,
            solver="saga",
            class_weight="balanced",
            random_state=cfg.random_state,
        ),
        method="isotonic",
        cv=3,
    )
    return X, y, sensitive, None, None   # swap happens in evaluation.py


def get_mitigation(name: str) -> callable:
    if name not in MITIGATION_REGISTRY:
        raise ValueError(
            f"Unknown mitigation '{name}'. Available: {list(MITIGATION_REGISTRY)}"
        )
    return MITIGATION_REGISTRY[name]
