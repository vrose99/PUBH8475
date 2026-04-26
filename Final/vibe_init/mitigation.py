from __future__ import annotations

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
    try:
        from fairlearn.reductions import GridSearch, EqualizedOdds
    except ImportError:
        raise ImportError("fairlearn not installed — `pip install fairlearn`")

    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

    if isinstance(model, LogisticRegressionCV):
        base = LogisticRegression(
            max_iter=model.max_iter,
            solver=model.solver if model.solver != "warn" else "lbfgs",
            random_state=getattr(model, "random_state", None),
            class_weight=getattr(model, "class_weight", None),
        )
    else:
        base = clone(model)

    mitigator = GridSearch(
        estimator=base,
        constraints=EqualizedOdds(),
        grid_size=10,
    )
    mitigator.fit(X, y, sensitive_features=sensitive.astype(int))
    logger.debug("fairness_penalty: GridSearch fitted.")
    return X, y, sensitive, None, mitigator


# ── Per-group threshold optimisation ─────────────────────────────────────────

class _PerGroupThresholdWrapper:
    """
    Wraps a fitted classifier with per-group decision thresholds.

    The wrapper stores `per_group_thresholds_` so that evaluate_ts_single
    can apply logit rescaling before calling compute_detection_fairness_report.
    The raw probabilities from predict_proba() are unchanged; rescaling happens
    in the evaluation layer so the rest of the pipeline sees consistent scores.
    """

    def __init__(self, base_model, per_group_thresholds: dict):
        self._model = base_model
        self.per_group_thresholds_ = per_group_thresholds  # {group_val: threshold}

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y, **kwargs)
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def predict(self, X):
        return self._model.predict(X)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


@register("threshold_optimization")
def apply_threshold_optimization(X, y, sensitive, model, cfg: Config):
    """
    Post-hoc per-group threshold optimisation grounded in PhysioNet utility weights.

    Objective
    ---------
    Minimise a utility-weighted fairness gap across a (t_female, t_male) grid:

        score = |w_fn| * |TPR_gap| + |w_fp| * |FPR_gap|
              + utility_penalty

    where w_fn = -2.0 and w_fp = -0.05 come directly from the PhysioNet 2019
    challenge utility structure (Config.fairness).  The 40× asymmetry reflects
    the clinical reality that missing sepsis (FN) is far worse than a false
    alarm (FP), so the search prioritises closing the TPR gap between groups.

    TP / FP / FN / TN are computed using the utility-function definitions:
      TP  — positive patient (sepsis) correctly alarmed (utility reward +w_tp)
      FP  — negative patient (no sepsis) falsely alarmed (utility penalty w_fp)
      FN  — positive patient missed, no alarm (utility penalty w_fn)
      TN  — negative patient correctly not alarmed (utility  w_tn = 0)

    Strategy
    --------
    1. Hold out 20 % of training data for threshold calibration.
    2. Fit model on the remaining 80 %.
    3. Grid-search 20 × 20 = 400 (t_f, t_m) pairs in [0.10, 0.70].
    4. Select the pair minimising the objective above.
    5. Refit the final model on the full training set (no data is wasted
       at inference; the validation split is only used for threshold search).

    The chosen thresholds are stored in `per_group_thresholds_` on the
    returned wrapper and are consumed by evaluate_ts_single via logit
    rescaling so the global decision threshold (0.3) still applies uniformly.
    """
    import warnings
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.base import clone

    rng = np.random.default_rng(cfg.random_state)

    f_val = cfg.fairness.female_value
    m_val = cfg.fairness.male_value

    # Utility weights directly from PhysioNet 2019 challenge (via config)
    w_fn = cfg.fairness.utility_w_fn   # -2.0  (missed sepsis)
    w_fp = cfg.fairness.utility_w_fp   # -0.05 (false alarm)

    # ── Step 1: 80/20 split for threshold calibration ─────────────────────────
    strat = np.array([f"{y_i}_{s_i}" for y_i, s_i in zip(y, sensitive)])
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=cfg.random_state
    )
    try:
        tr_idx, val_idx = next(sss.split(X, strat))
    except ValueError:
        # Any (group, label) cell with < 2 samples breaks stratification
        n = len(y)
        perm = rng.permutation(n)
        val_n = max(1, int(0.2 * n))
        val_idx, tr_idx = perm[:val_n], perm[val_n:]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val, s_val = X[val_idx], y[val_idx], sensitive[val_idx]

    # ── Step 2: Fit calibration model on 80 % ─────────────────────────────────
    cal_model = clone(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal_model.fit(X_tr, y_tr)

    y_prob_val = cal_model.predict_proba(X_val)[:, 1]

    f_mask = s_val == f_val
    m_mask = s_val == m_val

    if not f_mask.any() or not m_mask.any():
        logger.warning(
            "threshold_optimization: one group absent from val split — "
            "falling back to t=0.3 for both groups"
        )
        final = clone(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final.fit(X, y)
        return X, y, sensitive, None, _PerGroupThresholdWrapper(
            final, {f_val: 0.3, m_val: 0.3}
        )

    y_f, p_f = y_val[f_mask], y_prob_val[f_mask]
    y_m, p_m = y_val[m_mask], y_prob_val[m_mask]

    # ── Step 3: Grid search over (t_female, t_male) ───────────────────────────
    thresholds = np.linspace(0.10, 0.70, 20)

    def _group_stats(y_true, y_pred_bin):
        """TPR, FPR, and per-row utility using PhysioNet TP/FP/FN/TN weights."""
        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())
        tpr = float((y_pred_bin[y_true == 1] == 1).mean()) if n_pos > 0 else 0.0
        fpr = float((y_pred_bin[y_true == 0] == 1).mean()) if n_neg > 0 else 0.0
        tp = int(((y_pred_bin == 1) & (y_true == 1)).sum())
        fp = int(((y_pred_bin == 1) & (y_true == 0)).sum())
        fn = int(((y_pred_bin == 0) & (y_true == 1)).sum())
        util = (
            cfg.fairness.utility_w_tp * tp
            + w_fp * fp
            + w_fn * fn
        ) / max(len(y_true), 1)
        return tpr, fpr, util

    best_score = np.inf
    best_tf, best_tm = 0.3, 0.3

    for t_f in thresholds:
        pf_bin = (p_f >= t_f).astype(int)
        tpr_f, fpr_f, util_f = _group_stats(y_f, pf_bin)

        for t_m in thresholds:
            pm_bin = (p_m >= t_m).astype(int)
            tpr_m, fpr_m, util_m = _group_stats(y_m, pm_bin)

            # Utility-weighted fairness gap using PhysioNet 2019 weights:
            #   |w_fn|=2.0  weights the TPR gap  (missing sepsis matters most)
            #   |w_fp|=0.05 weights the FPR gap  (false alarms matter less)
            fairness_score = (
                abs(w_fn) * abs(tpr_f - tpr_m)
                + abs(w_fp) * abs(fpr_f - fpr_m)
            )

            # Utility guard: penalise configurations that make either group
            # far worse off than the conservative global-threshold baseline
            util_penalty = (
                max(0.0, -0.25 - util_f) * 10.0
                + max(0.0, -0.25 - util_m) * 10.0
            )

            score = fairness_score + util_penalty

            if score < best_score:
                best_score = score
                best_tf, best_tm = float(t_f), float(t_m)

    logger.info(
        "threshold_optimization: t_female=%.3f  t_male=%.3f  "
        "objective=%.4f  (|w_fn|*|TPR_gap| + |w_fp|*|FPR_gap|)",
        best_tf, best_tm, best_score,
    )

    # ── Step 4: Refit on full training set ────────────────────────────────────
    final = clone(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(X, y)

    return X, y, sensitive, None, _PerGroupThresholdWrapper(
        final, {f_val: best_tf, m_val: best_tm}
    )


def get_mitigation(name: str) -> callable:
    if name not in MITIGATION_REGISTRY:
        raise ValueError(
            f"Unknown mitigation '{name}'. Available: {list(MITIGATION_REGISTRY)}"
        )
    return MITIGATION_REGISTRY[name]
