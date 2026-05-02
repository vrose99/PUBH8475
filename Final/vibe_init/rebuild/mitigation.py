"""
Bias mitigation strategies for sepsis prediction models.

Strategies (matching main pipeline):
  - none: baseline, no modification
  - reweighting: inverse-frequency sample weights per (group, label) cell
  - smote: oversample underrepresented gender group
  - threshold_optimization: per-group threshold search on validation set
"""

import logging
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _compute_reweighting(
    y: np.ndarray,
    sensitive: np.ndarray,
    female_val: str = 'F',
    male_val: str = 'M',
) -> np.ndarray:
    """
    Compute inverse-frequency weights: weight = 1 / P(group) / P(label|group).

    Normalized so weights sum to n_samples.
    Ensures each (group × label) cell contributes equally.
    """
    weights = np.ones(len(y), dtype=float)

    for g in [female_val, male_val]:
        for lbl in [0, 1]:
            mask = (sensitive == g) & (y == lbl)
            if mask.sum() > 0:
                # Weight inversely proportional to cell size
                weights[mask] = len(y) / (4 * mask.sum())

    return weights


def apply_none(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Baseline: no modification."""
    return X_train, y_train, None


def apply_reweighting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    female_val: str = 'F',
    male_val: str = 'M',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reweight samples so each (group × label) cell contributes equally.

    Returns:
        (X_train, y_train, sample_weights)
    """
    weights = _compute_reweighting(
        y_train, sensitive_train,
        female_val=female_val,
        male_val=male_val,
    )
    logger.debug(
        "Reweighting: weight range [%.3f, %.3f]", weights.min(), weights.max()
    )
    return X_train, y_train, weights


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    female_val: str = 'F',
    male_val: str = 'M',
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Oversample the underrepresented gender group to match the majority group's size.

    Uses SMOTE (Synthetic Minority Oversampling Technique) or falls back to
    random oversampling.

    Requires: imbalanced-learn (`pip install imbalanced-learn`)

    Returns:
        (X_train_resampled, y_train_resampled, None)
    """
    try:
        from imblearn.over_sampling import SMOTE, RandomOverSampler
    except ImportError:
        logger.warning("imbalanced-learn not installed, falling back to reweighting")
        return apply_reweighting(X_train, y_train, sensitive_train, female_val, male_val)

    rng = np.random.default_rng(random_state)

    f_mask = (sensitive_train == female_val) | (sensitive_train == 'Female') | (sensitive_train == 1)
    m_mask = (sensitive_train == male_val) | (sensitive_train == 'Male') | (sensitive_train == 0)
    n_female, n_male = f_mask.sum(), m_mask.sum()
    target = max(n_female, n_male)

    parts_X, parts_y, parts_s = [X_train], [y_train], [sensitive_train]

    for g_mask, g_val, n_g in [
        (f_mask, female_val, n_female),
        (m_mask, male_val, n_male),
    ]:
        if n_g >= target:
            continue  # already the majority group

        X_g, y_g = X_train[g_mask], y_train[g_mask]
        n_synth = target - n_g

        # Per-class target counts proportional to current distribution
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
                    random_state=random_state,
                )
                X_res, y_res = oversampler.fit_resample(X_g, y_g)
                method = "SMOTE"
            except Exception as exc:
                logger.debug("SMOTE failed, falling back to RandomOverSampler: %s", exc)
                oversampler = RandomOverSampler(
                    sampling_strategy=new_counts,
                    random_state=random_state,
                )
                X_res, y_res = oversampler.fit_resample(X_g, y_g)
                method = "RandomOverSampler"
        else:
            oversampler = RandomOverSampler(
                sampling_strategy=new_counts,
                random_state=random_state,
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
    return X_out[shuffle], y_out[shuffle], None


def apply_threshold_optimization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    sensitive_val: Optional[np.ndarray] = None,
    female_val: str = 'F',
    male_val: str = 'M',
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], dict]:
    """
    Per-group threshold optimization.

    Trains a logistic regression model on training data, then searches for
    optimal thresholds on validation set to maximize balanced accuracy per group.

    If validation set not provided, splits training data (80/20).

    Returns:
        (X_train, y_train, None, {'female_threshold': t_f, 'male_threshold': t_m})
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # Split training data if validation not provided
    if X_val is None or y_val is None or sensitive_val is None:
        X_train_fit, X_val, y_train_fit, y_val, s_train_fit, sensitive_val = train_test_split(
            X_train, y_train, sensitive_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )
    else:
        X_train_fit, y_train_fit, s_train_fit = X_train, y_train, sensitive_train

    # Train a simple logistic regression to get probability estimates
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_fit, y_train_fit)
    y_val_proba = lr.predict_proba(X_val)[:, 1]

    # Find optimal threshold per group to maximize balanced accuracy
    thresholds = {}
    for g_name, g_val in [('female_threshold', female_val), ('male_threshold', male_val)]:
        g_mask = sensitive_val == g_val
        if g_mask.sum() > 0:
            y_g = y_val[g_mask]
            proba_g = y_val_proba[g_mask]

            best_threshold = 0.5
            best_score = -1

            # Search for threshold that maximizes balanced accuracy (TPR + TNR) / 2
            for thresh in np.arange(0.1, 0.9, 0.01):
                y_pred = (proba_g >= thresh).astype(int)

                # Compute per-group balanced accuracy
                tp = ((y_pred == 1) & (y_g == 1)).sum()
                fp = ((y_pred == 1) & (y_g == 0)).sum()
                fn = ((y_pred == 0) & (y_g == 1)).sum()
                tn = ((y_pred == 0) & (y_g == 0)).sum()

                if (tp + fn) > 0 and (fp + tn) > 0:
                    tpr = tp / (tp + fn)  # Sensitivity
                    tnr = tn / (fp + tn)  # Specificity
                    balanced_acc = (tpr + tnr) / 2
                    if balanced_acc > best_score:
                        best_score = balanced_acc
                        best_threshold = thresh
        else:
            best_threshold = 0.5

        thresholds[g_name] = best_threshold

    logger.debug(
        "Threshold optimization: female=%.2f (score=%.3f), male=%.2f (score=%.3f)",
        thresholds['female_threshold'],
        best_score if g_name == 'female_threshold' else 0,
        thresholds['male_threshold'],
        best_score if g_name == 'male_threshold' else 0,
    )

    return X_train, y_train, None, thresholds


def get_mitigation(name: str):
    """
    Get mitigation strategy by name.

    Args:
        name: 'none', 'reweighting', 'smote', or 'threshold_optimization'

    Returns:
        Callable mitigation function
    """
    strategies = {
        'none': apply_none,
        'reweighting': apply_reweighting,
        'smote': apply_smote,
        'threshold_optimization': apply_threshold_optimization,
    }

    if name not in strategies:
        raise ValueError(f"Unknown mitigation strategy: {name}. "
                        f"Available: {list(strategies.keys())}")

    return strategies[name]
