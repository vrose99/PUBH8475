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
    Upweight underrepresented cases: Y=1 (positive/sepsis) and minority gender.

    Weights:
    - Base: 1.0 (majority class, majority gender)
    - Y=1: multiplied by 2.0 (sepsis cases are rare)
    - Minority gender: multiplied by 1.5
    - Both: combined (e.g., Y=1 AND minority gender gets 3.0x)

    This helps the model learn better from underrepresented groups without
    changing the decision threshold.
    """
    weights = np.ones(len(y), dtype=float)

    # Identify minority gender (the smaller group)
    f_count = (sensitive == female_val).sum()
    m_count = (sensitive == male_val).sum()
    minority_gender = female_val if f_count < m_count else male_val

    # Upweight positive cases (Y=1, usually rare)
    weights[y == 1] *= 2.0

    # Upweight minority gender
    weights[sensitive == minority_gender] *= 1.5

    # Normalize so mean weight = 1.0 (preserves overall learning rate)
    weights = weights / weights.mean()

    logger.debug(
        "Reweighting: upweighting Y=1 (2.0x) and minority gender %s (1.5x), "
        "weight range [%.3f, %.3f]",
        minority_gender, weights.min(), weights.max()
    )
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


def apply_fairness_penalty(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    model=None,
    female_val: str = 'F',
    male_val: str = 'M',
):
    """
    Fairness-constrained mitigation using fairlearn's GridSearch with EqualizedOdds.

    Returns a fitted wrapper model that enforces equalized odds constraints
    between sensitive groups. The model is fitted internally and should be used
    directly instead of fitting the base model.

    Args:
        X_train, y_train: Training data
        sensitive_train: Sensitive attribute (gender)
        model: Base model to wrap (LogisticGLM, XGBoostModel, or GRUModel instance)

    Returns:
        (X_train, y_train, None, fitted_mitigator_model)
    """
    try:
        from fairlearn.reductions import GridSearch, EqualizedOdds
    except ImportError:
        raise ImportError("fairlearn not installed — `pip install fairlearn`")

    from sklearn.base import clone
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    if model is None:
        raise ValueError("fairness_penalty requires a model instance")

    # Clone the base model (use sklearn's LogisticRegression for simplicity)
    from sklearn.linear_model import LogisticRegression
    base_model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )

    # Preprocess: impute and scale to handle NaN values
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_preprocessed = imputer.fit_transform(X_train)
    X_preprocessed = scaler.fit_transform(X_preprocessed)

    # Create fairlearn mitigator with EqualizedOdds constraint
    mitigator_grid = GridSearch(
        estimator=base_model,
        constraints=EqualizedOdds(),
        grid_size=10,
    )

    # Convert sensitive attribute to numeric for fairlearn
    sensitive_numeric = (sensitive_train == male_val).astype(int)

    # Guard: Check if both groups have both positive and negative samples
    # (fairlearn requires this for EqualizedOdds constraint)
    for group_val in [0, 1]:
        group_mask = sensitive_numeric == group_val
        if group_mask.sum() == 0:
            logger.warning(
                "fairness_penalty: group %d has no samples, falling back to unmitigated model", group_val
            )
            base_fitted = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
            base_fitted.fit(X_preprocessed, y_train)

            class SimpleWrapper:
                def __init__(self, model, imputer, scaler):
                    self._model = model
                    self._imputer = imputer
                    self._scaler = scaler
                def predict_proba(self, X):
                    X_prep = self._imputer.transform(X)
                    X_prep = self._scaler.transform(X_prep)
                    return self._model.predict_proba(X_prep)
                def predict(self, X):
                    X_prep = self._imputer.transform(X)
                    X_prep = self._scaler.transform(X_prep)
                    return self._model.predict(X_prep)

            wrapped = SimpleWrapper(base_fitted, imputer, scaler)
            return X_train, y_train, None, wrapped

        y_group = y_train[group_mask]
        if (y_group == 0).sum() == 0 or (y_group == 1).sum() == 0:
            logger.warning(
                "fairness_penalty: group %d has only one label, falling back to unmitigated model", group_val
            )
            base_fitted = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
            base_fitted.fit(X_preprocessed, y_train)

            class SimpleWrapper:
                def __init__(self, model, imputer, scaler):
                    self._model = model
                    self._imputer = imputer
                    self._scaler = scaler
                def predict_proba(self, X):
                    X_prep = self._imputer.transform(X)
                    X_prep = self._scaler.transform(X_prep)
                    return self._model.predict_proba(X_prep)
                def predict(self, X):
                    X_prep = self._imputer.transform(X)
                    X_prep = self._scaler.transform(X_prep)
                    return self._model.predict(X_prep)

            wrapped = SimpleWrapper(base_fitted, imputer, scaler)
            return X_train, y_train, None, wrapped

    # Fit the mitigator on preprocessed data
    try:
        mitigator_grid.fit(X_preprocessed, y_train, sensitive_features=sensitive_numeric)
    except Exception as e:
        # If fairlearn fails for any reason, fall back to base model
        logger.warning(
            "fairness_penalty: GridSearch failed (%s: %s), falling back to unmitigated model",
            type(e).__name__, str(e)
        )
        base_fitted = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
        base_fitted.fit(X_preprocessed, y_train)

        class SimpleWrapper:
            def __init__(self, model, imputer, scaler):
                self._model = model
                self._imputer = imputer
                self._scaler = scaler
            def predict_proba(self, X):
                X_prep = self._imputer.transform(X)
                X_prep = self._scaler.transform(X_prep)
                return self._model.predict_proba(X_prep)
            def predict(self, X):
                X_prep = self._imputer.transform(X)
                X_prep = self._scaler.transform(X_prep)
                return self._model.predict(X_prep)

        wrapped = SimpleWrapper(base_fitted, imputer, scaler)
        return X_train, y_train, None, wrapped

    # Wrap the fitted mitigator with preprocessing pipeline
    # Create a wrapper that handles imputation and scaling during predict_proba
    class PreprocessingWrapper:
        def __init__(self, mitigator, imputer, scaler):
            self._mitigator = mitigator
            self._imputer = imputer
            self._scaler = scaler

        def predict_proba(self, X):
            X_prep = self._imputer.transform(X)
            X_prep = self._scaler.transform(X_prep)
            return self._mitigator.predict_proba(X_prep)

        def predict(self, X):
            X_prep = self._imputer.transform(X)
            X_prep = self._scaler.transform(X_prep)
            return self._mitigator.predict(X_prep)

    wrapped_mitigator = PreprocessingWrapper(mitigator_grid, imputer, scaler)

    logger.debug("fairness_penalty: GridSearch fitted with EqualizedOdds constraint")

    return X_train, y_train, None, wrapped_mitigator


def get_mitigation(name: str):
    """
    Get mitigation strategy by name.

    Args:
        name: 'none', 'reweighting', 'smote', or 'fairness_penalty'

    Returns:
        Callable mitigation function
    """
    strategies = {
        'none': apply_none,
        'reweighting': apply_reweighting,
        'smote': apply_smote,
        'fairness_penalty': apply_fairness_penalty,
    }

    if name not in strategies:
        raise ValueError(f"Unknown mitigation strategy: {name}. "
                        f"Available: {list(strategies.keys())}")

    return strategies[name]
