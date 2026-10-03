"""Frozen, group-free probability averaging of fitted tree classifiers."""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ProbabilityMixture(ClassifierMixin, BaseEstimator):
    """Serve fitted oracles without randomization or implicit retraining."""

    def __init__(
        self, predictors: tuple[Any, ...], weights: tuple[float, ...], background: np.ndarray
    ):
        self.predictors = predictors
        self.weights = weights
        self.background = background
        if not predictors or len(predictors) != len(weights):
            raise ValueError("Oracle and weight counts differ.")
        values = np.asarray(weights, dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Oracle weights must be positive and finite.")
        dimensions = {model.n_features_in_ for model in predictors}
        if len(dimensions) != 1:
            raise ValueError("Oracle feature dimensions differ.")
        if (
            background.ndim != 2
            or background.shape[1] not in dimensions
            or not 1 <= len(background) <= 100
        ):
            raise ValueError("Provide 1–100 transformed train background rows.")

    @property
    def n_features_in_(self) -> int:
        return self.predictors[0].n_features_in_

    @property
    def classes_(self) -> np.ndarray:
        return np.array([0, 1])

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def fit(self, x: Any, y: Any = None) -> "ProbabilityMixture":
        raise RuntimeError("Frozen probability mixtures cannot be retrained.")

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return deterministic ranking scores; these are not calibrated PDs."""
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError("Mixture input dimension differs.")
        scores = sum(
            weight * model.predict_proba(x)[:, 1]
            for model, weight in zip(self.predictors, self.weights, strict=True)
        ) / sum(self.weights)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Invalid mixture scores.")
        return np.column_stack((1 - scores, scores))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Default classifier convention; business policy uses its frozen threshold."""
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)
