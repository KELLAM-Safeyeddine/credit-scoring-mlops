"""Monotone sigmoid calibration of a frozen probability mixture."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import brier_score_loss, log_loss

from src.models.probability_mixture import ProbabilityMixture


class CalibratedMixture(ClassifierMixin, BaseEstimator):
    """Keep the trained mixture fixed and map its score to a default probability."""

    def __init__(self, mixture: ProbabilityMixture, slope: float, intercept: float):
        if not np.isfinite([slope, intercept]).all() or slope <= 0:
            raise ValueError("Calibration must be finite and strictly increasing.")
        self.mixture = mixture
        self.slope = slope
        self.intercept = intercept

    @property
    def n_features_in_(self):
        return self.mixture.n_features_in_

    @property
    def classes_(self):
        return self.mixture.classes_

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, x, y=None):
        raise ValueError("Frozen calibration: create a new independent protocol to refit.")

    def calibrate(self, scores):
        """Apply the same strictly increasing map to scores and the refusal threshold."""
        scores = np.asarray(scores, dtype=float)
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Raw scores must be finite probabilities.")
        return expit(self.slope * scores + self.intercept)

    def predict_proba(self, x):
        positive = self.calibrate(self.mixture.predict_proba(x)[:, 1])
        return np.column_stack([1 - positive, positive])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def fit_sigmoid(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit two unweighted logistic parameters on a separate calibration cohort only."""
    scores, y = np.asarray(scores, dtype=float), np.asarray(y)
    if scores.ndim != 1 or y.shape != scores.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Calibration needs aligned scores and both binary classes.")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid calibration scores.")

    def objective(parameters):
        z = parameters[0] * scores + parameters[1]
        error = expit(z) - y
        return np.mean(np.logaddexp(0, z) - y * z), np.array(
            [np.mean(error * scores), np.mean(error)]
        )

    result = minimize(
        objective,
        [1.0, float(logit(y.mean()) - scores.mean())],
        jac=True,
        method="L-BFGS-B",
        bounds=[(1e-8, None), (None, None)],
        options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1000},
    )
    if not result.success or result.x[0] <= 1e-7:
        raise ValueError("Monotone sigmoid calibration did not converge to a useful slope.")
    return float(result.x[0]), float(result.x[1])


def calibration_metrics(y, scores, bins: int = 10) -> dict:
    """Measure proper scoring rules and fixed equal-width reliability bins."""
    scores, y = np.asarray(scores), np.asarray(y)
    assignment = np.minimum((scores * bins).astype(int), bins - 1)
    reliability = []
    for index in range(bins):
        mask = assignment == index
        if mask.any():
            reliability.append(
                {
                    "lower": index / bins,
                    "upper": (index + 1) / bins,
                    "count": int(mask.sum()),
                    "mean_prediction": float(scores[mask].mean()),
                    "default_rate": float(y[mask].mean()),
                }
            )
    return {
        "brier": float(brier_score_loss(y, scores)),
        "log_loss": float(log_loss(y, scores, labels=[0, 1])),
        "ece": sum(
            row["count"] / len(y) * abs(row["mean_prediction"] - row["default_rate"])
            for row in reliability
        ),
        "mean_prediction": float(scores.mean()),
        "default_rate": float(y.mean()),
        "reliability_bins": reliability,
    }
