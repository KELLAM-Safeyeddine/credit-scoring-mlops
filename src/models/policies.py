"""Persistable comparative decision policies; protected attributes stay outside X."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models.fairness import intersection_groups


@dataclass
class ScoringPolicy:
    """Score and refuse with a single global threshold."""

    pipeline: Any
    threshold: float

    def scores(self, x: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(x)[:, 1]

    def predict(self, x: pd.DataFrame, sensitive: pd.DataFrame | None = None) -> np.ndarray:
        return (self.scores(x) >= self.threshold).astype("int8")


@dataclass
class ThresholdPolicy:
    """Group-dependent randomized comparator; not intended for API deployment."""

    pipeline: Any
    optimizer: Any
    known_groups: tuple[str, ...]
    fallback_threshold: float
    random_state: int

    def scores(self, x: pd.DataFrame) -> np.ndarray:
        """Underlying ranking score, not the probability of randomized refusal."""
        return self.pipeline.predict_proba(x)[:, 1]

    def predict(self, x: pd.DataFrame, sensitive: pd.DataFrame | None = None) -> np.ndarray:
        if sensitive is None or not x.index.equals(sensitive.index):
            raise ValueError("ThresholdOptimizer comparator needs aligned audit groups.")
        groups = intersection_groups(sensitive)
        mask = groups.isin(self.known_groups)
        decisions = (self.scores(x) >= self.fallback_threshold).astype("int8")
        if mask.any():
            decisions[mask] = self.optimizer.predict(
                x.loc[mask], sensitive_features=groups[mask], random_state=self.random_state
            )
        return decisions


@dataclass
class ReductionPolicy:
    """ExponentiatedGradient's mixture score and reproducible randomized decisions."""

    reduction: Any
    random_state: int

    def scores(self, x: pd.DataFrame) -> np.ndarray:
        """Probability of refusal under the mixture; not a calibrated default PD."""
        result = np.zeros(len(x))
        for index, weight in self.reduction.weights_.items():
            if weight > 0:
                result += weight * self.reduction.predictors_.loc[index].predict(x)
        return result

    def predict(self, x: pd.DataFrame, sensitive: pd.DataFrame | None = None) -> np.ndarray:
        return self.reduction.predict(x, random_state=self.random_state).astype("int8")
