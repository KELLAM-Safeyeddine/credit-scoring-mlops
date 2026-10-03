"""Check deterministic mixture scores without inheriting randomized guarantees."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.models.constrained_study import deterministic_scores


class Oracle:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, x):
        return np.column_stack((1 - self.probabilities, self.probabilities))

    def predict(self, x):
        return (self.probabilities >= 0.5).astype(int)


def mixture(weights):
    return SimpleNamespace(
        weights_=pd.Series(weights, index=[2, 7]),
        predictors_=pd.Series({2: Oracle([0.2, 0.8]), 7: Oracle([0.7, 0.4])}),
    )


def test_soft_and_hard_scores_are_distinct_and_repeatable():
    reduction = mixture([1.0, 3.0])
    x = np.zeros((2, 1))
    np.testing.assert_allclose(
        deterministic_scores(reduction, x, "probability_average"), [0.575, 0.5]
    )
    np.testing.assert_allclose(deterministic_scores(reduction, x, "hard_vote"), [0.75, 0.25])
    np.testing.assert_array_equal(
        deterministic_scores(reduction, x, "hard_vote"),
        deterministic_scores(reduction, x, "hard_vote"),
    )


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [np.nan, 1], [np.inf, 1]])
def test_invalid_mixtures_fail_closed(weights):
    with pytest.raises(ValueError, match="weights"):
        deterministic_scores(mixture(weights), np.zeros((2, 1)), "hard_vote")


def test_zero_weight_oracle_is_not_called():
    reduction = mixture([1, 0])
    reduction.predictors_.loc[7] = None
    np.testing.assert_allclose(
        deterministic_scores(reduction, np.zeros((2, 1)), "probability_average"), [0.2, 0.8]
    )


def test_invalid_oracle_probability_is_rejected():
    reduction = mixture([1, 0])
    reduction.predictors_.loc[2] = Oracle([1.2, 0.3])
    with pytest.raises(ValueError, match="\u005b0, 1\u005d"):
        deterministic_scores(reduction, np.zeros((2, 1)), "probability_average")
