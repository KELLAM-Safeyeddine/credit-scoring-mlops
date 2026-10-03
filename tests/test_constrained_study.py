"""Check deterministic mixture scores without inheriting randomized guarantees."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.models.constrained_study import (
    deterministic_scores,
    selected_candidate,
    selection_config,
    study_trials,
)


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


def test_weighted_probability_roundoff_is_clipped_without_accepting_bad_oracles():
    weights = np.array(
        [
            0.43715191887233074,
            0.8326781960578374,
            0.7002651020022491,
            0.31236664138204107,
            0.8322598013952011,
            0.8047643574968019,
            0.38747837903017446,
            0.2883281039302441,
            0.6824955039749755,
            0.1397524836093098,
        ]
    )
    assert sum(float(w) for w in weights) / weights.sum() > 1
    reduction = SimpleNamespace(
        weights_=pd.Series(weights),
        predictors_=pd.Series([Oracle([1.0, 0.0]) for _ in weights]),
    )
    np.testing.assert_array_equal(
        deterministic_scores(reduction, np.zeros((2, 1)), "probability_average"), [1.0, 0.0]
    )
    reduction.predictors_.iloc[0] = Oracle([1.01, 0.0])
    with pytest.raises(ValueError, match="Oracle scores"):
        deterministic_scores(reduction, np.zeros((2, 1)), "probability_average")


def test_invalid_oracle_probability_is_rejected():
    reduction = mixture([1, 0])
    reduction.predictors_.loc[2] = Oracle([1.2, 0.3])
    with pytest.raises(ValueError, match="\u005b0, 1\u005d"):
        deterministic_scores(reduction, np.zeros((2, 1)), "probability_average")


def test_study_plans_are_bounded_and_copied():
    cfg = load_config()
    assert len(study_trials(cfg["constrained_development"])) == 4
    settings = cfg["constrained_capacity_development"]
    trials = study_trials(settings)
    assert len(trials) == 2
    trials[0]["lightgbm"]["n_estimators"] = 1
    assert settings["trials"][0]["lightgbm"]["n_estimators"] == 300


@pytest.mark.parametrize(
    "mutation",
    ["excess_fits", "unsafe_name", "sensitive_parameter", "excess_iterations", "duplicate_name"],
)
def test_invalid_study_plan_fails_before_training(mutation):
    settings = load_config()["constrained_capacity_development"]
    if mutation == "excess_fits":
        settings["trials"] *= 3
    elif mutation == "unsafe_name":
        settings["trials"][0]["name"] = "../outside"
    elif mutation == "sensitive_parameter":
        settings["trials"][0]["lightgbm"]["CODE_GENDER"] = True
    elif mutation == "excess_iterations":
        settings["trials"][0]["max_iter"] = 21
    else:
        settings["trials"][1]["name"] = settings["trials"][0]["name"]
    with pytest.raises(ValueError):
        study_trials(settings)


def test_prospective_margins_preserve_production_rules():
    from pathlib import Path

    cfg = load_config(Path("configs/protocol-v3.yaml"))
    settings = cfg["independent_protocol"]["qualification_plan"]
    cfg["qualification"] = settings
    selected = selection_config(cfg, settings)
    assert selected["quality_gate"]["min_auc"] == 0.745
    assert selected["fairness"]["min_disparate_impact"] == 0.85
    assert selected["fairness"]["max_equal_opp_diff"] == 0.075
    assert cfg["quality_gate"]["min_auc"] == 0.74
    assert cfg["fairness"]["min_disparate_impact"] == 0.8
    assert cfg["fairness"]["max_equal_opp_diff"] == 0.1


def test_prospective_plan_cannot_be_changed_after_freeze():
    import copy
    from pathlib import Path

    cfg = load_config(Path("configs/protocol-v3.yaml"))
    cfg["qualification"] = cfg["independent_protocol"]["qualification_plan"]
    altered = copy.deepcopy(cfg["qualification"])
    altered["trials"][0]["max_iter"] = 16
    with pytest.raises(ValueError, match="frozen selection plan"):
        selection_config(cfg, altered)
    cfg["qualification"]["validation_selection_margin"]["min_disparate_impact"] = 0.79
    with pytest.raises(ValueError, match="selection margins"):
        selection_config(cfg, cfg["qualification"])


def test_selection_excludes_diagnostics_and_uses_stable_ties():
    def row(name, auc, recall, passed=True):
        return {
            "fit": name,
            "mode": "probability_average",
            "run_id": name,
            "snapshot": f"{name}.pkl",
            "screen_passed": passed,
            "metrics": {"validation_auc": auc, "validation_recall": recall},
        }

    rows = [row("diagnostic", 0.99, 0.99, False), row("first", 0.75, 0.7)]
    assert selected_candidate(rows)["fit"] == "first"
    rows.append(row("better-recall", 0.75, 0.8))
    rows.append(row("same-recall-later", 0.75, 0.8))
    assert selected_candidate(rows)["fit"] == "better-recall"
    assert selected_candidate([rows[0]]) is None
