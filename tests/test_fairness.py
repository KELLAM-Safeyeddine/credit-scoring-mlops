"""Acceptance semantics, support policy and non-degenerate mitigation checks."""

import numpy as np
import pandas as pd
import pytest

from src.models.fairness import audit_fairness, reweighing_weights
from src.models.mitigate import select_fair_threshold


def audit_settings(config):
    cfg = config["fairness"]
    cfg.update(min_group_size=1, min_class_count=1, sensitive_features=["CODE_GENDER"])
    cfg["required_groups"] = {"CODE_GENDER": ["F", "M"]}
    return cfg


def test_default_positive_acceptance_and_equal_opportunity(config):
    y = pd.Series([1, 1, 0, 0, 1, 1, 0, 0])
    sensitive = pd.DataFrame({"CODE_GENDER": ["F"] * 4 + ["M"] * 4})
    pred = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    report = audit_fairness(y, pred, sensitive, audit_settings(config))
    attr = report["attributes"]["CODE_GENDER"]
    assert attr["by_group"]["F"]["acceptance_rate"] == 0.5
    assert attr["by_group"]["M"]["tpr"] == 0.5
    assert attr["by_group"]["M"]["fpr"] == 0.5
    assert attr["disparate_impact"] == 1
    assert attr["demographic_parity_difference"] == 0
    assert attr["equal_opportunity_difference"] == 0.5
    assert attr["equalized_odds_difference"] == 0.5
    assert not report["passes_thresholds"]


def test_disparate_impact_measures_acceptance_not_refusal(config):
    y = pd.Series([1, 1, 0, 0] * 2)
    sensitive = pd.DataFrame({"CODE_GENDER": ["F"] * 4 + ["M"] * 4})
    pred = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    attr = audit_fairness(y, pred, sensitive, audit_settings(config))["attributes"]["CODE_GENDER"]
    assert attr["disparate_impact"] == pytest.approx(1 / 3)
    assert attr["by_group"]["F"]["acceptance_rate"] == 0.75


def test_unknown_gender_reports_undefined_tpr_and_coverage(config):
    y = pd.Series([0, 1, 0, 1, 0])
    sensitive = pd.DataFrame({"CODE_GENDER": ["F", "F", "M", "M", "XNA"]})
    attr = audit_fairness(y, y.to_numpy(), sensitive, audit_settings(config))["attributes"][
        "CODE_GENDER"
    ]
    assert attr["by_group"]["XNA"]["tpr"] is None
    assert not attr["by_group"]["XNA"]["comparison_eligible"]
    assert attr["comparison_coverage"] == 0.8
    assert attr["passes_thresholds"]


def test_missing_required_group_and_zero_acceptance_fail(config):
    settings = audit_settings(config)
    y = pd.Series([0, 1, 0, 1])
    sensitive = pd.DataFrame({"CODE_GENDER": ["F", "F", "M", "M"]})
    all_refused = audit_fairness(y, np.ones(4), sensitive, settings)
    assert all_refused["attributes"]["CODE_GENDER"]["disparate_impact"] is None
    assert not all_refused["passes_thresholds"]
    sensitive["CODE_GENDER"] = "F"
    assert not audit_fairness(y, y.to_numpy(), sensitive, settings)["passes_thresholds"]


def test_fairness_rejects_misaligned_audit_rows(config):
    y = pd.Series([0, 1, 0, 1])
    sensitive = pd.DataFrame({"CODE_GENDER": ["F", "F", "M", "M"]}, index=[1, 0, 2, 3])
    with pytest.raises(ValueError, match="aligned"):
        audit_fairness(y, y.to_numpy(), sensitive, audit_settings(config))


def test_reweighing_restores_label_independence_in_training():
    y = pd.Series([0, 0, 0, 1, 0, 1, 1, 1])
    groups = pd.Series(["F|young"] * 4 + ["M|old"] * 4)
    weights = reweighing_weights(y, groups)
    assert weights.mean() == pytest.approx(1)
    assert np.all(weights > 0)
    for group in groups.unique():
        mask = groups == group
        assert np.average(y[mask], weights=weights[mask]) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="align"):
        reweighing_weights(y, groups.sort_index(ascending=False))


def test_fair_threshold_does_not_accept_everyone(config):
    audit_settings(config)
    config["train"]["threshold"].update(min=0.1, max=0.9, steps=9)
    y = pd.Series([0, 0, 1, 1] * 2)
    sensitive = pd.DataFrame({"CODE_GENDER": ["F"] * 4 + ["M"] * 4})
    scores = np.array([0.1, 0.2, 0.7, 0.8] * 2)
    threshold, feasible = select_fair_threshold(y, scores, sensitive, config)
    assert feasible
    assert threshold <= 0.7
    assert np.mean(scores[y == 1] >= threshold) >= 0.5


def test_impossible_utility_floor_is_reported_not_claimed(config):
    audit_settings(config)
    y = pd.Series([0, 1, 0, 1])
    sensitive = pd.DataFrame({"CODE_GENDER": ["F", "F", "M", "M"]})
    _, feasible = select_fair_threshold(y, np.zeros(4), sensitive, config)
    assert not feasible
