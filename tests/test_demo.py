"""Measured synthetic lifecycle conditions without Docker or Kaggle in CI."""

import numpy as np
import pytest
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from src.features.contracts import EXCLUDED
from src.models.fairness import audit_fairness, intersection_groups, reweighing_weights
from src.monitoring.demo_data import cohort, training_cohort


def test_demo_cohorts_are_disjoint_and_group_inputs_excluded():
    train, _, _ = training_cohort(1)
    val, y, sensitive = cohort(123)
    production, _, _ = cohort(456)
    assert not EXCLUDED.intersection(train.columns)
    assert len(train.index.intersection(val.index)) == 0
    assert len(production.index.intersection(val.index)) == 0
    assert train.index.is_unique and val.index.equals(y.index) and val.index.equals(sensitive.index)
    with pytest.raises(ValueError):
        training_cohort(4)


def test_demo_improved_biased_candidate_and_measured_mitigation(config):
    x_val, y_val, audit = cohort(123)
    results = []
    for version in (1, 2, 3):
        x, y, sensitive = training_cohort(version)
        weights = reweighing_weights(y, intersection_groups(sensitive)) if version == 3 else None
        model = LGBMClassifier(**config["demo"]["model"], verbosity=-1).fit(
            x, y, sample_weight=weights
        )
        scores = model.predict_proba(x_val)[:, 1]
        report = audit_fairness(
            y_val, (scores >= config["demo"]["threshold"]).astype(int), audit, config["fairness"]
        )
        results.append((roc_auc_score(y_val, scores), report))
    assert results[0][0] >= config["quality_gate"]["min_auc"]
    assert results[1][0] > results[0][0]
    assert results[1][1]["attributes"]["CODE_GENDER"]["disparate_impact"] < 0.8
    assert results[2][0] >= results[0][0] - config["quality_gate"]["max_auc_drop_vs_prod"]
    for version in (0, 2):
        for report in results[version][1]["attributes"].values():
            assert report["sufficient_support"]
            assert report["disparate_impact"] >= 0.8
            assert report["equal_opportunity_difference"] <= 0.10
    assert np.isfinite(results[2][0])
