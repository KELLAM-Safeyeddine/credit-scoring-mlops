"""Controlled simulations, delayed evidence and actual Evidently metrics."""

import json

import numpy as np
import pandas as pd
import pytest

from src.data.simulate_production import assign_batches, scenario_features
from src.monitoring.alerts import evaluate_alerts
from src.monitoring.api_logs import read_api_events, report_api_events
from src.monitoring.drift_report import (
    auc_interval,
    build_performance_report,
    build_population_report,
)
from src.monitoring.fairness_monitor import monitor_fairness


def test_financial_shifts_recompute_ratios_without_age(config):
    frame = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [100.0],
            "AMT_CREDIT": [200.0],
            "AMT_ANNUITY": [10.0],
            "CREDIT_INCOME_RATIO": [2.0],
            "ANNUITY_INCOME_RATIO": [0.1],
            "CREDIT_TERM": [20.0],
        }
    )
    changed = scenario_features(
        frame, pd.DataFrame({"CODE_GENDER": ["F"]}), 4, config["monitoring"]
    )
    assert changed.AMT_INCOME_TOTAL.iloc[0] == 80
    assert changed.AMT_CREDIT.iloc[0] == 240
    assert changed.CREDIT_INCOME_RATIO.iloc[0] == 3
    assert changed.ANNUITY_INCOME_RATIO.iloc[0] == 0.125
    assert changed.CREDIT_TERM.iloc[0] == 24
    assert list(changed) == list(frame)


def test_targeted_degradation_changes_inputs_only(config):
    frame = pd.DataFrame(
        {
            "EXT_SOURCE_1": [0.8, 0.8],
            "EXT_SOURCE_2": [0.6, 0.6],
            "EXT_SOURCE_3": [0.4, 0.4],
            "EXT_SOURCE_MEAN": [0.6, 0.6],
        }
    )
    changed = scenario_features(
        frame, pd.DataFrame({"CODE_GENDER": ["F", "M"]}), 7, config["monitoring"]
    )
    assert changed.EXT_SOURCE_MEAN.iloc[0] == pytest.approx(0.03)
    pd.testing.assert_series_equal(frame.iloc[1], changed.iloc[1])


def test_population_shift_is_disjoint_and_reproducible(config):
    index = pd.Index(range(800))
    sensitive = pd.DataFrame(
        {"AGE_GROUP": np.where(np.arange(800) % 4 == 0, "<30", "45-60")}, index=index
    )
    batches = assign_batches(index, sensitive, config["monitoring"], 42)
    assert len(np.unique(np.concatenate(batches))) == 800
    assert sensitive.loc[batches[5], "AGE_GROUP"].eq("<30").mean() == 0.6
    repeated = assign_batches(index, sensitive, config["monitoring"], 42)
    for first, second in zip(batches, repeated, strict=True):
        np.testing.assert_array_equal(first, second)


def test_auc_confidence_and_evidently_performance(tmp_path, config):
    labels = pd.Series([0, 1] * 50)
    predicted = pd.DataFrame(
        {"probability_default": np.where(labels == 1, 0.9, 0.1), "predicted_default": labels}
    )
    settings = config["monitoring"] | {"auc_bootstrap_samples": 20}
    result = build_performance_report(
        labels, predicted, labels, predicted, tmp_path, 0.37, settings, 42
    )
    assert result["auc"] == result["auc_upper"] == result["evidently_metrics"]["RocAuc"] == 1
    assert (tmp_path / "performance.html").stat().st_size > 1000
    assert auc_interval(pd.Series([0, 0]), pd.Series([0.1, 0.2]), settings, 42)["auc"] is None


def test_population_report_uses_audit_separately(tmp_path, config):
    reference = pd.DataFrame(
        {"CODE_GENDER": ["F", "M"] * 100, "AGE_GROUP": ["<30", "30-45", "45-60", "60+"] * 50}
    )
    current = reference.copy()
    current["AGE_GROUP"] = "<30"
    result = build_population_report(reference, current, tmp_path, config["monitoring"])
    assert result["AGE_GROUP"] > 0.2
    assert result["CODE_GENDER"] == 0


def test_alerts_keep_noise_and_pending_outcomes_visible(config):
    drift = {"drift_feature_share": 0, "score_psi": 0.02, "auc": 0.66, "auc_upper": 0.74}
    fairness = {"attributes": {"CODE_GENDER": {"disparate_impact": 0.9}}}
    result = evaluate_alerts(drift, fairness, config["monitoring"])
    assert not result["trigger_retraining"] and result["raw_auc_below_floor"]
    assert evaluate_alerts(
        drift, fairness, config["monitoring"] | {"auc_alert_rule": "point_estimate_below_floor"}
    )["alerts"] == ["performance"]
    drift.update(auc=None, auc_upper=None)
    assert "performance" in evaluate_alerts(drift, fairness, config["monitoring"])["not_evaluable"]
    drift.update(auc=0.3, auc_upper=0.4)
    assert "performance" in evaluate_alerts(drift, fairness, config["monitoring"])["alerts"]


def test_unsupported_group_does_not_claim_fairness(config):
    sensitive = pd.DataFrame({"CODE_GENDER": ["M", "F"], "AGE_GROUP": ["<30", "60+"]})
    result = monitor_fairness(pd.Series([0, 1]), sensitive, config["fairness"], min_group_size=20)
    assert result["attributes"]["CODE_GENDER"]["disparate_impact"] is None
    assert not result["attributes"]["CODE_GENDER"]["sufficient_support"]


def test_api_event_adapter_preserves_missingness_and_model_contract(tmp_path):
    events = [
        {
            "request_id": "a",
            "model_run_id": "run",
            "default_score": 0.2,
            "predicted_default": 0,
            "features": {"AMT_CREDIT": 100.0, "EXT_SOURCE_1": None},
        },
        {"request_id": "other", "model_run_id": "old", "features": {}},
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(event) for event in events))
    x, predictions = read_api_events(path, ["AMT_CREDIT", "EXT_SOURCE_1"], "run")
    assert x.index.equals(predictions.index) and pd.isna(x.EXT_SOURCE_1.iloc[0])
    with pytest.raises(ValueError, match="contract"):
        read_api_events(path, ["EXT_SOURCE_1", "AMT_CREDIT"], "run")


def test_api_events_feed_evidently_with_missing_input_alert(tmp_path, config):
    from sklearn.linear_model import LogisticRegression

    reference = pd.DataFrame(
        {"AMT_CREDIT": np.arange(100, dtype=float), "EXT_SOURCE_1": np.linspace(0, 1, 100)}
    )
    pipeline = LogisticRegression().fit(reference, np.arange(100) > 50)
    events = [
        {
            "request_id": str(i),
            "model_run_id": "run",
            "default_score": 0.2,
            "predicted_default": 0,
            "features": {"AMT_CREDIT": float(i), "EXT_SOURCE_1": None},
        }
        for i in range(30)
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(event) for event in events))
    report = report_api_events(
        path,
        {"columns": list(reference), "run_id": "run", "pipeline": pipeline},
        reference,
        tmp_path / "report",
        config["monitoring"],
    )
    assert report["requests"] == 30
    assert report["missingness_delta"]["EXT_SOURCE_1"] == 1
    assert "EXT_SOURCE_1" in report["untestable_current_features"]
    assert (tmp_path / "report/drift.html").exists()
    alerts = evaluate_alerts(report, {"attributes": {}}, config["monitoring"])
    assert "data_missingness.EXT_SOURCE_1" in alerts["alerts"]
