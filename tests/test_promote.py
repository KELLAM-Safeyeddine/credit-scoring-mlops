"""Gate boundaries and actual alias promotions in an isolated SQLite Registry."""

import copy
from unittest.mock import Mock

import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.promote import SHAP_FILES, champion_version, quality_gate, register_and_promote


def valid_evidence(config):
    return {
        "metrics": {
            "validation_auc": 0.76,
            "validation_recall": 0.6,
            "validation_acceptance_rate": 0.75,
            "requires_sensitive_at_inference": False,
            "randomized_decision": False,
        },
        "evaluation_data_sha256": "toy-validation-hash",
        "shap_verified": True,
        "shap_files": sorted(SHAP_FILES),
        "fairness_report": {
            "attributes": {
                attr: {
                    "sufficient_support": True,
                    "eligible_groups": config["fairness"]["required_groups"][attr],
                    "disparate_impact": 0.85,
                    "equal_opportunity_difference": 0.08,
                }
                for attr in config["fairness"]["sensitive_features"]
            }
        },
    }


def test_gate_accepts_all_rules_and_exact_boundaries(config):
    candidate = valid_evidence(config)
    candidate["metrics"]["validation_auc"] = 0.74
    for report in candidate["fairness_report"]["attributes"].values():
        report.update(disparate_impact=0.8, equal_opportunity_difference=0.10)
    passed, rules = quality_gate(candidate, None, config)
    assert passed and all(rule["passed"] for rule in rules.values())


@pytest.mark.parametrize("auc", [0.7399, None, float("nan"), float("inf")])
def test_gate_rejects_invalid_or_low_auc(config, auc):
    candidate = valid_evidence(config)
    candidate["metrics"]["validation_auc"] = auc
    passed, rules = quality_gate(candidate, None, config)
    assert not passed and not rules["minimum_auc"]["passed"]


@pytest.mark.parametrize("attr", ["CODE_GENDER", "AGE_GROUP"])
def test_gate_rejects_unfair_groups(config, attr):
    candidate = valid_evidence(config)
    candidate["fairness_report"]["attributes"][attr]["disparate_impact"] = 0.7999
    passed, rules = quality_gate(candidate, None, config)
    assert not passed and not rules[f"{attr}.disparate_impact"]["passed"]
    candidate["fairness_report"]["attributes"][attr].update(
        disparate_impact=0.9, equal_opportunity_difference=0.1001
    )
    assert not quality_gate(candidate, None, config)[0]


def test_gate_rejects_missing_evidence_and_degenerate_policy(config):
    candidate = valid_evidence(config)
    candidate["shap_verified"] = False
    assert not quality_gate(candidate, None, config)[0]
    candidate["shap_verified"] = True
    candidate["shap_files"] = []
    assert not quality_gate(candidate, None, config)[0]
    candidate["shap_files"] = sorted(SHAP_FILES)
    candidate["fairness_report"]["attributes"]["AGE_GROUP"]["eligible_groups"] = ["<30", "30-45"]
    assert not quality_gate(candidate, None, config)[0]
    candidate = valid_evidence(config)
    candidate["metrics"]["validation_recall"] = 0
    assert not quality_gate(candidate, None, config)[0]
    candidate = valid_evidence(config)
    candidate["metrics"]["requires_sensitive_at_inference"] = True
    assert not quality_gate(candidate, None, config)[0]


def test_gate_requires_comparable_production_auc(config):
    candidate = valid_evidence(config)
    prod = copy.deepcopy(candidate)
    prod["metrics"]["validation_auc"] = 0.77
    assert not quality_gate(candidate, prod, config)[0]
    prod["metrics"]["validation_auc"] = 0.765
    assert quality_gate(candidate, prod, config)[0]
    prod["evaluation_data_sha256"] = "other-split"
    passed, rules = quality_gate(candidate, prod, config)
    assert not passed and not rules["comparable_validation_data"]["passed"]


def test_changed_branch_cannot_mutate_registry(config):
    client = Mock()
    client.get_registered_model.return_value.aliases = {}
    with pytest.raises(RuntimeError, match="no Registry mutation"):
        register_and_promote(valid_evidence(config), config, client, expected_pass=False)
    client.log_dict.assert_not_called()
    client.set_registered_model_alias.assert_not_called()
    client.search_model_versions.assert_not_called()


def test_registry_promotes_preserves_previous_and_rejects_idempotently(tmp_path, config):
    previous_uri = mlflow.get_tracking_uri()
    cfg = copy.deepcopy(config)
    cfg["mlflow"]["registered_model_name"] = "isolated-credit-test"
    try:
        mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'registry.db').as_posix()}")
        experiment_id = mlflow.create_experiment(
            "isolated-registry", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        client = mlflow.MlflowClient()
        x = pd.DataFrame({"score": [0.0, 1.0, 2.0, 3.0]})
        model = LogisticRegression().fit(x, [0, 0, 1, 1])

        def create(auc):
            evidence = valid_evidence(cfg)
            evidence["metrics"]["validation_auc"] = auc
            with mlflow.start_run(experiment_id=experiment_id) as run:
                info = mlflow.sklearn.log_model(
                    model, name="model", pip_requirements=["scikit-learn==1.7.2"]
                )
            evidence.update(run_id=run.info.run_id, model_uri=info.model_uri)
            return evidence

        first = register_and_promote(create(0.76), cfg, client)
        assert first["passed"] and first["champion_version"] == "1"
        second = register_and_promote(create(0.77), cfg, client)
        assert second["passed"] and second["champion_version"] == "2"
        old = client.get_model_version_by_alias(cfg["mlflow"]["registered_model_name"], "previous")
        assert str(old.version) == "1"
        candidate = create(0.73)
        rejected = register_and_promote(candidate, cfg, client)
        retry = register_and_promote(candidate, cfg, client)
        assert not rejected["passed"] and retry["version"] == rejected["version"] == "3"
        assert str(champion_version(client, cfg["mlflow"]["registered_model_name"]).version) == "2"
        assert len(client.search_model_versions("name='isolated-credit-test'")) == 3
        assert (
            client.get_model_version(cfg["mlflow"]["registered_model_name"], "3").tags[
                "gate_status"
            ]
            == "rejected"
        )
        unfair = create(0.78)
        unfair["fairness_report"]["attributes"]["AGE_GROUP"]["disparate_impact"] = 0.6
        unfair_result = register_and_promote(unfair, cfg, client)
        assert not unfair_result["passed"] and unfair_result["champion_version"] == "2"
        assert "AGE_GROUP.disparate_impact" in unfair_result["failed_rules"]
    finally:
        mlflow.set_tracking_uri(previous_uri)
