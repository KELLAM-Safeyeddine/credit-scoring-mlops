"""Packaging must reject a changed selection before loading evaluation data."""

import copy
import hashlib
import json
import pickle

import pytest

from src.models import prepare_ensemble


@pytest.mark.parametrize("violation", ["test_evaluated", "different_run", "no_selection"])
def test_packaging_rejects_invalid_development_selection(tmp_path, monkeypatch, violation):
    snapshot = pickle.dumps({"mode": "probability_average", "run_id": "selected-run"})
    (tmp_path / "selected.pkl").write_bytes(snapshot)
    report = {
        "development_only": True,
        "test_evaluated": False,
        "selected_candidate": {
            "run_id": "selected-run",
            "snapshot": "selected.pkl",
            "mode": "probability_average",
        },
    }
    if violation == "test_evaluated":
        report["test_evaluated"] = True
    elif violation == "different_run":
        report["selected_candidate"]["run_id"] = "another-run"
    else:
        report["selected_candidate"] = None
    (tmp_path / "report.json").write_text(json.dumps(report))
    settings = {
        "source_params": "source.yaml",
        "snapshot": "selected.pkl",
        "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
        "study_report": "report.json",
        "output": "must-not-exist",
    }
    monkeypatch.setattr(prepare_ensemble, "ROOT", tmp_path)
    monkeypatch.setattr(
        prepare_ensemble,
        "load_config",
        lambda path=None: {"custom": settings} if path is None else {},
    )
    monkeypatch.setattr(
        prepare_ensemble, "protocol_context", lambda cfg: (tmp_path / "unopened.json", {})
    )
    with pytest.raises(ValueError, match="selected unopened development"):
        prepare_ensemble.prepare("custom")
    assert not (tmp_path / "must-not-exist").exists()


def test_validation_screen_keeps_real_promotion_blocked(config):
    from src.models.promote import SHAP_FILES, quality_gate

    config["qualification"]["require_independent_test"] = True
    original_cfg = copy.deepcopy(config)
    evidence = {
        "independent_test_required": True,
        "metrics": {
            "validation_auc": 0.76,
            "validation_recall": 0.7,
            "validation_acceptance_rate": 0.6,
            "requires_sensitive_at_inference": False,
            "randomized_decision": False,
        },
        "shap_verified": True,
        "shap_files": sorted(SHAP_FILES),
        "fairness_report": {
            "attributes": {
                attr: {
                    "sufficient_support": True,
                    "eligible_groups": config["fairness"]["required_groups"][attr],
                    "disparate_impact": 0.9,
                    "equal_opportunity_difference": 0.05,
                }
                for attr in config["fairness"]["sensitive_features"]
            }
        },
    }
    original_evidence = copy.deepcopy(evidence)
    assert prepare_ensemble.validation_shap_screen(evidence, config)[0]
    assert config == original_cfg and evidence == original_evidence
    passed, rules = quality_gate(evidence, None, config)
    assert not passed and not rules["independent_test"]["passed"]
    evidence["metrics"]["validation_auc"] = 0.73
    assert not prepare_ensemble.validation_shap_screen(evidence, config)[0]
