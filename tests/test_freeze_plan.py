"""Freezing a bounded plan must preserve IDs and refuse invalid or overwritten plans."""

import copy
import hashlib
import json

import pytest
import yaml

from src.data import freeze_protocol as freezing


def test_custom_plan_is_frozen_before_training(tmp_path, monkeypatch, applications, config):
    monkeypatch.setattr(freezing, "ROOT", tmp_path)
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    applications.to_csv(raw / "application_train.csv", index=False)
    (tmp_path / "data/raw.dvc").write_text("outs:\n- md5: fake-raw-version\n")
    history = tmp_path / "history"
    history.mkdir()
    for index, name in enumerate(freezing.SPLITS):
        applications.iloc[index * 30 : (index + 1) * 30][["SK_ID_CURR"]].to_csv(
            history / f"{name}.csv", index=False
        )
    plan = {"max_trials": 2, "selection_rule": "fixed two-trial margin selection"}
    config["independent_protocol"] = {
        "id": "prospective-custom",
        "directory": "runtime/prospective-custom",
        "source_fraction": 0.3,
        "random_state": 17,
        "used_membership_roots": ["history"],
        "qualification_plan": plan,
    }
    manifest = freezing.freeze_protocol(config)
    output = tmp_path / "runtime/prospective-custom"
    frozen = yaml.safe_load((output / "params.yaml").read_text())
    assert manifest["selection_budget"] == 2
    assert manifest["previously_used_rows"] == 120
    assert manifest["overlap_with_historical"] == 0
    assert sum(manifest["rows"].values()) == 180
    assert frozen["qualification"]["require_independent_test"] is True
    assert frozen["qualification"]["selection_rule"] == plan["selection_rule"]
    assert frozen["quality_gate"] == config["quality_gate"]
    assert (
        manifest["prospective_config_sha256"]
        == hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    )
    content = (output / "protocol.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        freezing.freeze_protocol(config)
    assert (output / "protocol.json").read_bytes() == content


@pytest.mark.parametrize("budget", [0, 5, True, "2"])
def test_invalid_budget_writes_nothing(tmp_path, monkeypatch, config, budget):
    monkeypatch.setattr(freezing, "ROOT", tmp_path)
    config = copy.deepcopy(config)
    config["independent_protocol"]["directory"] = "rejected-plan"
    config["independent_protocol"]["qualification_plan"] = {
        "max_trials": budget,
        "selection_rule": "fixed",
    }
    with pytest.raises(ValueError, match="qualification plan"):
        freezing.freeze_protocol(config)
    assert not (tmp_path / "rejected-plan").exists()
