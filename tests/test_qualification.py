"""Qualification search must preserve gates and reject unsafe search definitions."""

import json

import pandas as pd
import pytest

from src.models.qualify import eligible, search, specifications


@pytest.mark.parametrize("change", ["budget", "parameter", "nonfinite_power"])
def test_search_rejects_unsafe_configuration(config, change):
    if change == "budget":
        config["qualification"]["max_trials"] = 1
    elif change == "parameter":
        config["qualification"]["configurations"][0]["class_weight"] = None
    else:
        config["qualification"]["reweighing_powers"] = [float("nan")]
    with pytest.raises(ValueError):
        specifications(config)


@pytest.mark.parametrize("failure", ["auc", "fairness", "recall", "acceptance"])
def test_screening_rejects_each_gate_failure(config, failure):
    metrics = {
        "validation_auc": 0.76,
        "validation_recall": 0.6,
        "validation_acceptance_rate": 0.7,
    }
    fairness = {"passes_thresholds": True}
    assert eligible(metrics, fairness, config)
    if failure == "auc":
        metrics["validation_auc"] = 0.739
    elif failure == "fairness":
        fairness["passes_thresholds"] = False
    elif failure == "recall":
        metrics["validation_recall"] = 0.49
    else:
        metrics["validation_acceptance_rate"] = 0.99
    assert not eligible(metrics, fairness, config)


def test_search_rejects_overlapping_splits_before_fitting(config, tmp_path, monkeypatch):
    """No reserved test files are required or read by qualification."""
    (tmp_path / "feature_columns.json").write_text(json.dumps(["AMT_CREDIT"]))
    index = pd.Index([1, 2], name="SK_ID_CURR")
    seen = []

    def read_frame(path, **kwargs):
        seen.append(path.stem)
        assert "test" not in path.stem
        if path.stem.startswith("X_"):
            return pd.DataFrame({"AMT_CREDIT": [100, 200]}, index=index)
        if path.stem.startswith("y_"):
            return pd.DataFrame({"TARGET": [0, 1]}, index=index)
        return pd.DataFrame({"CODE_GENDER": ["F", "M"]}, index=index)

    monkeypatch.setattr("src.models.qualify.project_path", lambda *args: tmp_path)
    monkeypatch.setattr("src.models.qualify.pd.read_csv", read_frame)
    with pytest.raises(ValueError, match="disjoint"):
        search(config, tmp_path / "output")
    assert len(seen) == 6
    assert not (tmp_path / "output").exists()
