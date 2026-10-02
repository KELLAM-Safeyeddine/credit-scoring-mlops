"""Check isolated workspaces, delayed labels and actual Evidently reports."""

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.monitoring.drift_report import build_drift_report
from src.monitoring.fairness_monitor import monitor_fairness
from src.orchestration import monitoring, worker


def test_workspace_isolation_and_stricter_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "RUNTIME", tmp_path / "runs")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    first = worker.create_workspace("training:first", {"min_auc": 1.0})
    second = worker.create_workspace("training:second", {})
    assert first != second
    cfg = worker.load_config(first / "params.yaml")
    assert cfg["quality_gate"]["min_auc"] == 1
    assert cfg["mlflow"]["tracking_uri"] == "http://mlflow:5000"
    assert all(str(first) in value for key, value in cfg["paths"].items() if key != "raw")
    assert worker.create_workspace("training:first", {"min_auc": 1.0}) == first
    with pytest.raises(ValueError, match="different"):
        worker.create_workspace("training:first", {})
    with pytest.raises(ValueError, match="min_auc"):
        worker.create_workspace("unsafe", {"min_auc": 0.1})
    with pytest.raises(ValueError, match="Invalid"):
        worker.checked_workspace(str(tmp_path))


def test_evidently_snapshot_detects_shift_and_checks_columns(tmp_path, config):
    rng = np.random.default_rng(123)
    reference = pd.DataFrame({"income": rng.normal(10, 1, 500)})
    scores = pd.Series(rng.uniform(0.1, 0.4, 500))
    same = build_drift_report(
        reference, reference, scores, scores, tmp_path / "same", config["monitoring"]
    )
    assert same["drift_feature_share"] == 0
    assert same["score_psi"] == 0
    shifted = build_drift_report(
        reference, reference + 10, scores, scores + 0.5, tmp_path / "shift", config["monitoring"]
    )
    assert shifted["drift_feature_share"] == 1
    assert shifted["score_psi"] > 0.2
    assert (tmp_path / "shift/drift.html").stat().st_size > 1000
    assert json.loads((tmp_path / "shift/evidently.json").read_text())["metrics"]
    with pytest.raises(ValueError, match="columns"):
        build_drift_report(
            reference,
            reference.rename(columns={"income": "other"}),
            scores,
            scores,
            tmp_path,
            config["monitoring"],
        )


def test_unlabelled_fairness_uses_acceptance_not_default(config):
    settings = config["fairness"]
    settings["sensitive_features"] = ["CODE_GENDER"]
    sensitive = pd.DataFrame({"CODE_GENDER": ["F"] * 4 + ["M"] * 4})
    result = monitor_fairness(pd.Series([0, 0, 0, 1, 0, 1, 1, 1]), sensitive, settings)
    assert result["attributes"]["CODE_GENDER"]["disparate_impact"] == pytest.approx(1 / 3)
    assert result["labels_available"] is False
    assert "outcome_audit" not in result


def test_ground_truth_not_read_before_release(tmp_path, config, monkeypatch):
    config["paths"] = {key: str(tmp_path / key) for key in config["paths"]}
    monkeypatch.setattr(monitoring, "load_config", lambda: config)
    production, processed, models = [
        tmp_path / name for name in ("production", "processed", "models")
    ]
    for directory in (production / "ground_truth", processed, models):
        directory.mkdir(parents=True)
    x = pd.DataFrame({"income": [1.0, 2.0]}, index=pd.Index([10, 11], name="SK_ID_CURR"))
    pipeline = DummyClassifier(strategy="prior").fit(x, [0, 1])
    (models / "baseline.pkl").write_bytes(
        pickle.dumps({"columns": ["income"], "pipeline": pipeline})
    )
    x.to_csv(production / "X_01.csv")
    x.to_csv(processed / "X_val.csv")
    pd.DataFrame(
        {"predicted_default": [0, 1], "probability_default": [0.5, 0.5]}, index=x.index
    ).to_csv(production / "predictions_01.csv")
    pd.DataFrame({"CODE_GENDER": ["F", "M"], "AGE_GROUP": ["<30", "60+"]}, index=x.index).to_csv(
        production / "sensitive_01.csv"
    )
    (production / "manifest.json").write_text(
        json.dumps({"batches": [{"batch_id": "01", "labels_available_at_batch": 3}]})
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monitoring.load_batch(workspace, {"batch_id": "01"})
    # No label file exists: an early read would fail instead of silently leaking it.
    assert monitoring.frames(workspace)[6] is None
    pd.DataFrame({"TARGET": [0, 1]}, index=x.index).to_csv(production / "ground_truth/y_01.csv")
    monitoring.load_batch(workspace, {"batch_id": "01", "observed_batch": 3})
    assert monitoring.frames(workspace)[6].tolist() == [0, 1]
    with pytest.raises(ValueError, match="Unknown"):
        monitoring.load_batch(workspace, {"batch_id": "../../secret"})
