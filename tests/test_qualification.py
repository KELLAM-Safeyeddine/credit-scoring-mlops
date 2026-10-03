"""Qualification search must preserve gates and reject unsafe search definitions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from src.models.qualify import eligible, search, specifications, training_weights


def test_age_weights_ignore_gender_but_intersection_weights_do_not():
    y = pd.Series([0, 0, 0, 1, 0, 1, 1, 1])
    sensitive = pd.DataFrame(
        {
            "AGE_GROUP": ["<30"] * 4 + ["60+"] * 4,
            "CODE_GENDER": ["F", "F", "M", "M", "F", "F", "M", "M"],
        }
    )
    original = training_weights(y, sensitive, ("AGE_GROUP",))
    changed = sensitive.copy()
    changed.CODE_GENDER = ["M", "M", "M", "F", "M", "F", "F", "F"]
    np.testing.assert_allclose(original, training_weights(y, changed, ("AGE_GROUP",)))
    assert not np.allclose(
        training_weights(y, sensitive, ("CODE_GENDER", "AGE_GROUP")),
        training_weights(y, changed, ("CODE_GENDER", "AGE_GROUP")),
    )
    assert original.mean() == pytest.approx(1)


def test_reweighing_cannot_use_target_or_missing_audit_groups(config):
    config["qualification"]["reweighing_scopes"] = [["TARGET"]]
    with pytest.raises(ValueError, match="scopes"):
        specifications(config)
    with pytest.raises(ValueError, match="non-null"):
        training_weights(
            pd.Series([0, 1]), pd.DataFrame({"AGE_GROUP": ["<30", None]}), ("AGE_GROUP",)
        )


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


def test_expansion_provenance_survives_multiple_runs(config, tmp_path, monkeypatch):
    """A temporary lock filename must not overwrite the expanded snapshot directory."""
    config["qualification"].update(
        snapshot_root=str(tmp_path),
        max_trials=2,
        reweighing_powers=[1.0, 1.25],
        configurations=[config["qualification"]["configurations"][0]],
    )
    config["mlflow"]["tracking_uri"] = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    for split, start in (("train", 1), ("val", 5)):
        index = pd.Index(range(start, start + 4), name="SK_ID_CURR")
        for name, frame in (
            ("X", pd.DataFrame({"AMT_CREDIT": [100, 200, 300, 400]}, index=index)),
            ("y", pd.DataFrame({"TARGET": [0, 1, 0, 1]}, index=index)),
            (
                "sensitive",
                pd.DataFrame(
                    {
                        "CODE_GENDER": ["F", "M", "F", "M"],
                        "AGE_GROUP": ["<30", "30-45", "45-60", "60+"],
                    },
                    index=index,
                ),
            ),
        ):
            frame.to_csv(tmp_path / f"{name}_{split}.csv")
    (tmp_path / "feature_columns.json").write_text('["AMT_CREDIT"]')
    (tmp_path / "feature_encoder.pkl").write_bytes(b"unused fixture artifact")
    for name in ("processed.dvc", "interim.dvc"):
        (tmp_path / name).write_text("outs:\n- md5: fixture.dir\n")
    (tmp_path / "split_manifest.json").write_text("{}")

    class FixturePipeline:
        def fit(self, *args, **kwargs):
            return self

        def predict_proba(self, frame):
            scores = np.array([0.1, 0.8, 0.2, 0.7])
            return np.column_stack([1 - scores, scores])

    monkeypatch.setattr("src.models.qualify.project_path", lambda *args: tmp_path)
    monkeypatch.setattr("src.models.qualify.ROOT", tmp_path)
    monkeypatch.setattr("src.models.qualify.build_pipeline", lambda *args: FixturePipeline())
    monkeypatch.setattr("src.models.qualify.data_provenance", lambda *args: ({}, b"fixture lock"))
    monkeypatch.setattr("src.models.qualify.select_fair_threshold", lambda *args: (0.5, False))
    result = search(config, tmp_path / "output")
    assert result["status"] == "no_eligible_candidate"
    assert result["test_split"] == "untouched"
    assert len(result["trials"]) == 2
    client = MlflowClient(config["mlflow"]["tracking_uri"])
    for trial in result["trials"]:
        run = client.get_run(trial["run_id"])
        assert run.info.status == "FINISHED"
        assert run.data.tags["qualification_processed_dvc_hash"] == "fixture.dir"
        names = {
            Path(item.path).name
            for item in client.list_artifacts(trial["run_id"], "provenance/expansion")
        }
        assert names == {"processed.dvc", "interim.dvc", "split_manifest.json"}
    ledger = tmp_path / "runtime/qualification-holdout.json"
    ledger.parent.mkdir()
    ledger.write_text("{}")
    later = search(config, tmp_path / "later-output")
    assert later["test_split"] == "previously_evaluated_not_read_by_search"
