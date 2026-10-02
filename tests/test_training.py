"""Ranking metrics, train-only preprocessing and actual MLflow round-trip."""

import hashlib

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from mlflow.models import infer_signature

from src.models.evaluate import performance_metrics
from src.models.experiments import build_pipeline, data_provenance
from src.models.train import select_threshold


def test_perfect_and_reversed_rankings(config):
    y = pd.Series([0, 0, 1, 1])
    good = performance_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5, config["train"]["threshold"])
    assert good["validation_auc"] == good["validation_pr_auc"] == 1
    assert good["validation_ks"] == good["validation_gini"] == 1
    assert good["validation_business_cost"] == 0
    bad = performance_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]), 0.5, config["train"]["threshold"])
    assert bad["validation_auc"] == 0
    assert bad["validation_gini"] == -1
    assert bad["validation_business_cost"] == 22


def test_business_cost_selects_low_refusal_threshold(config):
    config["train"]["threshold"].update(min=0.1, max=0.9, steps=9)
    y = pd.Series([0, 0, 1, 1])
    scores = np.array([0.1, 0.3, 0.2, 0.4])
    threshold = select_threshold(y, scores, config)
    assert threshold == pytest.approx(0.2)
    metrics = performance_metrics(y, scores, threshold, config["train"]["threshold"])
    assert metrics["validation_fn"] == 0
    assert metrics["validation_fp"] == 1


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "xgboost", "lightgbm"])
def test_family_preprocessing_fits_only_train(name, config):
    for family in ("random_forest", "xgboost", "lightgbm"):
        config["train"][family]["n_estimators"] = 4
    train = pd.DataFrame(
        {"income": [1.0, 2.0, 4.0, 9.0, np.nan, 12.0], "score": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]}
    )
    validation = pd.DataFrame({"income": [10000.0, np.nan], "score": [0.0, 1.0]})
    pipeline = build_pipeline(name, config).fit(train, [0, 1, 0, 1, 0, 1])
    assert pipeline["imputer"].statistics_[0] == 4
    probabilities = pipeline.predict_proba(validation)
    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert np.isfinite(probabilities).all()


def test_provenance_matches_lock_and_changes_with_input(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "dvc.lock").write_bytes(b"schema: '2.0'\n")
    (tmp_path / "data/raw.dvc").write_text("outs:\n- md5: abc.dir\n", encoding="utf-8")
    processed = tmp_path / "processed"
    processed.mkdir()
    names = ("X_train.csv", "y_train.csv", "X_val.csv", "y_val.csv", "feature_columns.json")
    for name in names:
        (processed / name).write_bytes(b"fixture")
    tags, lock = data_provenance(tmp_path, processed)
    assert tags["dvc_lock_sha256"] == hashlib.sha256(lock).hexdigest()
    (processed / "X_val.csv").write_bytes(b"changed")
    changed, _ = data_provenance(tmp_path, processed)
    assert tags["training_data_sha256"] != changed["training_data_sha256"]


def test_mlflow_310_model_probability_round_trip(tmp_path, config):
    previous = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}")
        experiment_id = mlflow.create_experiment(
            "round-trip", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        x = pd.DataFrame({"income": [1.0, 2.0, 3.0, 4.0], "score": [0.0, 0.0, 1.0, 1.0]})
        pipeline = build_pipeline("logistic_regression", config).fit(x, [0, 0, 1, 1])
        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.set_tag("dvc_lock_sha256", "verified-fixture-hash")
            mlflow.log_metric("threshold", 0.2)
            info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                pyfunc_predict_fn="predict_proba",
                signature=infer_signature(x, pipeline.predict_proba(x)),
                input_example=x,
                pip_requirements=["scikit-learn==1.7.2"],
            )
        loaded = mlflow.sklearn.load_model(info.model_uri)
        np.testing.assert_allclose(loaded.predict_proba(x), pipeline.predict_proba(x))
        pyfunc = mlflow.pyfunc.load_model(info.model_uri)
        np.testing.assert_allclose(pyfunc.predict(x), pipeline.predict_proba(x))
        recorded = mlflow.MlflowClient().get_run(run.info.run_id)
        assert recorded.data.metrics["threshold"] == 0.2
        assert recorded.data.tags["dvc_lock_sha256"] == "verified-fixture-hash"
    finally:
        mlflow.set_tracking_uri(previous)
