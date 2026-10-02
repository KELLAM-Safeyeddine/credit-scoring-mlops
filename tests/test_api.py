"""Real isolated MLflow Registry, strict API payloads and coherent SHAP inference."""

import json
import pickle
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from api.main import create_app
from src.features.encoding import FeatureEncoder
from src.models.explain import create_shap_artifacts
from src.models.promote import register_and_promote
from src.serving.service import RegistryLoader, ScoringService
from tests.test_promote import valid_evidence

PAYLOAD = {
    "customer_id": "private-customer-42",
    "AMT_INCOME_TOTAL": 80000.0,
    "AMT_CREDIT": 200000.0,
    "AMT_ANNUITY": 12000.0,
    "EXT_SOURCE_1": 0.8,
}


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    """Synthetic accepted evidence tests plumbing, not credit-model qualification."""
    from src.config import load_config

    root = tmp_path_factory.mktemp("api-registry")
    uri = "sqlite:///" + (root / "tracking.db").as_posix()
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment = client.create_experiment(
        "api-fixture", artifact_location=(root / "artifacts").as_uri()
    )
    cfg = load_config()
    cfg["mlflow"]["registered_model_name"] = "api-synthetic-only"
    raw = pd.DataFrame(
        {
            "AMT_CREDIT": np.arange(100, 300, dtype=float),
            "AMT_INCOME_TOTAL": np.full(200, 100.0),
            "AMT_ANNUITY": np.full(200, 20.0),
            "EXT_SOURCE_1": np.linspace(0, 1, 200),
        }
    )
    encoder = FeatureEncoder().fit(raw)
    x = encoder.transform(raw)
    y = (raw.EXT_SOURCE_1 < 0.5).astype(int)
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(keep_empty_features=True)),
            ("model", LGBMClassifier(n_estimators=12, verbosity=-1, n_jobs=1)),
        ]
    ).fit(x, y)
    with mlflow.start_run(experiment_id=experiment):
        logged = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            metadata={"threshold": 0.37},
            input_example=x.head(2),
            pyfunc_predict_fn="predict_proba",
        )
        evidence = valid_evidence(cfg)
        evidence["metrics"]["threshold"] = 0.37
        evidence.update(
            run_id=mlflow.active_run().info.run_id,
            model_uri=logged.model_uri,
            training_data_sha256="synthetic-training",
            raw_dvc_hash="synthetic-raw",
        )
        mlflow.log_dict(encoder.columns, "feature_columns.json")
        path = root / "feature_encoder.pkl"
        path.write_bytes(pickle.dumps(encoder))
        mlflow.log_artifact(str(path), "features")
        shap_dir = root / "shap"
        create_shap_artifacts(
            {
                "pipeline": pipeline,
                "columns": encoder.columns,
                "threshold": 0.37,
                "run_id": evidence["run_id"],
            },
            x,
            shap_dir,
            30,
            42,
        )
        mlflow.log_artifacts(str(shap_dir), "shap")
    register_and_promote(evidence, cfg, client)
    yield uri, cfg["mlflow"]["registered_model_name"], client, evidence
    mlflow.set_tracking_uri(previous)


@pytest.fixture
def service(registry, tmp_path):
    uri, name, _, _ = registry
    return ScoringService(
        RegistryLoader(uri, name), tmp_path / "predictions.jsonl", "test-key-" * 8
    )


def test_predict_explain_logs_and_metadata(service):
    with TestClient(create_app(service, "reload-secret")) as client:
        predicted = client.post("/predict", json=PAYLOAD)
        explained = client.post("/explain", json=PAYLOAD)
        assert predicted.status_code == explained.status_code == 200
        prediction, explanation = predicted.json(), explained.json()
        assert prediction["default_score"] == explanation["default_score"]
        assert prediction["predicted_default"] == int(prediction["default_score"] >= 0.37)
        assert prediction["model_version"] == explanation["model_version"] == "1"
        assert len(explanation["reasons"]) == 3
        assert all(
            reason["label"] and reason["interpretation"] for reason in explanation["reasons"]
        )
        assert client.get("/health").status_code == 200
        info = client.get("/model-info").json()
        assert info["data_hash"] == "synthetic-training" and info["feature_count"] == 4
        assert client.post("/model/reload").status_code == 403
        assert (
            client.post("/model/reload", headers={"X-Reload-Token": "reload-secret"}).status_code
            == 200
        )
    text = service.log_path.read_text(encoding="utf-8")
    assert PAYLOAD["customer_id"] not in text
    events = [json.loads(line) for line in text.splitlines()]
    assert len(events) == 2
    assert events[0]["customer_pseudonym"] == events[1]["customer_pseudonym"]
    assert len(events[0]["customer_pseudonym"]) == 64
    assert set(events[0]["features"]) == set(service.snapshot.columns)
    assert events[0]["request_id"] != events[1]["request_id"]


@pytest.mark.parametrize(
    "changes",
    [
        {"AMT_CREDIT": -1},
        {"AMT_ANNUITY": 0},
        {"AMT_INCOME_TOTAL": "80000"},
        {"EXT_SOURCE_1": 1.1},
        {"CODE_GENDER": "M"},
        {"AGE_YEARS": 42},
        {"DAYS_BIRTH": -10000},
        {"EMPLOYED_AGE_RATIO": 0.1},
        {"TARGET": 1},
        {"DAYS_EMPLOYED": 4},
        {"history": {"unrecognized": 1}},
    ],
)
def test_invalid_payload_is_422(service, changes):
    with TestClient(create_app(service)) as client:
        assert client.post("/predict", json=PAYLOAD | changes).status_code == 422
    assert not service.log_path.exists()


def test_absent_champion_is_unavailable(tmp_path):
    loader = Mock()
    loader.load.side_effect = ValueError("no champion")
    service = ScoringService(loader, tmp_path / "log", "key" * 16)
    with TestClient(create_app(service)) as client:
        assert client.get("/health").status_code == 503
        assert client.get("/model-info").status_code == 503
        assert client.post("/predict", json=PAYLOAD).status_code == 503


def test_failed_reload_preserves_verified_snapshot(service):
    with TestClient(create_app(service, "token")) as client:
        snapshot = service.snapshot
        service.loader = Mock()
        service.loader.load.side_effect = RuntimeError("Registry offline")
        assert client.post("/model/reload", headers={"X-Reload-Token": "token"}).status_code == 503
        assert service.snapshot is snapshot
        assert client.post("/predict", json=PAYLOAD).status_code == 200


def test_rejected_alias_cannot_be_loaded(registry, service):
    _, name, client, _ = registry
    client.set_model_version_tag(name, "1", "gate_status", "rejected")
    try:
        with pytest.raises(ValueError, match="accepted champion"):
            service.loader.load()
    finally:
        client.set_model_version_tag(name, "1", "gate_status", "accepted")


def test_hmac_key_required(tmp_path):
    with pytest.raises(ValueError, match="API_LOG_HMAC_KEY"):
        ScoringService(Mock(), tmp_path / "log", "")


def test_missing_values_and_history_are_explicit(service):
    with TestClient(create_app(service)) as client:
        payload = {key: value for key, value in PAYLOAD.items() if key != "EXT_SOURCE_1"}
        payload["history"] = {"BUREAU_COUNT": 2.0}
        response = client.post("/explain", json=payload)
        assert response.status_code == 200
        assert "EXT_SOURCE_1" in response.json()["missing_inputs"]
    event = json.loads(service.log_path.read_text(encoding="utf-8").splitlines()[0])
    assert event["features"]["EXT_SOURCE_1"] is None


@pytest.mark.parametrize("nonfinite", ["NaN", "Infinity"])
def test_nonfinite_payload_returns_serializable_422(service, nonfinite):
    with TestClient(create_app(service)) as client:
        content = json.dumps(PAYLOAD).replace(
            '"AMT_CREDIT": 200000.0', '"AMT_CREDIT": ' + nonfinite
        )
        response = client.post(
            "/predict", content=content, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        assert PAYLOAD["customer_id"] not in response.text


def test_online_ratios_match_offline_and_keep_missingness(service):
    from src.features.build_features import add_ratios

    assert service.reload()
    snapshot = service.snapshot
    raw = pd.DataFrame(
        [
            {
                **PAYLOAD,
                "DAYS_EMPLOYED": -500,
                "DAYS_BIRTH": -15000,
                "EXT_SOURCE_2": None,
                "EXT_SOURCE_3": 0.4,
            }
        ]
    )
    expected_raw = add_ratios(raw.drop(columns="customer_id"))
    expected_raw["EMPLOYED_ANOMALY"] = 0
    expected_raw["UNAVAILABLE_NUMERIC"] = np.nan
    expected_raw["UNAVAILABLE_CATEGORY"] = pd.Series([None], dtype="string")
    encoder = FeatureEncoder().fit(expected_raw)
    snapshot = replace(snapshot, encoder=encoder, columns=encoder.columns)
    actual, missing = snapshot.transform(raw.drop(columns="customer_id").iloc[0].to_dict())
    pd.testing.assert_frame_equal(actual, encoder.transform(expected_raw))
    assert {"EXT_SOURCE_2", "UNAVAILABLE_NUMERIC", "UNAVAILABLE_CATEGORY"}.issubset(missing)
    assert not {"DAYS_BIRTH", "EMPLOYED_AGE_RATIO"}.intersection(actual.columns)


def test_accepted_version_change_is_reloaded(registry, service, tmp_path):
    uri, name, client, evidence = registry
    previous_uri = mlflow.get_tracking_uri()
    with TestClient(create_app(service, "token")) as api:
        first = service.snapshot
        mlflow.set_tracking_uri(uri)
        try:
            experiment = client.get_run(first.run_id).info.experiment_id
            with mlflow.start_run(experiment_id=experiment):
                logged = mlflow.sklearn.log_model(
                    first.pipeline, name="model", metadata={"threshold": 0.6}
                )
                new_evidence = json.loads(json.dumps(evidence))
                new_evidence.update(
                    run_id=mlflow.active_run().info.run_id, model_uri=logged.model_uri
                )
                new_evidence["metrics"]["threshold"] = 0.6
                mlflow.log_dict(first.columns, "feature_columns.json")
                encoder_path = tmp_path / "feature_encoder.pkl"
                encoder_path.write_bytes(pickle.dumps(first.encoder))
                mlflow.log_artifact(str(encoder_path), "features")
                original = client.download_artifacts(first.run_id, "shap")
                mlflow.log_artifacts(original, "shap")
                report_path = client.download_artifacts(first.run_id, "shap/shap_report.json")
                report = json.loads(Path(report_path).read_text())
                report["model_run_id"] = new_evidence["run_id"]
                mlflow.log_dict(report, "shap/shap_report.json")
            from src.config import load_config

            cfg = load_config()
            cfg["mlflow"]["registered_model_name"] = name
            result = register_and_promote(new_evidence, cfg, client)
            assert result["passed"] and result["version"] == "2"
            assert api.get("/model-info").json()["version"] == "1"
            assert api.post("/model/reload", headers={"X-Reload-Token": "token"}).status_code == 200
            predicted = api.post("/predict", json=PAYLOAD).json()
            explained = api.post("/explain", json=PAYLOAD).json()
            assert predicted["model_version"] == explained["model_version"] == "2"
            assert predicted["threshold"] == 0.6
            assert predicted["default_score"] == explained["default_score"]
        finally:
            client.set_registered_model_alias(name, "champion", "1")
            mlflow.set_tracking_uri(previous_uri)
