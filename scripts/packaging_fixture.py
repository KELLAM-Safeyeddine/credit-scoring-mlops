"""Build a synthetic champion in a disposable container-only SQLite Registry."""

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.features.encoding import FeatureEncoder
from src.models.evaluate import performance_metrics
from src.models.explain import create_shap_artifacts
from src.models.promote import SHAP_FILES, register_and_promote


def main() -> None:
    """Test packaging with actual perfect synthetic predictions and balanced groups."""
    if Path(sys.prefix).resolve() != Path("/opt/creditS").resolve():
        raise SystemExit("Run only inside the disposable API-image validation container.")
    root = Path("/app/runtime/api/checks")
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cfg["mlflow"]["registered_model_name"] = "packaging-fixture"
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "sqlite:////app/runtime/api/checks/tracking.db")
    )
    client = mlflow.MlflowClient()
    experiment = client.create_experiment("synthetic-packaging-only")
    index = np.arange(1600)
    labels = pd.Series(index % 2, name="TARGET")
    rng = np.random.default_rng(42)
    raw = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": rng.uniform(40000, 100000, len(index)),
            "AMT_CREDIT": rng.uniform(100000, 400000, len(index)),
            "AMT_ANNUITY": rng.uniform(5000, 15000, len(index)),
            "EXT_SOURCE_1": np.where(labels == 1, 0.1, 0.9),
        }
    )
    encoder = FeatureEncoder().fit(raw.iloc[:800])
    train, val = encoder.transform(raw.iloc[:800]), encoder.transform(raw.iloc[800:])
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(keep_empty_features=True)),
            ("model", LGBMClassifier(n_estimators=30, verbosity=-1, n_jobs=1)),
        ]
    )
    pipeline.fit(train, labels.iloc[:800])
    threshold = 0.37
    probabilities = pipeline.predict_proba(val)[:, 1]
    predicted = (probabilities >= threshold).astype(int)
    np.testing.assert_array_equal(predicted, labels.iloc[800:])
    metrics = performance_metrics(
        labels.iloc[800:], probabilities, threshold, cfg["train"]["threshold"]
    )
    metrics.update(
        threshold=threshold, requires_sensitive_at_inference=False, randomized_decision=False
    )
    attributes = {}
    for attr, values in (
        ("CODE_GENDER", np.where((index[800:] // 2) % 2, "M", "F")),
        ("AGE_GROUP", np.array(["<30", "30-45", "45-60", "60+"])[(index[800:] // 4) % 4]),
    ):
        # Every group has equal class support and perfect decisions by construction.
        required = cfg["fairness"]["required_groups"][attr]
        for group in required:
            group_labels = labels.iloc[800:].to_numpy()[values == group]
            assert len(group_labels) >= cfg["fairness"]["min_group_size"]
            assert min(np.bincount(group_labels)) >= cfg["fairness"]["min_class_count"]
            assert np.mean(group_labels) == 0.5
        attributes[attr] = {
            "sufficient_support": True,
            "eligible_groups": required,
            "disparate_impact": 1.0,
            "equal_opportunity_difference": 0.0,
        }
    with mlflow.start_run(experiment_id=experiment) as run:
        logged = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            metadata={"threshold": threshold},
            input_example=train.head(2),
            pyfunc_predict_fn="predict_proba",
        )
        mlflow.log_dict(encoder.columns, "feature_columns.json")
        path = root / "feature_encoder.pkl"
        path.write_bytes(pickle.dumps(encoder))
        mlflow.log_artifact(str(path), "features")
        create_shap_artifacts(
            {
                "pipeline": pipeline,
                "columns": encoder.columns,
                "threshold": threshold,
                "run_id": run.info.run_id,
            },
            val,
            root / "shap",
            40,
            42,
        )
        mlflow.log_artifacts(str(root / "shap"), "shap")
        evidence = {
            "run_id": run.info.run_id,
            "model_uri": logged.model_uri,
            "metrics": metrics,
            "fairness_report": {"attributes": attributes},
            "shap_verified": True,
            "shap_files": sorted(SHAP_FILES),
            "training_data_sha256": hashlib.sha256(train.to_csv().encode()).hexdigest(),
            "evaluation_data_sha256": hashlib.sha256(val.to_csv().encode()).hexdigest(),
            "raw_dvc_hash": "synthetic-packaging-fixture-not-home-credit",
        }
    result = register_and_promote(evidence, cfg, client)
    assert result["passed"] and result["champion_version"] == "1"
    print(json.dumps({"registry": "isolated-synthetic", "version": "1", "gate_passed": True}))


if __name__ == "__main__":
    main()
