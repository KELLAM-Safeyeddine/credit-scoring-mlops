"""Business operations for the isolated Airflow lifecycle demonstration."""

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import urllib.request
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.config import ROOT, load_config, write_json
from src.features.encoding import FeatureEncoder
from src.models.evaluate import performance_metrics
from src.models.explain import create_shap_artifacts
from src.models.fairness import audit_fairness, intersection_groups, reweighing_weights
from src.models.promote import candidate_evidence, register_and_promote
from src.monitoring.alerts import evaluate_alerts
from src.monitoring.demo_data import cohort, training_cohort
from src.monitoring.drift_report import build_drift_report
from src.monitoring.fairness_monitor import monitor_fairness


def context(workspace: Path) -> tuple[dict, dict]:
    """Restrict the demonstration to its own private namespace and Registry name."""
    if workspace.resolve().parent != (ROOT / "runtime/demo").resolve():
        raise ValueError("Invalid demo workspace.")
    value = json.loads((workspace / "context.json").read_text())
    if value["mode"] != "synthetic" or value["name"] != "credit-demo-" + workspace.name:
        raise ValueError("Unsafe demo Registry namespace.")
    cfg = load_config()
    cfg["mlflow"]["registered_model_name"] = value["name"]
    cfg["mlflow"]["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    return value, cfg


def bootstrap(workspace: Path) -> dict:
    """Version generated fixtures with DVC, including a separate local remote."""
    value, _ = context(workspace)
    directory = workspace / "fixture"
    directory.mkdir(exist_ok=True)
    for split, frames in (
        ("train_v1", training_cohort(1)),
        ("train_biased", training_cohort(2)),
        ("val", cohort(123)),
        ("production", cohort(456)),
    ):
        for key, frame in zip(("X", "y", "sensitive"), frames, strict=True):
            frame.to_csv(directory / f"{key}_{split}.csv", index_label="SK_ID_CURR")
    commands = [["init", "--no-scm"], ["add", "fixture"]]
    if (workspace / ".dvc").exists():
        commands = [["add", "fixture"]]
    for command in commands:
        subprocess.run([sys.executable, "-m", "dvc", *command], cwd=workspace, check=True)
    subprocess.run(
        [sys.executable, "-m", "dvc", "remote", "add", "-f", "-d", "demo", "dvc-storage"],
        cwd=workspace,
        check=True,
    )
    subprocess.run([sys.executable, "-m", "dvc", "push"], cwd=workspace, check=True)
    descriptor = (workspace / "fixture.dvc").read_text()
    digest = yaml.safe_load(descriptor)["outs"][0]["md5"]
    write_json(workspace / "data.json", {"dvc_hash": digest, "mode": value["mode"]})
    return {"dvc_hash": digest, "mode": "synthetic"}


def validate_fixture(workspace: Path) -> dict:
    """Check the local DVC descriptor and cohort contract inside Airflow."""
    context(workspace)
    digest = yaml.safe_load((workspace / "fixture.dvc").read_text())["outs"][0]["md5"]
    if digest != json.loads((workspace / "data.json").read_text())["dvc_hash"]:
        raise ValueError("Fixture DVC provenance differs.")
    for split in ("train_v1", "train_biased", "val", "production"):
        read_split(workspace, split)
    return {"dvc_hash": digest, "mode": "synthetic"}


def read_split(workspace: Path, split: str) -> tuple:
    """Load feature, target and audit tables with identical indexes."""
    frames = [
        pd.read_csv(workspace / "fixture" / f"{key}_{split}.csv", index_col="SK_ID_CURR")
        for key in ("X", "y", "sensitive")
    ]
    if not all(frames[0].index.equals(frame.index) for frame in frames):
        raise ValueError("Demo split alignment failed.")
    return frames[0], frames[1].TARGET, frames[2]


def train(workspace: Path, version: int) -> dict:
    """Train, measure Fairlearn metrics, verify SHAP, then apply the unchanged gate."""
    value, cfg = context(workspace)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(value["name"])
    experiment_id = (
        experiment.experiment_id if experiment else client.create_experiment(value["name"])
    )
    raw, y, sensitive = read_split(workspace, "train_v1" if version == 1 else "train_biased")
    raw_val, y_val, s_val = read_split(workspace, "val")
    encoder = FeatureEncoder().fit(raw)
    x, val = encoder.transform(raw), encoder.transform(raw_val)
    weights = reweighing_weights(y, intersection_groups(sensitive)) if version == 3 else None
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(keep_empty_features=True)),
            (
                "model",
                LGBMClassifier(**cfg["demo"]["model"], verbosity=-1),
            ),
        ]
    )
    pipeline.fit(x, y, model__sample_weight=weights)
    threshold = cfg["demo"]["threshold"]  # Never tuned on production outcomes.
    scores = pipeline.predict_proba(val)[:, 1]
    decisions = (scores >= threshold).astype(int)
    fairness = audit_fairness(y_val, decisions, s_val, cfg["fairness"], scores)
    metrics = performance_metrics(y_val, scores, threshold, cfg["train"]["threshold"])
    metrics.update(
        threshold=threshold, requires_sensitive_at_inference=False, randomized_decision=False
    )
    digest = json.loads((workspace / "data.json").read_text())["dvc_hash"]
    output = workspace / f"v{version}"
    output.mkdir(exist_ok=True)
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"synthetic-v{version}") as run:
        mlflow.set_tags(
            {
                "mode": "synthetic-demonstration-not-home-credit",
                "raw_dvc_hash": digest,
                "dvc_lock_sha256": hashlib.sha256(
                    (workspace / "fixture.dvc").read_bytes()
                ).hexdigest(),
                "data_sha256.X_val.csv": hashlib.sha256(raw_val.to_csv().encode()).hexdigest(),
                "data_sha256.y_val.csv": hashlib.sha256(y_val.to_csv().encode()).hexdigest(),
                "training_data_sha256": hashlib.sha256(raw.to_csv().encode()).hexdigest(),
            }
        )
        mlflow.log_params(
            {"demo_version": version, "mitigation": "reweighing" if version == 3 else "none"}
        )
        mlflow.log_metrics(
            {key: item for key, item in metrics.items() if isinstance(item, (int, float))}
        )
        mlflow.log_dict(fairness, "fairness_report.json")
        mlflow.log_artifact(str(workspace / "fixture.dvc"), "provenance")
        mlflow.log_dict(encoder.columns, "feature_columns.json")
        (output / "feature_encoder.pkl").write_bytes(pickle.dumps(encoder))
        mlflow.log_artifact(str(output / "feature_encoder.pkl"), "features")
        logged = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=x.head(2),
            metadata={"threshold": threshold},
            pyfunc_predict_fn="predict_proba",
        )
        bundle = {
            "pipeline": pipeline,
            "columns": encoder.columns,
            "threshold": threshold,
            "run_id": run.info.run_id,
            "model_uri": logged.model_uri,
            "metrics": metrics,
            "fairness_report": fairness,
        }
        create_shap_artifacts(bundle, val, output / "shap", 40, 42)
        mlflow.log_artifacts(str(output / "shap"), "shap")
    (output / "bundle.pkl").write_bytes(pickle.dumps(bundle))
    evidence = candidate_evidence(bundle, client)
    result = register_and_promote(evidence, cfg, client, expected_pass=version != 2)
    write_json(output / "gate.json", result)
    if version == 2:
        first = json.loads((workspace / "v1/gate.json").read_text())
        assert (
            metrics["validation_auc"]
            > client.get_run(first["run_id"]).data.metrics["validation_auc"]
        )
        assert any("CODE_GENDER" in rule for rule in result["failed_rules"])
        assert result["champion_version"] == first["version"]
    return {
        "version": result["version"],
        "gate_status": result["gate_status"],
        "auc": metrics["validation_auc"],
        "failed_rules": result["failed_rules"],
    }


def reload_api(workspace: Path, version: int) -> dict:
    """Reload and exercise the running Docker API against the promoted demo version."""
    value, _ = context(workspace)
    request = urllib.request.Request(
        value["api_url"] + "/model/reload",
        data=b"{}",
        headers={"Content-Type": "application/json", "X-Reload-Token": value["reload_token"]},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        reloaded = json.load(response)
    expected = json.loads((workspace / f"v{version}/gate.json").read_text())["version"]
    assert reloaded["model_version"] == expected
    payload = {
        "customer_id": "synthetic-demo-client",
        "AMT_INCOME_TOTAL": 40000.0,
        "AMT_CREDIT": 200000.0,
        "AMT_ANNUITY": 12000.0,
        "EXT_SOURCE_1": 0.8,
        "EXT_SOURCE_2": 0.7,
    }
    results = []
    for endpoint in ("predict", "explain"):
        request = urllib.request.Request(
            value["api_url"] + "/" + endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            results.append(json.load(response))
    assert results[0]["model_version"] == results[1]["model_version"] == expected
    assert results[0]["default_score"] == results[1]["default_score"]
    assert len(results[1]["reasons"]) == 3
    write_json(
        workspace / f"api_v{version}.json",
        {
            "model_version": expected,
            "shap_reasons": 3,
            "predict_score": results[0]["default_score"],
        },
    )
    return {"model_version": expected, "shap_reasons": 3}


def monitor(workspace: Path) -> dict:
    """Score a shifted held-out cohort with v1 and feed actual Evidently drift to the branch."""
    _, cfg = context(workspace)
    bundle = pickle.loads((workspace / "v1/bundle.pkl").read_bytes())
    reference, _, _ = read_split(workspace, "val")
    current, _, sensitive = read_split(workspace, "production")
    current["AMT_INCOME_TOTAL"] *= cfg["demo"]["income_multiplier"]
    current["EXT_SOURCE_2"] *= cfg["demo"]["source_multiplier"]
    ref_scores = pd.Series(bundle["pipeline"].predict_proba(reference)[:, 1], index=reference.index)
    scores = pd.Series(bundle["pipeline"].predict_proba(current)[:, 1], index=current.index)
    drift = build_drift_report(
        reference, current, ref_scores, scores, workspace / "monitoring", cfg["monitoring"]
    )
    fairness = monitor_fairness(
        (scores >= bundle["threshold"]).astype(int), sensitive, cfg["fairness"], min_group_size=100
    )
    result = evaluate_alerts(drift, fairness, cfg["monitoring"])
    assert result["trigger_retraining"] and any("drift" in alert for alert in result["alerts"])
    write_json(workspace / "alerts.json", result)
    return {"trigger_retraining": True, "alerts": result["alerts"]}


def main() -> None:
    """Dispatch one operation in the container worker venv."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        choices=[
            "bootstrap",
            "validate_fixture",
            "train_v1",
            "train_v2",
            "train_v3",
            "deploy_v1",
            "deploy_v3",
            "monitor",
        ],
    )
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    if args.operation.startswith("train_"):
        result = train(args.workspace, int(args.operation[-1]))
    elif args.operation.startswith("deploy_"):
        result = reload_api(args.workspace, int(args.operation[-1]))
    else:
        result = {"bootstrap": bootstrap, "validate_fixture": validate_fixture, "monitor": monitor}[
            args.operation
        ](args.workspace)
    write_json(args.workspace / f"task_{args.operation}.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
