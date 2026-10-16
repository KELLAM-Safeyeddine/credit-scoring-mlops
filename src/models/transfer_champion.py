"""Transfer a qualified champion through MLflow APIs without reevaluating test."""

import argparse
import copy
import importlib.metadata
import json
import pickle
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from src.config import load_config, project_path, write_json
from src.models.calibration import CalibratedMixture
from src.models.promote import candidate_evidence, quality_gate, register_and_promote


def transfer(params: Path, candidate: Path, target_uri: str) -> dict:
    """Copy the same fitted pipeline and retain original qualification provenance."""
    cfg = load_config(params)
    source_uri = cfg["mlflow"]["tracking_uri"]
    if target_uri == source_uri:
        raise ValueError("Source and target tracking servers must differ.")
    bundle = pickle.loads(candidate.read_bytes())
    source_client = mlflow.MlflowClient(tracking_uri=source_uri)
    evidence = candidate_evidence(bundle, source_client)
    if not quality_gate(evidence, None, cfg)[0]:
        raise ValueError("Only independently qualified candidates can be transferred.")
    source_run = source_client.get_run(bundle["run_id"])
    target_cfg = copy.deepcopy(cfg)
    target_cfg["mlflow"]["tracking_uri"] = target_uri
    mlflow.set_tracking_uri(target_uri)
    mlflow.set_experiment("credit-scoring-qualified-imports")
    with mlflow.start_run(run_name="qualified-home-credit-import") as run:
        run_id = run.info.run_id
        mlflow.set_tags(
            {
                **{k: v for k, v in source_run.data.tags.items() if k != "parent_run_id"},
                "qualified_source_run_id": bundle["run_id"],
                "qualified_source_tracking_uri": source_uri,
                "qualification_state": "qualified_model_import_no_test_reevaluation",
            }
        )
        mlflow.log_metrics(source_run.data.metrics)
        mlflow.log_dict(evidence, "qualification/original_source_evidence.json")
        for directory in ("features", "provenance"):
            local = source_client.download_artifacts(bundle["run_id"], directory)
            mlflow.log_artifacts(local, directory)
        for name in ("feature_columns.json", "fairness_report.json"):
            mlflow.log_artifact(source_client.download_artifacts(bundle["run_id"], name))
        local_shap = Path(source_client.download_artifacts(bundle["run_id"], "shap"))
        for item in local_shap.iterdir():
            if item.name != "shap_report.json":
                mlflow.log_artifact(str(item), "shap")
        shap_report = json.loads((local_shap / "shap_report.json").read_text(encoding="utf-8"))
        shap_report["qualified_source_run_id"] = bundle["run_id"]
        shap_report["model_run_id"] = run_id
        mlflow.log_dict(shap_report, "shap/shap_report.json")
        holdout = copy.deepcopy(evidence["independent_test_report"])
        holdout["qualified_source_run_id"] = bundle["run_id"]
        holdout["run_id"] = run_id
        holdout["imported_without_test_reevaluation"] = True
        mlflow.log_dict(holdout, "qualification/holdout_report.json")
        calibrated = isinstance(bundle["pipeline"][-1], CalibratedMixture)
        if calibrated:
            for item in source_client.list_artifacts(bundle["run_id"], "calibration"):
                if item.is_dir or item.path.endswith("/report.json"):
                    continue
                mlflow.log_artifact(
                    source_client.download_artifacts(bundle["run_id"], item.path), "calibration"
                )
            calibration = json.loads(
                Path(
                    source_client.download_artifacts(bundle["run_id"], "calibration/report.json")
                ).read_text(encoding="utf-8")
            )
            calibration["qualified_source_run_id"] = bundle["run_id"]
            calibration["run_id"] = run_id
            calibration["imported_without_test_reevaluation"] = True
            mlflow.log_dict(calibration, "calibration/report.json")
        example = pd.read_csv(
            project_path(cfg, "processed") / "X_val.csv", index_col="SK_ID_CURR", nrows=3
        ).astype(float)
        info = mlflow.sklearn.log_model(
            sk_model=bundle["pipeline"],
            name="qualified_probability_ensemble",
            input_example=example,
            signature=infer_signature(example, bundle["pipeline"].predict_proba(example)),
            pyfunc_predict_fn="predict_proba",
            pip_requirements=[
                f"{package}=={importlib.metadata.version(package)}"
                for package in ("scikit-learn", "numpy", "pandas", "lightgbm", "mlflow", "scipy")
            ],
            metadata={
                "threshold": bundle["threshold"],
                "positive_label": 1,
                "calibrated": calibrated,
                "score_semantics": "calibrated default probability"
                if calibrated
                else "uncalibrated weighted oracle probabilities",
            },
        )
    imported = {**bundle, "run_id": run_id, "model_uri": info.model_uri}
    target_client = mlflow.MlflowClient(tracking_uri=target_uri)
    imported_evidence = candidate_evidence(imported, target_client)
    result = register_and_promote(imported_evidence, target_cfg, target_client, expected_pass=True)
    result.update(
        qualified_source_run_id=bundle["run_id"],
        source_tracking_uri=source_uri,
        target_tracking_uri=target_uri,
        test_reevaluated=False,
    )
    write_json(candidate.parent / "registry_transfer.json", result)
    return result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--target-uri", required=True)
    args = parser.parse_args()
    print(json.dumps(transfer(args.params, args.candidate, args.target_uri), indent=2))
