"""Run business operations in the container's isolated creditS environment."""

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import mlflow
import yaml

from src.config import ROOT, load_config, project_path, write_json

RUNTIME = ROOT / "runtime/airflow/runs"


def create_workspace(run_id: str, conf: dict) -> Path:
    """Give each DAG run private outputs; allow only a stricter AUC override."""
    if set(conf) - {
        "min_auc",
        "source_monitoring_run",
        "batch_id",
        "observed_batch",
        "max_score_psi",
    }:
        raise ValueError("Unknown training configuration keys.")
    cfg = load_config()
    minimum = conf.get("min_auc", cfg["quality_gate"]["min_auc"])
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not cfg["quality_gate"]["min_auc"] <= minimum <= 1
    ):
        raise ValueError("min_auc may only tighten the configured gate, up to 1.")
    workspace = RUNTIME / hashlib.sha256(run_id.encode()).hexdigest()[:24]
    workspace.mkdir(parents=True, exist_ok=True)
    context = {"airflow_run_id": run_id, "conf": conf}
    if (workspace / "context.json").exists():
        if json.loads((workspace / "context.json").read_text()) != context:
            raise ValueError("Cannot reuse a workspace with different run configuration.")
        return workspace
    cfg["quality_gate"]["min_auc"] = minimum
    cfg["mlflow"]["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    for name in cfg["paths"]:
        cfg["paths"][name] = (
            str(ROOT / cfg["paths"][name]) if name == "raw" else str(workspace / name)
        )
    cfg["runtime"]["python"] = sys.executable
    # Linux container scratch avoids thousands of writes across a Windows bind mount.
    cfg["features"]["scratch_dir"] = str(
        Path("/tmp/credit-features") / workspace.name
        if os.name == "posix"
        else workspace / "scratch"
    )
    (workspace / "params.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(workspace / "context.json", context)
    return workspace


def checked_workspace(value: str) -> Path:
    """Reject paths outside the private run area."""
    path = Path(value).resolve()
    if path.parent != RUNTIME.resolve() or not (path / "params.yaml").is_file():
        raise ValueError("Invalid Airflow workspace.")
    return path


def invoke_module(module: str, workspace: Path) -> None:
    """Execute existing CLI services with the exact worker interpreter."""
    cfg = load_config(workspace / "params.yaml")
    scratch = Path(cfg["features"].get("scratch_dir", workspace / "scratch"))
    scratch.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", module, "--params", str(workspace / "params.yaml")],
        cwd=ROOT,
        check=True,
        env={**os.environ, "TMPDIR": str(scratch)},
    )


def gate_evidence(cfg: dict) -> tuple[dict, mlflow.MlflowClient]:
    """Read the selected auditable candidate and its verified MLflow evidence."""
    from src.models.promote import candidate_evidence

    directory = project_path(cfg, "audit_models")
    candidate = directory / "candidate.pkl"
    if not candidate.exists():
        candidate = directory / "baseline.pkl"
    bundle = pickle.loads(candidate.read_bytes())
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    client = mlflow.MlflowClient()
    return candidate_evidence(bundle, client), client


def execute(operation: str, workspace: Path) -> dict:
    """Run one retryable task; return only small control values."""
    cfg = load_config(workspace / "params.yaml")
    if operation == "load_production_batch":
        from src.orchestration.monitoring import load_batch

        context = json.loads((workspace / "context.json").read_text())
        load_batch(workspace, context["conf"])
        return {"workspace": str(workspace)}
    if operation in {"drift_report", "fairness_monitor", "evaluate_alert_rules"}:
        from src.orchestration.monitoring import execute as monitor

        return monitor(operation, workspace)
    if operation == "ingest":
        from scripts.extract_data import extract_archive

        extract_archive(ROOT, project_path(cfg, "raw"))
        return {"workspace": str(workspace)}
    if operation == "validate_data":
        from src.data.validate import load_validated_application

        frame = load_validated_application(project_path(cfg, "raw") / "application_train.csv")
        return {"rows": len(frame)}
    if operation == "build_features":
        invoke_module("src.data.make_dataset", workspace)
        invoke_module("src.features.build_features", workspace)
    elif operation == "train_models":
        invoke_module("src.models.train", workspace)
    elif operation == "evaluate_performance":
        bundle = pickle.loads((project_path(cfg, "models") / "baseline.pkl").read_bytes())
        return {"run_id": bundle["run_id"], "validation_auc": bundle["metrics"]["validation_auc"]}
    elif operation == "evaluate_fairness":
        invoke_module("src.models.mitigate", workspace)
    elif operation == "compute_shap":
        invoke_module("src.models.explain", workspace)
    elif operation == "quality_gate":
        from src.models.promote import champion_version, quality_gate

        evidence, client = gate_evidence(cfg)
        current = champion_version(client, cfg["mlflow"]["registered_model_name"])
        prod = None
        if current:
            prod = json.loads(
                Path(
                    client.download_artifacts(current.run_id, "promotion/evidence.json")
                ).read_text()
            )
        passed, rules = quality_gate(evidence, prod, cfg)
        write_json(
            workspace / "gate.json",
            {"passed": passed, "rules": rules, "run_id": evidence["run_id"]},
        )
        return {"passed": passed, "run_id": evidence["run_id"]}
    elif operation in {"register_and_promote", "reject_model"}:
        from src.models.promote import register_and_promote

        evidence, client = gate_evidence(cfg)
        result = register_and_promote(
            evidence, cfg, client, expected_pass=operation == "register_and_promote"
        )
        write_json(workspace / "promotion.json", result)
        return {key: result[key] for key in ("gate_status", "run_id", "version")}
    elif operation == "notification":
        result = json.loads((workspace / "promotion.json").read_text())
        summary = {key: result[key] for key in ("gate_status", "run_id", "version", "failed_rules")}
        write_json(workspace / "notification.json", summary)
        print(json.dumps(summary), flush=True)
        return {"status": result["gate_status"]}
    else:
        raise ValueError(f"Unknown operation: {operation}")
    return {"completed": operation}


def main() -> None:
    """Dispatch an operation and persist its small result for the thin DAG."""
    parser = argparse.ArgumentParser()
    parser.add_argument("operation")
    parser.add_argument("--workspace")
    parser.add_argument("--run-id")
    parser.add_argument("--conf", default="{}")
    args = parser.parse_args()
    workspace = (
        checked_workspace(args.workspace)
        if args.workspace
        else create_workspace(args.run_id, json.loads(args.conf))
    )
    checkpoint = workspace / f"task_{args.operation}.json"
    # Resume completed expensive deterministic work on Airflow retries.
    if not checkpoint.exists() or args.operation in {
        "quality_gate",
        "register_and_promote",
        "reject_model",
        "notification",
    }:
        write_json(checkpoint, execute(args.operation, workspace))
    print(json.dumps({"result_file": str(checkpoint)}), flush=True)


if __name__ == "__main__":
    main()
