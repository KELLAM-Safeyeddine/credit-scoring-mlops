"""Fail-closed quality gate and explicit MLflow alias promotion."""

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException

from src.config import ROOT, load_config, project_path, write_json

SHAP_FILES = {
    "summary.png",
    "bar.png",
    "waterfall_accepted.png",
    "waterfall_refused.png",
    "waterfall_borderline.png",
    "shap_report.json",
}


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def quality_gate(candidate: dict, prod: dict | None, cfg: dict) -> tuple[bool, dict]:
    """Require finite performance, supported fairness and verified SHAP evidence."""
    rules: dict[str, dict] = {}

    def rule(name: str, passed: bool, actual: Any, requirement: Any) -> None:
        rules[name] = {"passed": bool(passed), "actual": actual, "requirement": requirement}

    auc = candidate.get("metrics", {}).get("validation_auc")
    settings = cfg["quality_gate"]
    rule("minimum_auc", _number(auc) and settings["min_auc"] <= auc <= 1, auc, settings["min_auc"])
    if prod is None:
        rule("auc_vs_production", True, None, "first promotion: no production comparator")
    else:
        prod_auc = prod.get("metrics", {}).get("validation_auc")
        same_data = bool(candidate.get("evaluation_data_sha256")) and (
            candidate.get("evaluation_data_sha256") == prod.get("evaluation_data_sha256")
        )
        rule(
            "comparable_validation_data",
            same_data,
            candidate.get("evaluation_data_sha256"),
            prod.get("evaluation_data_sha256"),
        )
        rule(
            "auc_vs_production",
            _number(auc)
            and _number(prod_auc)
            and auc >= prod_auc - settings["max_auc_drop_vs_prod"],
            auc,
            {"production_auc": prod_auc, "max_drop": settings["max_auc_drop_vs_prod"]},
        )
    attributes = candidate.get("fairness_report", {}).get("attributes", {})
    for attr in cfg["fairness"]["sensitive_features"]:
        report = attributes.get(attr, {})
        required = cfg["fairness"].get("required_groups", {}).get(attr, [])
        supported = report.get("sufficient_support") is True and set(required).issubset(
            report.get("eligible_groups", [])
        )
        rule(f"{attr}.support", supported, report.get("eligible_groups"), required)
        di = report.get("disparate_impact")
        gap = report.get("equal_opportunity_difference")
        rule(
            f"{attr}.disparate_impact",
            supported and _number(di) and cfg["fairness"]["min_disparate_impact"] <= di <= 1,
            di,
            cfg["fairness"]["min_disparate_impact"],
        )
        rule(
            f"{attr}.equal_opportunity_difference",
            supported and _number(gap) and 0 <= gap <= cfg["fairness"]["max_equal_opp_diff"],
            gap,
            cfg["fairness"]["max_equal_opp_diff"],
        )
    if settings["require_shap"]:
        rule(
            "shap_artifacts",
            candidate.get("shap_verified") is True
            and SHAP_FILES.issubset(candidate.get("shap_files") or []),
            candidate.get("shap_files", []),
            sorted(SHAP_FILES),
        )
    # Operational guardrails supplement the requested performance/fairness rules.
    utility = cfg["fairness"]["mitigation"]
    recall = candidate.get("metrics", {}).get("validation_recall")
    acceptance = candidate.get("metrics", {}).get("validation_acceptance_rate")
    rule(
        "minimum_recall",
        _number(recall) and recall >= utility["min_recall"],
        recall,
        utility["min_recall"],
    )
    rule(
        "acceptance_bounds",
        _number(acceptance)
        and utility["min_acceptance_rate"] <= acceptance <= utility["max_acceptance_rate"],
        acceptance,
        [utility["min_acceptance_rate"], utility["max_acceptance_rate"]],
    )
    rule(
        "group_free_deterministic_inference",
        candidate.get("metrics", {}).get("requires_sensitive_at_inference") is False
        and candidate.get("metrics", {}).get("randomized_decision") is False,
        {
            key: candidate.get("metrics", {}).get(key)
            for key in ("requires_sensitive_at_inference", "randomized_decision")
        },
        "deterministic scoring without protected attributes at inference",
    )
    return all(value["passed"] for value in rules.values()), rules


def candidate_evidence(bundle: dict, client: mlflow.MlflowClient) -> dict:
    """Verify stored run metrics and actual SHAP artifacts rather than trusting a tag."""
    run_id = bundle["run_id"]
    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise ValueError("Only finished training/audit runs can be evaluated.")
    metrics = dict(bundle["metrics"])
    for key in ("validation_auc", "validation_recall", "validation_acceptance_rate"):
        if metrics.get(key) != run.data.metrics.get(key):
            raise ValueError(f"Local and MLflow metrics disagree: {key}")
    report_path = client.download_artifacts(run_id, "fairness_report.json")
    fairness = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if fairness != bundle["fairness_report"]:
        raise ValueError("Local and MLflow fairness evidence disagree.")
    files = {
        item.path.rsplit("/", 1)[-1]
        for item in client.list_artifacts(run_id, "shap")
        if not item.is_dir and item.file_size is not None and item.file_size > 0
    }
    verified = False
    if SHAP_FILES.issubset(files):
        path = client.download_artifacts(run_id, "shap/shap_report.json")
        shap = json.loads(Path(path).read_text(encoding="utf-8"))
        error = shap.get("max_probability_reconstruction_error")
        verified = (
            shap.get("model_run_id") == run_id
            and _number(error)
            and 0 <= error < 1e-6
            and shap.get("feature_count") == len(bundle["columns"])
        )
    # The parent training run records the frozen validation input hash.
    parent_id = run.data.tags.get("parent_run_id")
    parent = client.get_run(parent_id) if parent_id else run
    x_hash = parent.data.tags.get("data_sha256.X_val.csv")
    y_hash = parent.data.tags.get("data_sha256.y_val.csv")
    if not x_hash or not y_hash:
        raise ValueError("Missing validation data provenance.")
    data_hash = f"{x_hash}:{y_hash}"
    return {
        "run_id": run_id,
        "model_uri": bundle["model_uri"],
        "metrics": metrics,
        "fairness_report": fairness,
        "shap_verified": verified,
        "shap_files": sorted(files),
        "evaluation_data_sha256": data_hash,
        "training_data_sha256": parent.data.tags.get("training_data_sha256"),
        "raw_dvc_hash": parent.data.tags.get("raw_dvc_hash"),
    }


def champion_version(client: mlflow.MlflowClient, name: str) -> Any:
    """Read the current champion; distinguish a missing alias from other failures."""
    try:
        registered = client.get_registered_model(name)
    except MlflowException as error:
        if error.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise
    version = registered.aliases.get("champion")
    return client.get_model_version(name, version) if version else None


def register_and_promote(candidate: dict, cfg: dict, client: mlflow.MlflowClient) -> dict:
    """Register rejected candidates too; change champion only after all rules pass."""
    name = cfg["mlflow"]["registered_model_name"]
    if "'" in name:
        raise ValueError("Registry name cannot contain single quotes in this workflow.")
    current = champion_version(client, name)
    prod = None
    if current is not None:
        path = client.download_artifacts(current.run_id, "promotion/evidence.json")
        prod = json.loads(Path(path).read_text(encoding="utf-8"))
    passed, rules = quality_gate(candidate, prod, cfg)
    failed = [key for key, value in rules.items() if not value["passed"]]
    status = "accepted" if passed else "rejected"
    result = {
        "passed": passed,
        "gate_status": status,
        "rules": rules,
        "failed_rules": failed,
        "run_id": candidate["run_id"],
        "registered_model_name": name,
    }
    client.log_dict(candidate["run_id"], candidate, "promotion/evidence.json")
    client.log_dict(candidate["run_id"], result, "promotion/quality_gate.json")
    client.log_dict(candidate["run_id"], cfg, "promotion/params.json")
    client.set_tag(candidate["run_id"], "gate_status", status)
    client.set_tag(candidate["run_id"], "gate_failed_rules", json.dumps(failed))
    # Idempotent registration by source run: retries do not create new versions.
    try:
        existing = list(
            client.search_model_versions(f"name = '{name}' AND run_id = '{candidate['run_id']}'")
        )
    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        existing = []
    matches = [version for version in existing if version.run_id == candidate["run_id"]]
    if matches:
        version = matches[0]
    else:
        version = mlflow.register_model(candidate["model_uri"], name)
    result["version"] = str(version.version)
    for key, value in {
        "gate_status": status,
        "gate_failed_rules": json.dumps(failed),
        "validation_auc": str(candidate["metrics"]["validation_auc"]),
        "evaluation_data_sha256": candidate["evaluation_data_sha256"],
    }.items():
        client.set_model_version_tag(name, version.version, key, value)
    if passed:
        latest = champion_version(client, name)
        if (latest.version if latest else None) != (current.version if current else None):
            raise RuntimeError(
                "Champion changed during evaluation; rerun the gate before promotion."
            )
        if current is not None and current.version != version.version:
            client.set_registered_model_alias(name, "previous", current.version)
        client.set_registered_model_alias(name, "champion", version.version)
    final_champion = champion_version(client, name)
    result["champion_version"] = str(final_champion.version) if final_champion else None
    client.log_dict(candidate["run_id"], result, "promotion/quality_gate.json")
    return result


def main() -> None:
    """Evaluate a trusted local candidate and apply the gate to the real Registry."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, default=ROOT / "params.yaml")
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.params)
    path = args.candidate or project_path(cfg, "audit_models") / "candidate.pkl"
    bundle = pickle.loads(path.read_bytes())
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    client = mlflow.MlflowClient()
    evidence = candidate_evidence(bundle, client)
    result = register_and_promote(evidence, cfg, client)
    write_json(project_path(cfg, "reports") / "promotion" / f"{evidence['run_id']}.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
