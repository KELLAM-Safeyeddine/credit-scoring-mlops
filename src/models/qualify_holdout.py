"""Verify a frozen candidate and evaluate the reserved holdout without retuning."""

import argparse
import copy
import hashlib
import json
import pickle
import sys
from pathlib import Path

import mlflow
import pandas as pd

from src.config import load_config, project_path, write_json
from src.models.evaluate import performance_metrics
from src.models.explain import create_shap_artifacts
from src.models.fairness import audit_fairness
from src.models.promote import candidate_evidence, quality_gate
from src.models.qualification_protocol import protocol_context, verify_memberships


def reserve_holdout(ledger: Path, identity: dict) -> dict | None:
    """Reuse a cached result only for the exact frozen candidate/data/config identity."""
    if ledger.exists():
        previous = json.loads(ledger.read_text(encoding="utf-8"))
        if previous["identity"] != identity:
            raise ValueError("Holdout already reserved for another frozen candidate; no retuning.")
        return previous.get("report")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("x", encoding="utf-8") as stream:
        json.dump({"identity": identity, "status": "reserved"}, stream, indent=2)
    return None


def evaluate_holdout(bundle: dict, cfg: dict, evidence: dict, processed: Path) -> dict:
    """Apply the frozen policy; reuse gate rules with explicitly identified test metrics."""
    x = pd.read_csv(processed / "X_test.csv", index_col="SK_ID_CURR")
    y = pd.read_csv(processed / "y_test.csv", index_col="SK_ID_CURR").TARGET
    sensitive = pd.read_csv(processed / "sensitive_test.csv", index_col="SK_ID_CURR")
    if list(x) != bundle["columns"] or not x.index.is_unique:
        raise ValueError("Holdout feature contract differs.")
    if not x.index.equals(y.index) or not x.index.equals(sensitive.index):
        raise ValueError("Holdout labels/audit attributes must align.")
    _, protocol = protocol_context(cfg)
    if protocol:
        verify_memberships(cfg, protocol, {"test": x})
    for split in ("train", "val"):
        ids = pd.read_csv(processed / f"X_{split}.csv", usecols=["SK_ID_CURR"]).SK_ID_CURR
        if set(ids).intersection(x.index):
            raise ValueError("Holdout overlaps development data.")
    scores = bundle["pipeline"].predict_proba(x)[:, 1]
    metrics = performance_metrics(y, scores, bundle["threshold"], cfg["train"]["threshold"])
    metrics.update(requires_sensitive_at_inference=False, randomized_decision=False)
    fairness = audit_fairness(
        y, (scores >= bundle["threshold"]).astype(int), sensitive, cfg["fairness"], scores
    )
    test_evidence = copy.deepcopy(evidence)
    test_evidence.pop("independent_test_report", None)
    test_evidence.update(metrics=metrics, fairness_report=fairness)
    # Same numerical rules, no production comparator: this is an independent diagnostic.
    passed, rules = quality_gate(test_evidence, None, cfg)
    return {
        "evaluation_split": "test",
        "run_id": bundle["run_id"],
        "rows": len(x),
        "frozen_threshold": bundle["threshold"],
        "passed": passed,
        "rules": rules,
        "failed_rules": [key for key, rule in rules.items() if not rule["passed"]],
        "metrics": {key.replace("validation_", "test_"): value for key, value in metrics.items()},
        "fairness_report": fairness,
        "selection_or_retuning_allowed": False,
    }


def main() -> None:
    """Create SHAP evidence and a single-candidate holdout ledger; never promote here."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.params)
    ledger, _ = protocol_context(cfg)
    content = args.candidate.read_bytes()
    bundle = pickle.loads(content)  # Trusted locally generated bundle only.
    processed = project_path(cfg, "processed")
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    client = mlflow.MlflowClient()
    output = args.candidate.parent
    validation = pd.read_csv(processed / "X_val.csv", index_col="SK_ID_CURR")
    create_shap_artifacts(
        bundle,
        validation,
        output / "shap",
        cfg["fairness"]["shap_sample_size"],
        cfg["data"]["random_state"],
    )
    client.log_artifacts(bundle["run_id"], str(output / "shap"), "shap")
    evidence = candidate_evidence(bundle, client)
    validation_evidence = {
        key: value for key, value in evidence.items() if key != "independent_test_report"
    }
    passed, rules = quality_gate(validation_evidence, None, cfg)
    write_json(output / "validation_gate.json", {"passed": passed, "rules": rules})
    if not passed:
        raise ValueError("Validation/SHAP gate failed; reserved holdout remains unevaluated.")
    identity = {
        "candidate_sha256": hashlib.sha256(content).hexdigest(),
        "run_id": bundle["run_id"],
        "policy_config_sha256": hashlib.sha256(
            json.dumps(
                {key: cfg[key] for key in ("fairness", "quality_gate", "train")}, sort_keys=True
            ).encode()
        ).hexdigest(),
        "data_sha256": {
            name: hashlib.sha256((processed / name).read_bytes()).hexdigest()
            for name in ("X_test.csv", "y_test.csv", "sensitive_test.csv")
        },
    }
    report = reserve_holdout(ledger, identity)
    if report is None:
        report = evaluate_holdout(bundle, cfg, evidence, processed)
    report["data_sha256"] = identity["data_sha256"]
    write_json(ledger, {"identity": identity, "status": "evaluated", "report": report})
    write_json(output / "holdout_report.json", report)
    client.log_dict(bundle["run_id"], report, "qualification/holdout_report.json")
    for key, value in report["metrics"].items():
        if key.startswith("test_"):
            client.log_metric(bundle["run_id"], key, value)
    client.log_dict(
        bundle["run_id"], {"passed": passed, "rules": rules}, "qualification/validation_gate.json"
    )
    client.set_tag(
        bundle["run_id"], "independent_test_status", "passed" if report["passed"] else "failed"
    )
    print(
        json.dumps(
            {
                "run_id": bundle["run_id"],
                "validation_gate_passed": passed,
                "test_passed": report["passed"],
                "test_metrics": report["metrics"],
                "failed_test_rules": report["failed_rules"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
