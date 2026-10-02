"""Load delayed batches, monitor with Evidently, and evaluate retraining alerts."""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import load_config, project_path, write_json
from src.monitoring.drift_report import build_drift_report
from src.monitoring.fairness_monitor import monitor_fairness


def load_batch(workspace: Path, conf: dict) -> dict:
    """Validate batch identifiers and the simulated observation clock."""
    if set(conf) - {"batch_id", "observed_batch", "max_score_psi"}:
        raise ValueError("Unknown monitoring configuration keys.")
    cfg = load_config()
    batch = str(conf.get("batch_id", "01"))
    if batch not in {
        item["batch_id"]
        for item in json.loads((project_path(cfg, "production") / "manifest.json").read_text())[
            "batches"
        ]
    }:
        raise ValueError("Unknown batch_id.")
    observed = conf.get("observed_batch", int(batch))
    if isinstance(observed, bool) or not isinstance(observed, int) or observed < int(batch):
        raise ValueError("observed_batch must be an integer at or after batch_id.")
    threshold = conf.get("max_score_psi", cfg["monitoring"]["max_score_psi"])
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (float, int))
        or not 0 <= threshold <= cfg["monitoring"]["max_score_psi"]
    ):
        raise ValueError("Score PSI override can only tighten the alert threshold.")
    value = {
        "batch_id": batch,
        "observed_batch": observed,
        "max_score_psi": threshold,
        "model_role": "frozen baseline candidate; not deployed champion",
    }
    write_json(workspace / "batch.json", value)
    return value


def frames(workspace: Path) -> tuple:
    """Read aligned frozen reference and current production tables."""
    cfg = load_config()
    context = json.loads((workspace / "batch.json").read_text())
    production = project_path(cfg, "production")
    processed = project_path(cfg, "processed")
    batch = context["batch_id"]

    def read(path):
        return pd.read_csv(path, index_col="SK_ID_CURR")

    current, reference = read(production / f"X_{batch}.csv"), read(processed / "X_val.csv")
    predictions = read(production / f"predictions_{batch}.csv")
    sensitive = read(production / f"sensitive_{batch}.csv")
    bundle = pickle.loads((project_path(cfg, "models") / "baseline.pkl").read_bytes())
    if list(current.columns) != bundle["columns"] or list(reference.columns) != bundle["columns"]:
        raise ValueError("Frozen production/model feature contract differs.")
    if not current.index.equals(predictions.index) or not current.index.equals(sensitive.index):
        raise ValueError("Production indices differ.")
    manifest = json.loads((production / "manifest.json").read_text())
    available = next(
        item["labels_available_at_batch"]
        for item in manifest["batches"]
        if item["batch_id"] == batch
    )
    labels = (
        read(production / f"ground_truth/y_{batch}.csv")["TARGET"]
        if context["observed_batch"] >= available
        else None
    )
    if labels is not None and not current.index.equals(labels.index):
        raise ValueError("Delayed labels are not aligned.")
    reference_scores = pd.Series(
        bundle["pipeline"].predict_proba(reference)[:, 1], index=reference.index
    )
    return cfg, context, current, reference, predictions, sensitive, labels, reference_scores


def execute(operation: str, workspace: Path) -> dict:
    """Perform one monitoring operation, preserving reports outside XCom."""
    cfg, context, current, reference, predictions, sensitive, labels, reference_scores = frames(
        workspace
    )
    if operation == "drift_report":
        settings = {**cfg["monitoring"], "max_score_psi": context["max_score_psi"]}
        value = build_drift_report(
            reference,
            current,
            reference_scores,
            predictions["probability_default"],
            workspace / "monitoring",
            settings,
        )
        value["auc"] = (
            float(roc_auc_score(labels, predictions["probability_default"]))
            if labels is not None and labels.nunique() == 2
            else None
        )
        value["labels_available"] = labels is not None
        write_json(workspace / "drift.json", value)
        return value
    if operation == "fairness_monitor":
        value = monitor_fairness(
            predictions["predicted_default"],
            sensitive,
            cfg["fairness"],
            labels,
            predictions["probability_default"],
        )
        write_json(workspace / "fairness.json", value)
        return {"labels_available": labels is not None}
    if operation == "evaluate_alert_rules":
        drift = json.loads((workspace / "drift.json").read_text())
        fairness = json.loads((workspace / "fairness.json").read_text())
        alerts = []
        if drift["drift_feature_share"] > cfg["monitoring"]["max_drift_feature_share"]:
            alerts.append("data_drift")
        if drift["score_psi"] > context["max_score_psi"]:
            alerts.append("prediction_drift")
        if drift["auc"] is not None and drift["auc"] < cfg["monitoring"]["min_auc"]:
            alerts.append("performance")
        for attribute, report in fairness["attributes"].items():
            if (
                report["disparate_impact"] is not None
                and report["disparate_impact"] < cfg["monitoring"]["min_disparate_impact"]
            ):
                alerts.append(f"fairness.{attribute}")
        value = {
            "trigger_retraining": bool(alerts),
            "alerts": alerts,
            "batch_id": context["batch_id"],
            "model_role": context["model_role"],
        }
        write_json(workspace / "alerts.json", value)
        return value
    raise ValueError(f"Unknown monitoring operation: {operation}")
