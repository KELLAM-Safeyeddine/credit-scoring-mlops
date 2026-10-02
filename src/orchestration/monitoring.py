"""Load delayed batches, monitor with Evidently, and evaluate retraining alerts."""

import json
import pickle
from pathlib import Path

import pandas as pd

from src.config import load_config, project_path, write_json
from src.data.simulate_production import monitor_model_path
from src.monitoring.alerts import evaluate_alerts
from src.monitoring.drift_report import (
    build_drift_report,
    build_performance_report,
    build_population_report,
)
from src.monitoring.fairness_monitor import monitor_fairness


def load_batch(workspace: Path, conf: dict, cfg: dict | None = None) -> dict:
    """Validate batch identifiers and the simulated observation clock."""
    if set(conf) - {"batch_id", "observed_batch", "max_score_psi"}:
        raise ValueError("Unknown monitoring configuration keys.")
    cfg = cfg or load_config()
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
        "model_role": "frozen mitigated candidate for diagnosis; not deployed champion",
    }
    write_json(workspace / "batch.json", value)
    return value


def frames(workspace: Path, cfg: dict | None = None) -> tuple:
    """Read aligned frozen reference and current production tables."""
    cfg = cfg or load_config()
    context = json.loads((workspace / "batch.json").read_text())
    production = project_path(cfg, "production")
    processed = project_path(cfg, "processed")
    batch = context["batch_id"]

    def read(path):
        return pd.read_csv(path, index_col="SK_ID_CURR")

    current, reference = read(production / f"X_{batch}.csv"), read(processed / "X_val.csv")
    predictions = read(production / f"predictions_{batch}.csv")
    sensitive = read(production / f"sensitive_{batch}.csv")
    bundle = pickle.loads(monitor_model_path(cfg).read_bytes())
    if list(current.columns) != bundle["columns"] or list(reference.columns) != bundle["columns"]:
        raise ValueError("Frozen production/model feature contract differs.")
    if not current.index.equals(predictions.index) or not current.index.equals(sensitive.index):
        raise ValueError("Production indices differ.")
    if not current.index.is_unique or not reference.index.is_unique:
        raise ValueError("Monitoring indices must be unique.")
    if (
        not predictions.predicted_default.isin([0, 1]).all()
        or not predictions.probability_default.between(0, 1).all()
    ):
        raise ValueError("Invalid production probabilities or decisions.")
    manifest = json.loads((production / "manifest.json").read_text())
    if manifest.get("model_run_id") != bundle.get("run_id"):
        raise ValueError("Batch scores and monitored model run differ.")
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


def execute(operation: str, workspace: Path, cfg: dict | None = None) -> dict:
    """Perform one monitoring operation, preserving reports outside XCom."""
    cfg, context, current, reference, predictions, sensitive, labels, reference_scores = frames(
        workspace, cfg
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
        processed = project_path(cfg, "processed")
        reference_sensitive = pd.read_csv(processed / "sensitive_val.csv", index_col="SK_ID_CURR")
        value["population_psi"] = build_population_report(
            reference_sensitive, sensitive, workspace / "monitoring", settings
        )
        value.update(auc=None, auc_upper=None)
        if labels is not None:
            bundle = pickle.loads(monitor_model_path(cfg).read_bytes())
            ref_predictions = pd.DataFrame(
                {
                    "probability_default": reference_scores,
                    "predicted_default": (reference_scores >= bundle["threshold"]).astype(int),
                },
                index=reference.index,
            )
            performance = build_performance_report(
                pd.read_csv(processed / "y_val.csv", index_col="SK_ID_CURR").TARGET,
                ref_predictions,
                labels,
                predictions,
                workspace / "monitoring",
                bundle["threshold"],
                settings,
                cfg["data"]["random_state"],
            )
            value.update(
                {key: item for key, item in performance.items() if key != "evidently_metrics"}
            )
            write_json(workspace / "performance.json", performance)
        value["labels_available"] = labels is not None
        write_json(workspace / "drift.json", value)
        return {
            key: value[key]
            for key in ("auc", "auc_upper", "score_psi", "drift_feature_share", "labels_available")
        }
    if operation == "fairness_monitor":
        value = monitor_fairness(
            predictions["predicted_default"],
            sensitive,
            {
                **cfg["fairness"],
                "min_group_size": cfg["monitoring"]["min_group_size"],
                "min_class_count": cfg["monitoring"]["min_class_count"],
            },
            labels,
            predictions["probability_default"],
            min_group_size=cfg["monitoring"]["min_group_size"],
        )
        write_json(workspace / "fairness.json", value)
        return {"labels_available": labels is not None}
    if operation == "evaluate_alert_rules":
        drift = json.loads((workspace / "drift.json").read_text())
        fairness = json.loads((workspace / "fairness.json").read_text())
        value = {
            **evaluate_alerts(drift, fairness, cfg["monitoring"], context["max_score_psi"]),
            "batch_id": context["batch_id"],
            "model_role": context["model_role"],
        }
        write_json(workspace / "alerts.json", value)
        return value
    raise ValueError(f"Unknown monitoring operation: {operation}")
