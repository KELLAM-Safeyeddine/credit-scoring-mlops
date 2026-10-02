"""Convert pseudonymized API events to Evidently inputs, separated by model run."""

import json
from pathlib import Path

import pandas as pd

from src.features.contracts import EXCLUDED
from src.monitoring.drift_report import build_drift_report


def read_api_events(
    path: Path, columns: list[str], run_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require the exact ordered model contract; never mix versions or infer audit groups."""
    if EXCLUDED.intersection(columns) or len(columns) != len(set(columns)):
        raise ValueError("Unsafe feature contract.")
    rows, predictions, ids = [], [], []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("model_run_id") != run_id:
                continue
            if list(event["features"]) != columns or "customer_id" in event:
                raise ValueError("API event violates the trained or pseudonymization contract.")
            score, decision = event["default_score"], event["predicted_default"]
            if not 0 <= score <= 1 or decision not in (0, 1):
                raise ValueError("Invalid logged prediction.")
            rows.append(event["features"])
            ids.append(event["request_id"])
            predictions.append({"probability_default": score, "predicted_default": decision})
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("No matching events or duplicate request IDs.")
    index = pd.Index(ids, name="request_id")
    features = pd.DataFrame(rows, index=index, columns=columns).astype(float)
    return features, pd.DataFrame(predictions, index=index)


def report_api_events(
    path: Path, bundle: dict, reference: pd.DataFrame, output: Path, settings: dict
) -> dict:
    """Feed model-aligned API features/scores to Evidently without inventing audit labels."""
    current, predictions = read_api_events(path, bundle["columns"], bundle["run_id"])
    reference_scores = pd.Series(
        bundle["pipeline"].predict_proba(reference)[:, 1], index=reference.index
    )
    result = build_drift_report(
        reference, current, reference_scores, predictions["probability_default"], output, settings
    )
    result.update(
        model_run_id=bundle["run_id"],
        requests=len(current),
        fairness_status="requires separately authorized audit join by request_id",
        performance_status="requires separately released outcomes by request_id",
    )
    return result
