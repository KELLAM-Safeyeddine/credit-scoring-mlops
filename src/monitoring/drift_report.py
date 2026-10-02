"""Evidently 0.7 reports for frozen model features and prediction scores."""

from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset


def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_scores: pd.Series,
    current_scores: pd.Series,
    output: Path,
    settings: dict,
) -> dict:
    """Save a real Evidently snapshot and return explicit drift indicators."""
    if list(reference.columns) != list(current.columns):
        raise ValueError("Reference/current feature columns differ.")
    if not reference.index.equals(reference_scores.index) or not current.index.equals(
        current_scores.index
    ):
        raise ValueError("Score/feature indices differ.")
    columns = [column for column in reference if reference[column].nunique(dropna=True) > 1]
    ref, cur = reference.copy(), current.copy()
    ref["probability_default"], cur["probability_default"] = reference_scores, current_scores
    definition = DataDefinition(numerical_columns=list(ref.columns), categorical_columns=[])
    snapshot = Report(
        [
            DataDriftPreset(
                columns=columns,
                drift_share=settings["max_drift_feature_share"],
                num_method="psi",
                num_threshold=0.2,
            ),
            ValueDrift(
                column="probability_default", method="psi", threshold=settings["max_score_psi"]
            ),
        ]
    ).run(
        reference_data=Dataset.from_pandas(ref, definition),
        current_data=Dataset.from_pandas(cur, definition),
    )
    output.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output / "drift.html"))
    snapshot.save_json(str(output / "evidently.json"))
    metrics = snapshot.dict()["metrics"]
    share = next(
        row["value"]["share"]
        for row in metrics
        if row["config"]["type"].endswith(":DriftedColumnsCount")
    )
    psi = next(
        row["value"] for row in metrics if row["config"].get("column") == "probability_default"
    )
    return {
        "drift_feature_share": float(share),
        "score_psi": float(psi),
        "tested_features": len(columns),
        "constant_reference_features": len(reference.columns) - len(columns),
    }
