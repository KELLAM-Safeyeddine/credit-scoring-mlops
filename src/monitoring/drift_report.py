"""Evidently 0.7 reports for frozen model features and prediction scores."""

from pathlib import Path

import numpy as np
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import ClassificationPreset, DataDriftPreset
from sklearn.metrics import roc_auc_score


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
    variable = [column for column in reference if reference[column].nunique(dropna=True) > 1]
    columns = [column for column in variable if current[column].notna().sum() >= 2]
    ref, cur = reference.copy(), current.copy()
    ref["probability_default"], cur["probability_default"] = reference_scores, current_scores
    # Evidently 0.7's legacy time-index plots cannot bin UUID/string indices.
    # Identity alignment is checked above; reports display observation order only.
    ref, cur = ref.reset_index(drop=True), cur.reset_index(drop=True)
    definition = DataDefinition(numerical_columns=list(ref.columns), categorical_columns=[])
    snapshot = Report(
        [
            *(
                [
                    DataDriftPreset(
                        columns=columns,
                        drift_share=settings["max_drift_feature_share"],
                        num_method="psi",
                        num_threshold=settings.get("feature_psi_threshold", 0.2),
                    )
                ]
                if columns
                else []
            ),
            ValueDrift(
                column="probability_default", method="psi", threshold=settings["max_score_psi"]
            ),
            *[
                ValueDrift(column=column, method="psi", threshold=settings["critical_feature_psi"])
                for column in settings.get("critical_features", [])
                if column in columns
            ],
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
        (
            row["value"]["share"]
            for row in metrics
            if row["config"]["type"].endswith(":DriftedColumnsCount")
        ),
        0.0,
    )
    psi = next(
        row["value"] for row in metrics if row["config"].get("column") == "probability_default"
    )
    return {
        "drift_feature_share": float(share),
        "score_psi": float(psi),
        "tested_features": len(columns),
        "constant_reference_features": len(reference.columns) - len(variable),
        "untestable_current_features": [column for column in variable if column not in columns],
        "missingness_delta": {
            column: float(abs(reference[column].isna().mean() - current[column].isna().mean()))
            for column in reference
        },
        "constant_emergence": {
            column: float(
                (~current[column].dropna().isin(reference[column].dropna().unique())).mean()
            )
            if current[column].notna().any()
            else 0.0
            for column in reference
            if column not in variable
        },
        "feature_psi": {
            row["config"]["column"]: float(row["value"])
            for row in metrics
            if row["config"].get("column") in columns
        },
        "critical_feature_psi": {
            row["config"]["column"]: float(row["value"])
            for row in metrics
            if row["config"].get("column") in settings.get("critical_features", [])
        },
    }


def build_population_report(
    reference: pd.DataFrame, current: pd.DataFrame, output: Path, settings: dict
) -> dict:
    """Audit population drift separately from protected-attribute-free model inputs."""
    columns = ["CODE_GENDER", "AGE_GROUP"]
    definition = DataDefinition(categorical_columns=columns, numerical_columns=[])
    snapshot = Report(
        [
            ValueDrift(column=column, method="psi", threshold=settings["max_population_psi"])
            for column in columns
        ]
    ).run(
        reference_data=Dataset.from_pandas(reference[columns].reset_index(drop=True), definition),
        current_data=Dataset.from_pandas(current[columns].reset_index(drop=True), definition),
    )
    output.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output / "population.html"))
    snapshot.save_json(str(output / "population.json"))
    return {row["config"]["column"]: float(row["value"]) for row in snapshot.dict()["metrics"]}


def auc_interval(labels: pd.Series, scores: pd.Series, settings: dict, seed: int) -> dict:
    """Stratified bootstrap avoids one-class draws in rare-default batches."""
    if not labels.index.equals(scores.index):
        raise ValueError("Outcome/score indices differ.")
    count, confidence = settings["auc_bootstrap_samples"], settings["auc_confidence"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 20
        or not 0 < confidence < 1
    ):
        raise ValueError("Use at least 20 bootstrap draws and a confidence between zero and one.")
    if not labels.isin([0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Outcomes must be binary and scores finite.")
    positive = np.flatnonzero(labels.to_numpy() == 1)
    negative = np.flatnonzero(labels.to_numpy() == 0)
    if not len(positive) or not len(negative):
        return {"auc": None, "auc_lower": None, "auc_upper": None, "status": "one_class"}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(settings["auc_bootstrap_samples"]):
        ids = np.concatenate(
            [rng.choice(positive, len(positive)), rng.choice(negative, len(negative))]
        )
        values.append(roc_auc_score(labels.iloc[ids], scores.iloc[ids]))
    tail = (1 - settings["auc_confidence"]) / 2
    lower, upper = np.quantile(values, [tail, 1 - tail])
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "auc_lower": float(lower),
        "auc_upper": float(upper),
        "confidence": settings["auc_confidence"],
        "bootstrap_samples": settings["auc_bootstrap_samples"],
        "status": "measured",
    }


def build_performance_report(
    reference_labels: pd.Series,
    reference_predictions: pd.DataFrame,
    labels: pd.Series,
    predictions: pd.DataFrame,
    output: Path,
    threshold: float,
    settings: dict,
    seed: int,
) -> dict:
    """Archive Evidently classification metrics only after ground truth is released."""
    result = auc_interval(labels, predictions["probability_default"], settings, seed)
    if result["auc"] is None:
        return result
    frames = []
    for truth, predicted in ((reference_labels, reference_predictions), (labels, predictions)):
        if not truth.index.equals(predicted.index):
            raise ValueError("Performance labels and predictions are not aligned.")
        frame = predicted.copy()
        frame["TARGET"] = truth
        frames.append(frame.reset_index(drop=True))
    definition = DataDefinition(
        classification=[
            BinaryClassification(
                target="TARGET",
                prediction_labels="predicted_default",
                prediction_probas="probability_default",
            )
        ]
    )
    snapshot = Report([ClassificationPreset(probas_threshold=threshold)]).run(
        reference_data=Dataset.from_pandas(frames[0], definition),
        current_data=Dataset.from_pandas(frames[1], definition),
    )
    output.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output / "performance.html"))
    snapshot.save_json(str(output / "performance.json"))
    result["evidently_metrics"] = {
        row["config"]["type"].rsplit(":", 1)[-1]: float(row["value"])
        for row in snapshot.dict()["metrics"]
        if isinstance(row["value"], (float, int, np.floating))
    }
    if not np.isclose(result["auc"], result["evidently_metrics"]["RocAuc"]):
        raise ValueError("Evidently and sklearn disagree on AUC.")
    return result
