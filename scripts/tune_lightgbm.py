"""Bounded validation-only LightGBM search, tracked in the same MLflow experiment."""

import json
import sys
import time

import mlflow
import pandas as pd

from src.config import load_config, project_path, write_json
from src.models.evaluate import performance_metrics
from src.models.experiments import ROOT, build_pipeline, data_provenance
from src.models.train import select_threshold


def main() -> None:
    """Log six predefined trials without accessing test or production labels."""
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    processed = project_path(cfg, "processed")
    frames = {
        name: pd.read_csv(processed / f"{name}.csv", index_col="SK_ID_CURR")
        for name in ("X_train", "y_train", "X_val", "y_val")
    }
    provenance, _ = data_provenance(ROOT, processed)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    rows = []
    for leaves, trees, weight in (
        (15, 150, None),
        (15, 300, None),
        (7, 300, None),
        (15, 150, "balanced"),
        (15, 300, "balanced"),
        (7, 300, "balanced"),
    ):
        settings = {
            **cfg["train"]["lightgbm"],
            "num_leaves": leaves,
            "n_estimators": trees,
            "class_weight": weight,
            "min_child_samples": 100,
            "reg_lambda": 5.0,
        }
        cfg["train"]["lightgbm"] = settings
        with mlflow.start_run(run_name=f"phase4-lightgbm-search-{leaves}-{trees}-{weight}") as run:
            mlflow.set_tags(
                {**provenance, "purpose": "bounded_validation_search", "test_split": "untouched"}
            )
            mlflow.log_params(settings)
            pipeline = build_pipeline("lightgbm", cfg)
            start = time.perf_counter()
            pipeline.fit(frames["X_train"], frames["y_train"].TARGET)
            elapsed = time.perf_counter() - start
            scores = pipeline.predict_proba(frames["X_val"])[:, 1]
            threshold = select_threshold(frames["y_val"].TARGET, scores, cfg)
            metrics = performance_metrics(
                frames["y_val"].TARGET, scores, threshold, cfg["train"]["threshold"]
            )
            mlflow.log_metrics({**metrics, "training_seconds": elapsed})
            row = {"params": settings, "metrics": metrics, "run_id": run.info.run_id}
            rows.append(row)
            print(json.dumps(row), flush=True)
    best = max(rows, key=lambda row: row["metrics"]["validation_auc"])
    write_json(
        project_path(cfg, "reports") / "lightgbm_search.json", {"trials": rows, "best": best}
    )
    print("Best parameters:", json.dumps(best["params"]))


if __name__ == "__main__":
    main()
