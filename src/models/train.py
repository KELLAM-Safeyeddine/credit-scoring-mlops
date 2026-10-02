"""Train a real local DVC baseline without promotion or MLflow tracking yet."""

import json
import pickle
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ROOT, load_config, parse_params, project_path, write_json
from src.features.build_features import EXCLUDED


def select_threshold(y: pd.Series, probabilities: np.ndarray, cfg: dict[str, Any]) -> float:
    """Minimize business cost on validation; predicted positive means refusal."""
    settings = cfg["train"]["threshold"]
    thresholds = np.linspace(settings["min"], settings["max"], settings["steps"])
    costs = []
    for threshold in thresholds:
        refused = probabilities >= threshold
        costs.append(
            settings["false_negative_cost"] * np.sum((y == 1) & ~refused)
            + settings["false_positive_cost"] * np.sum((y == 0) & refused)
        )
    return float(thresholds[int(np.argmin(costs))])


def fit_baseline(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: dict[str, Any],
) -> tuple[dict, dict]:
    """Fit transformations on train only; leave the test split untouched."""
    if list(x_train.columns) != list(x_val.columns) or EXCLUDED.intersection(x_train.columns):
        raise ValueError("Invalid feature contract.")
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    C=cfg["train"]["baseline"]["C"],
                    max_iter=cfg["train"]["baseline"]["max_iter"],
                    random_state=cfg["data"]["random_state"],
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_val)[:, 1]
    threshold = select_threshold(y_val, probabilities, cfg)
    predictions = (probabilities >= threshold).astype("int8")
    auc = float(roc_auc_score(y_val, probabilities))
    metrics = {
        "phase": 3,
        "model": "logistic_regression_engineered_baseline",
        "validation_auc": auc,
        "validation_pr_auc": float(average_precision_score(y_val, probabilities)),
        "validation_gini": 2 * auc - 1,
        "validation_recall": float(recall_score(y_val, predictions, zero_division=0)),
        "validation_precision": float(precision_score(y_val, predictions, zero_division=0)),
        "threshold": threshold,
        "feature_count": len(x_train.columns),
        "train_rows": len(x_train),
        "validation_rows": len(x_val),
        "promotion_status": "not_evaluated_not_promoted",
        "positive_definition": "1 = default predicted = refusal; acceptance = 1 - prediction",
    }
    return {"pipeline": pipeline, "columns": list(x_train.columns), "threshold": threshold}, metrics


def main() -> None:
    """Write a local baseline model and reproducible validation metrics."""
    cfg = load_config(parse_params())
    processed = project_path(cfg, "processed")

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(processed / f"{name}.csv", index_col="SK_ID_CURR")

    bundle, metrics = fit_baseline(
        read("X_train"), read("y_train").TARGET, read("X_val"), read("y_val").TARGET, cfg
    )
    raw_version = yaml.safe_load((ROOT / "data/raw.dvc").read_text(encoding="utf-8"))
    metrics["raw_dvc_hash"] = raw_version["outs"][0]["md5"]
    bundle["metrics"] = metrics
    model = project_path(cfg, "models") / "baseline.pkl"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL))
    write_json(project_path(cfg, "reports") / "baseline_metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
