"""Four train-only pipelines and MLflow 3 experiment logging."""

import hashlib
import importlib.metadata
import json
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import ROOT, project_path, write_json
from src.features.contracts import EXCLUDED
from src.models.evaluate import performance_metrics

FAMILIES = {"logistic_regression", "random_forest", "xgboost", "lightgbm"}


def build_pipeline(name: str, cfg: dict[str, Any]) -> Pipeline:
    """Create a deterministic estimator with transformations fitted on train."""
    seed = cfg["data"]["random_state"]
    settings = cfg["train"]
    constructors = {
        "logistic_regression": lambda: LogisticRegression(
            solver="liblinear", random_state=seed, **settings["baseline"]
        ),
        "random_forest": lambda: RandomForestClassifier(
            random_state=seed, **settings["random_forest"]
        ),
        "xgboost": lambda: XGBClassifier(
            random_state=seed, eval_metric="logloss", **settings["xgboost"]
        ),
        "lightgbm": lambda: LGBMClassifier(
            random_state=seed,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            **settings["lightgbm"],
        ),
    }
    if name not in constructors:
        raise ValueError(f"Unsupported model: {name}")
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    # Keep ndarray output: Home Credit category names contain JSON punctuation
    # rejected by LightGBM. The ordered original names remain in the bundle.
    if name == "logistic_regression":
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", constructors[name]()))
    return Pipeline(steps)


def data_provenance(root: Path, processed: Path) -> tuple[dict[str, str], bytes]:
    """Capture the existing lock and the exact train/validation inputs."""
    lock = (root / "dvc.lock").read_bytes()
    raw = yaml.safe_load((root / "data/raw.dvc").read_text(encoding="utf-8"))
    tags = {
        "dvc_lock_sha256": hashlib.sha256(lock).hexdigest(),
        "dvc_lock_scope": "snapshot_before_training; see artifact provenance/dvc.lock",
        "raw_dvc_hash": raw["outs"][0]["md5"],
    }
    digest = hashlib.sha256()
    for name in ("X_train.csv", "y_train.csv", "X_val.csv", "y_val.csv", "feature_columns.json"):
        content = (processed / name).read_bytes()
        tags[f"data_sha256.{name}"] = hashlib.sha256(content).hexdigest()
        digest.update(name.encode())
        digest.update(content)
    tags["training_data_sha256"] = digest.hexdigest()
    return tags, lock


def run_experiments(cfg: dict[str, Any]) -> dict:
    """Compare on validation and export the best candidate without promotion."""
    from src.models.train import select_threshold

    processed = project_path(cfg, "processed")
    frames = {
        name: pd.read_csv(processed / f"{name}.csv", index_col="SK_ID_CURR")
        for name in ("X_train", "y_train", "X_val", "y_val")
    }
    xt, xv = frames["X_train"], frames["X_val"]
    yt, yv = frames["y_train"].TARGET, frames["y_val"].TARGET
    expected = json.loads((processed / "feature_columns.json").read_text(encoding="utf-8"))
    if (
        list(xt.columns) != expected
        or list(xv.columns) != expected
        or EXCLUDED.intersection(xt.columns)
    ):
        raise ValueError("Invalid feature contract.")
    if not xt.index.equals(yt.index) or not xv.index.equals(yv.index):
        raise ValueError("Feature/target indices must align.")
    if not xt.index.is_unique or not xv.index.is_unique or len(xt.index.intersection(xv.index)):
        raise ValueError("Training and validation IDs must be unique and disjoint.")
    provenance, lock = data_provenance(ROOT, processed)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    experiment = mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    reports = project_path(cfg, "reports")
    models = project_path(cfg, "models")
    models.mkdir(parents=True, exist_ok=True)
    results, bundles = [], {}
    try:
        importlib.metadata.version("xgboost")
        xgboost_distribution = "xgboost"
    except importlib.metadata.PackageNotFoundError:
        xgboost_distribution = "xgboost-cpu"
    requirements = [
        f"{package}=={importlib.metadata.version(package)}"
        for package in (
            "scikit-learn",
            "numpy",
            "pandas",
            "lightgbm",
            xgboost_distribution,
            "mlflow",
        )
    ]
    for name in cfg["train"]["candidate_models"]:
        print(f"Training {name} on {len(xt)} rows / {len(xt.columns)} features", flush=True)
        with mlflow.start_run(run_name=f"phase4-{name}") as run:
            mlflow.set_tags(
                {
                    **provenance,
                    "phase": "4",
                    "model_family": name,
                    "gate_status": "not_evaluated",
                    "selection_split": "validation",
                    "positive_definition": "1=default/refusal; acceptance=1-prediction",
                    "test_split": "untouched",
                    "sensitive_features": "audit_only",
                }
            )
            pipeline = build_pipeline(name, cfg)
            mlflow.log_params(
                {
                    **{
                        f"estimator.{key}": value
                        for key, value in pipeline[-1].get_params().items()
                    },
                    **{
                        f"threshold.{key}": value
                        for key, value in cfg["train"]["threshold"].items()
                    },
                    "sample_frac": cfg["data"]["sample_frac"],
                    "random_state": cfg["data"]["random_state"],
                    "feature_count": len(xt.columns),
                    "train_rows": len(xt),
                    "validation_rows": len(xv),
                    "drop_sensitive": True,
                    "imputation": "train_median",
                }
            )
            start = time.perf_counter()
            pipeline.fit(xt, yt)
            duration = time.perf_counter() - start
            probabilities = pipeline.predict_proba(xv)[:, 1]
            threshold = select_threshold(yv, probabilities, cfg)
            metrics = performance_metrics(yv, probabilities, threshold, cfg["train"]["threshold"])
            metrics["training_seconds"] = duration
            mlflow.log_metrics(metrics)
            # Preserve exact lock bytes: log_text normalizes newlines on Windows.
            with tempfile.TemporaryDirectory(prefix="credit-lock-") as temporary:
                snapshot = Path(temporary) / "dvc.lock"
                snapshot.write_bytes(lock)
                mlflow.log_artifact(str(snapshot), "provenance")
            mlflow.log_dict(cfg, "params.json")
            mlflow.log_dict(list(xt.columns), "feature_columns.json")
            mlflow.log_dict(provenance, "provenance/data_hashes.json")
            mlflow.log_artifact(str(processed / "feature_encoder.pkl"), "features")
            mlflow.log_artifact(str(processed / "category_vocabulary.json"), "features")
            mlflow.log_dict(metrics, "validation_metrics.json")
            example = xt.head(5)
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                input_example=example,
                signature=infer_signature(example, pipeline.predict_proba(example)),
                pyfunc_predict_fn="predict_proba",
                pip_requirements=requirements,
                metadata={"threshold": threshold, "positive_label": 1, **provenance},
            )
            row = {
                "model": name,
                "run_id": run.info.run_id,
                "model_uri": model_info.model_uri,
                **metrics,
            }
            results.append(row)
            bundles[name] = {
                "pipeline": pipeline,
                "columns": list(xt.columns),
                "threshold": threshold,
                "run_id": run.info.run_id,
                "model_uri": model_info.model_uri,
                "metrics": {
                    **metrics,
                    **provenance,
                    "phase": 4,
                    "model": name,
                    "promotion_status": "not_evaluated_not_promoted",
                },
            }
            (models / f"{name}.pkl").write_bytes(pickle.dumps(bundles[name]))
            print(json.dumps(row), flush=True)
    best = max(results, key=lambda row: row["validation_auc"])
    # Compatibility path for the existing simulation; this is a candidate, not champion.
    (models / "baseline.pkl").write_bytes(pickle.dumps(bundles[best["model"]]))
    write_json(reports / "baseline_metrics.json", bundles[best["model"]]["metrics"])
    comparison = {
        "phase": 4,
        "experiment_id": experiment.experiment_id,
        "results": results,
        "selected_model": best["model"],
        "selection_criterion": "maximum validation ROC-AUC",
        "promotion_status": "not_evaluated_not_promoted",
        "provenance": provenance,
    }
    write_json(reports / "model_comparison.json", comparison)
    pd.DataFrame(results).to_csv(reports / "model_comparison.csv", index=False)
    return comparison
