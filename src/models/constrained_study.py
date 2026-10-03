"""Development-only cost-sensitive Fairlearn study; never opens test or promotes."""

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from fairlearn.reductions import (
    EqualizedOdds,
    ErrorRate,
    ExponentiatedGradient,
    TruePositiveRateParity,
)

from src.config import ROOT, load_config, project_path, write_json
from src.features.contracts import EXCLUDED
from src.models.evaluate import performance_metrics
from src.models.experiments import build_pipeline, data_provenance
from src.models.fairness import audit_fairness, intersection_groups
from src.models.mitigate import select_fair_threshold
from src.models.qualification_protocol import protocol_context, verify_memberships
from src.models.qualify import eligible


def deterministic_scores(reduction: Any, x: np.ndarray, mode: str) -> np.ndarray:
    """Average fitted oracle outputs; randomized-policy guarantees do not transfer."""
    if mode not in {"probability_average", "hard_vote"}:
        raise ValueError("Unknown deterministic score mode.")
    weights = reduction.weights_.to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Mixture weights must be finite, nonnegative and nonzero.")
    result = np.zeros(len(x))
    for index, weight in reduction.weights_.items():
        if weight > 0:
            model = reduction.predictors_.loc[index]
            values = (
                model.predict_proba(x)[:, 1] if mode == "probability_average" else model.predict(x)
            )
            values = np.asarray(values, dtype=float)
            if values.shape != result.shape or not np.isfinite(values).all():
                raise ValueError("Invalid oracle scores.")
            if ((values < 0) | (values > 1)).any():
                raise ValueError("Oracle scores must lie in [0, 1].")
            result += weight * values
    return result / weights.sum()


def study_trials(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate a maximum of four predeclared development fits before loading data."""
    trials = settings.get("trials")
    if trials is None:
        trials = [
            {
                "name": f"{constraint}-{scope}",
                "constraint": constraint,
                "scope": scope,
                "lightgbm": settings["lightgbm"],
                "max_iter": settings["max_iter"],
            }
            for constraint in ("TruePositiveRateParity", "EqualizedOdds")
            for scope in ("age", "intersection")
        ]
    if not isinstance(trials, list) or not 1 <= len(trials) <= 4:
        raise ValueError("Study must contain one to four fits.")
    names = set()
    allowed = {
        "n_estimators",
        "num_leaves",
        "learning_rate",
        "min_child_samples",
        "reg_lambda",
        "n_jobs",
    }
    for trial in trials:
        if set(trial) != {"name", "constraint", "scope", "lightgbm", "max_iter"}:
            raise ValueError("Invalid trial fields.")
        name = trial["name"]
        if (
            not isinstance(name, str)
            or not name
            or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for c in name
            )
            or name in names
        ):
            raise ValueError("Trial names must be unique safe identifiers.")
        names.add(name)
        if trial["constraint"] not in {"TruePositiveRateParity", "EqualizedOdds"} or trial[
            "scope"
        ] not in {"age", "intersection"}:
            raise ValueError("Unsupported constraint or audit scope.")
        iterations = trial["max_iter"]
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or not 1 <= iterations <= 20
        ):
            raise ValueError("Iteration budget must be between 1 and 20.")
        parameters = trial["lightgbm"]
        if set(parameters) != allowed:
            raise ValueError("Unsupported model parameters.")
        for key, low, high in (
            ("n_estimators", 1, 500),
            ("num_leaves", 2, 31),
            ("min_child_samples", 20, 1000),
            ("n_jobs", 1, 4),
        ):
            value = parameters[key]
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"Invalid {key} budget.")
        if (
            not 0.001 <= parameters["learning_rate"] <= 0.2
            or not 0 <= parameters["reg_lambda"] <= 100
        ):
            raise ValueError("Invalid learning rate or regularization.")
    return copy.deepcopy(trials)


def run_study(source: Path, settings: dict[str, Any], output: Path) -> dict[str, Any]:
    """Fit bounded predeclared reductions and audit two deterministic scores each."""
    trials = study_trials(settings)
    if output.exists():
        raise ValueError("Study output already exists; preserve the completed study.")
    cfg = load_config(source)
    ledger, manifest = protocol_context(cfg)
    if manifest is None or ledger.exists():
        raise ValueError("Study requires a frozen protocol with an unopened holdout.")
    processed = project_path(cfg, "processed")
    frames = {
        name: pd.read_csv(processed / f"{name}.csv", index_col="SK_ID_CURR")
        for name in ("X_train", "X_val", "y_train", "y_val", "sensitive_train", "sensitive_val")
    }
    xt, xv = frames["X_train"], frames["X_val"]
    yt, yv = frames["y_train"].TARGET, frames["y_val"].TARGET
    st, sv = frames["sensitive_train"], frames["sensitive_val"]
    columns = json.loads((processed / "feature_columns.json").read_text())
    for x, y, sensitive in ((xt, yt, st), (xv, yv, sv)):
        if list(x.columns) != columns or EXCLUDED.intersection(x.columns):
            raise ValueError("Invalid feature contract.")
        if not x.index.equals(y.index) or not x.index.equals(sensitive.index):
            raise ValueError("Misaligned development inputs.")
    if len(xt.index.intersection(xv.index)):
        raise ValueError("Train and validation must be disjoint.")
    verify_memberships(cfg, manifest, {"train": xt, "val": xv})
    model_cfg = copy.deepcopy(cfg)
    model_cfg["train"]["lightgbm"].update(trials[0]["lightgbm"])
    model_cfg["train"]["lightgbm"]["class_weight"] = None
    template = build_pipeline("lightgbm", model_cfg)
    train = template[:-1].fit_transform(xt, yt)
    validation = template[:-1].transform(xv)
    provenance, lock = data_provenance(ROOT, processed)
    cost = cfg["train"]["threshold"]
    total = cost["false_positive_cost"] + cost["false_negative_cost"]
    costs = {"fp": cost["false_positive_cost"] / total, "fn": cost["false_negative_cost"] / total}
    output.mkdir(parents=True)
    write_json(
        output / "definition.json",
        {
            "settings": settings,
            "source_protocol": manifest["protocol_id"],
            "costs": costs,
            "fits": len(trials),
            "trials": trials,
            "score_variants": 2,
            "development_only": True,
        },
    )
    (output / "dvc.lock").write_bytes(lock)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment("credit-scoring-constrained-development")
    report = {
        "development_only": True,
        "test_evaluated": False,
        "promotion_allowed": False,
        "source_protocol": manifest["protocol_id"],
        "rows": [],
    }
    for trial in trials:
        scope, name = trial["scope"], trial["name"]
        constraint_type = {
            "TruePositiveRateParity": TruePositiveRateParity,
            "EqualizedOdds": EqualizedOdds,
        }[trial["constraint"]]
        trial_cfg = copy.deepcopy(model_cfg)
        trial_cfg["train"]["lightgbm"].update(trial["lightgbm"])
        print(f"Fitting {name}", flush=True)
        reduction = ExponentiatedGradient(
            build_pipeline("lightgbm", trial_cfg)[-1],
            constraint_type(difference_bound=settings["difference_bound"]),
            objective=ErrorRate(costs=costs),
            eps=settings["eps"],
            max_iter=trial["max_iter"],
        )
        groups = st.AGE_GROUP if scope == "age" else intersection_groups(st)
        started = time.perf_counter()
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags(
                {
                    **provenance,
                    "development_only": "true",
                    "test_state": "untouched",
                    "source_protocol": manifest["protocol_id"],
                    "fairness_guarantee_for_deterministic_scores": "none",
                }
            )
            mlflow.log_params(
                {
                    **trial["lightgbm"],
                    "constraint": constraint_type.__name__,
                    "scope": scope,
                    "max_iter": trial["max_iter"],
                    "eps": settings["eps"],
                    "difference_bound": settings["difference_bound"],
                    "cost_fp": costs["fp"],
                    "cost_fn": costs["fn"],
                    "class_weight": "None",
                }
            )
            reduction.fit(train, yt, sensitive_features=groups)
            mlflow.log_metrics(
                {
                    "training_seconds": time.perf_counter() - started,
                    "oracle_calls": float(reduction.n_oracle_calls_),
                    "best_gap": float(reduction.best_gap_),
                }
            )
            for mode in ("probability_average", "hard_vote"):
                scores = deterministic_scores(reduction, validation, mode)
                threshold, feasible = select_fair_threshold(yv, scores, sv, cfg)
                metrics = performance_metrics(yv, scores, threshold, cost)
                fairness = audit_fairness(
                    yv, (scores >= threshold).astype(int), sv, cfg["fairness"]
                )
                row = {
                    "fit": name,
                    "mode": mode,
                    "run_id": run.info.run_id,
                    "threshold_feasible": feasible,
                    "screen_passed": bool(feasible and eligible(metrics, fairness, cfg)),
                    "metrics": metrics,
                    "fairness": fairness,
                }
                report["rows"].append(row)
                if row["screen_passed"]:
                    active = reduction.weights_[reduction.weights_ > 0]
                    bundle = {
                        "development_only": True,
                        "promotion_allowed": False,
                        "reduction": SimpleNamespace(
                            weights_=active.copy(),
                            predictors_=reduction.predictors_.loc[active.index].copy(),
                        ),
                        "imputer": template[0],
                        "columns": columns,
                        "mode": mode,
                        "threshold": threshold,
                        "metrics": metrics,
                        "fairness": fairness,
                        "run_id": run.info.run_id,
                        "provenance": provenance,
                    }
                    snapshot = output / f"{name}-{mode}.pkl"
                    snapshot.write_bytes(pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL))
                    restored = pickle.loads(snapshot.read_bytes())
                    restored_scores = deterministic_scores(
                        restored["reduction"],
                        restored["imputer"].transform(xv.iloc[:16]),
                        restored["mode"],
                    )
                    error = float(np.max(np.abs(restored_scores - scores[:16])))
                    if error > 1e-12:
                        raise RuntimeError("Frozen development candidate changed its scores.")
                    row["snapshot_roundtrip_max_error"] = error
                    mlflow.log_artifact(str(snapshot), "development_candidates")
                    row["snapshot"] = snapshot.name
                mlflow.log_metrics({f"{mode}.{key}": value for key, value in metrics.items()})
                write_json(output / f"{name}-{mode}.json", row)
                mlflow.log_artifact(str(output / f"{name}-{mode}.json"), "development")
                print(
                    f"{mode}: AUC={metrics['validation_auc']:.6f}; feasible={feasible}",
                    flush=True,
                )
            mlflow.log_artifact(str(output / "definition.json"), "development")
            mlflow.log_artifact(str(output / "dvc.lock"), "provenance")
            mlflow.log_artifact(str(processed.parent / "processed.dvc"), "provenance")
            write_json(output / "report.json", report)
    if ledger.exists():
        raise RuntimeError("Holdout ledger changed during development study.")
    report["status"] = (
        "development_screen_passed"
        if any(row["screen_passed"] for row in report["rows"])
        else "no_eligible_deterministic_score"
    )
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        choices=["constrained_development", "constrained_capacity_development"],
        default="constrained_development",
    )
    settings = load_config()[parser.parse_args().study]
    run_study(ROOT / settings["source_params"], settings, ROOT / settings["output"])
