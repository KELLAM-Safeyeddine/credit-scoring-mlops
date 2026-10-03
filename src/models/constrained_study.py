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
    result /= weights.sum()
    if ((result < -1e-12) | (result > 1 + 1e-12)).any():
        raise ValueError("Mixture score exceeds probability bounds beyond roundoff.")
    return np.clip(result, 0, 1)


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


def selection_config(cfg: dict, settings: dict) -> dict:
    """Apply stricter prospective selection margins without changing production rules."""
    selected = copy.deepcopy(cfg)
    margins = settings.get("validation_selection_margin")
    if margins is None:
        return selected
    if settings != cfg["qualification"] and settings.get("development_only") is not True:
        raise ValueError("Prospective study differs from the frozen selection plan.")
    if (
        settings["selection_score_mode"] != "probability_average"
        or settings["max_trials"] != len(study_trials(settings))
        or set(margins) != {"min_auc", "min_disparate_impact", "max_equal_opp_diff"}
        or not cfg["quality_gate"]["min_auc"] <= margins["min_auc"] <= 1
        or not cfg["fairness"]["min_disparate_impact"] <= margins["min_disparate_impact"] <= 1
        or not 0 <= margins["max_equal_opp_diff"] <= cfg["fairness"]["max_equal_opp_diff"]
    ):
        raise ValueError("Invalid prospective selection margins or budget.")
    selected["quality_gate"]["min_auc"] = margins["min_auc"]
    selected["fairness"].update(
        min_disparate_impact=margins["min_disparate_impact"],
        max_equal_opp_diff=margins["max_equal_opp_diff"],
    )
    return selected


def selected_candidate(rows: list[dict]) -> dict | None:
    """Choose once by declared AUC, recall and stable trial order; no test access."""
    admissible = [row for row in rows if row["screen_passed"]]
    if not admissible:
        return None
    best = max(
        admissible,
        key=lambda row: (row["metrics"]["validation_auc"], row["metrics"]["validation_recall"]),
    )
    return {key: best[key] for key in ("fit", "mode", "run_id", "snapshot")}


def run_study(
    source: Path,
    settings: dict[str, Any],
    output: Path,
    retry_run_id: str | None = None,
) -> dict[str, Any]:
    """Fit bounded predeclared reductions and audit two deterministic scores each."""
    trials = study_trials(settings)
    if output.exists() and retry_run_id is None:
        raise ValueError("Study output already exists; preserve the completed study.")
    cfg = load_config(source)
    screen_cfg = selection_config(cfg, settings)
    ledger, manifest = protocol_context(cfg)
    if manifest is None or ledger.exists():
        raise ValueError("Study requires a frozen protocol with an unopened holdout.")
    report = None
    retry_name = None
    if retry_run_id is not None:
        if "validation_selection_margin" not in settings or not output.exists():
            raise ValueError("A technical retry requires the original prospective study.")
        definition = json.loads((output / "definition.json").read_text())
        report = json.loads((output / "report.json").read_text())
        if (
            definition["settings"] != settings
            or report["source_protocol"] != manifest["protocol_id"]
            or (output / "technical_retry.json").exists()
        ):
            raise ValueError("Technical retry plan differs or retry budget is exhausted.")
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        failed_run = mlflow.MlflowClient().get_run(retry_run_id)
        retry_name = failed_run.data.tags.get("mlflow.runName")
        matching = [trial for trial in trials if trial["name"] == retry_name]
        if (
            failed_run.info.status != "FAILED"
            or failed_run.data.tags.get("source_protocol") != manifest["protocol_id"]
            or len(matching) != 1
            or any(row["fit"] == retry_name for row in report["rows"])
        ):
            raise ValueError("Only a failed, unreported fit can be retried.")
        expected = {
            **matching[0]["lightgbm"],
            "max_iter": matching[0]["max_iter"],
            "eps": settings["eps"],
            "difference_bound": settings["difference_bound"],
        }
        if any(failed_run.data.params.get(key) != str(value) for key, value in expected.items()):
            raise ValueError("Failed run parameters differ from the frozen trial.")
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
    output.mkdir(parents=True, exist_ok=retry_run_id is not None)
    if retry_run_id is not None:
        write_json(
            output / "technical_retry.json",
            {
                "failed_run_id": retry_run_id,
                "fit": retry_name,
                "same_configuration_and_seed": True,
                "reason": "weighted probability roundoff exceeded 1 during evaluation",
                "correction": "clip final roundoff within 1e-12; reject invalid oracles",
                "additional_configurations": 0,
                "additional_fit_attempts": 1,
            },
        )
    else:
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
                "selection_margin": settings.get("validation_selection_margin"),
            },
        )
        (output / "dvc.lock").write_bytes(lock)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment("credit-scoring-constrained-development")
    report = report or {
        "development_only": True,
        "test_evaluated": False,
        "promotion_allowed": False,
        "source_protocol": manifest["protocol_id"],
        "source_validation_reused": settings.get("development_only") is True,
        "rows": [],
    }
    for trial in trials:
        if retry_name is not None and trial["name"] != retry_name:
            continue
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
                    **({"technical_retry_of": retry_run_id} if retry_run_id else {}),
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
            training_seconds = time.perf_counter() - started
            active = reduction.weights_[reduction.weights_ > 0]
            checkpoint = {
                "reduction": SimpleNamespace(
                    weights_=active.copy(),
                    predictors_=reduction.predictors_.loc[active.index].copy(),
                ),
                "imputer": template[0],
                "columns": columns,
                "run_id": run.info.run_id,
            }
            fitted_path = output / f"{name}-fitted.pkl"
            fitted_path.write_bytes(pickle.dumps(checkpoint, protocol=pickle.HIGHEST_PROTOCOL))
            mlflow.log_artifact(str(fitted_path), "fitted_checkpoints")
            mlflow.log_metrics(
                {
                    "training_seconds": training_seconds,
                    "oracle_calls": float(reduction.n_oracle_calls_),
                    "best_gap": float(reduction.best_gap_),
                }
            )
            for mode in ("probability_average", "hard_vote"):
                scores = deterministic_scores(reduction, validation, mode)
                threshold, feasible = select_fair_threshold(yv, scores, sv, screen_cfg)
                metrics = performance_metrics(yv, scores, threshold, cost)
                fairness = audit_fairness(
                    yv, (scores >= threshold).astype(int), sv, cfg["fairness"]
                )
                selection_fairness = audit_fairness(
                    yv, (scores >= threshold).astype(int), sv, screen_cfg["fairness"]
                )
                mode_eligible = mode == settings.get("selection_score_mode", mode)
                row = {
                    "fit": name,
                    "mode": mode,
                    "run_id": run.info.run_id,
                    "threshold_feasible": feasible,
                    "screen_passed": bool(
                        mode_eligible
                        and feasible
                        and eligible(metrics, selection_fairness, screen_cfg)
                    ),
                    "eligible_score_mode": mode_eligible,
                    "selection_fairness": selection_fairness,
                    "production_screen_passed": eligible(metrics, fairness, cfg),
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
    report["selected_candidate"] = selected_candidate(report["rows"])
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        choices=[
            "constrained_development",
            "constrained_capacity_development",
            "constrained_refinement_development",
        ],
        default="constrained_development",
    )
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--retry-run-id")
    args = parser.parse_args()
    if args.protocol:
        if args.output is None:
            parser.error("--output is required with --protocol")
        cfg = load_config(args.protocol)
        run_study(args.protocol, cfg["qualification"], args.output, args.retry_run_id)
    else:
        settings = load_config()[args.study]
        run_study(ROOT / settings["source_params"], settings, ROOT / settings["output"])
