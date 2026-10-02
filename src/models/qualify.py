"""Bounded Home Credit search on frozen train/validation data; test remains unread."""

import argparse
import copy
import importlib.metadata
import json
import pickle
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from src.config import ROOT, load_config, project_path, write_json
from src.features.contracts import EXCLUDED
from src.features.proxy_mask import ProxyMask
from src.models.evaluate import performance_metrics
from src.models.experiments import build_pipeline, data_provenance
from src.models.fairness import audit_fairness, intersection_groups, reweighing_weights
from src.models.mitigate import select_fair_threshold
from src.models.policies import ScoringPolicy


def eligible(metrics: dict, fairness: dict, cfg: dict) -> bool:
    """Screen actual operating points; this is not a substitute for the SHAP gate."""
    utility = cfg["fairness"]["mitigation"]
    return bool(
        metrics["validation_auc"] >= cfg["quality_gate"]["min_auc"]
        and fairness["passes_thresholds"]
        and metrics["validation_recall"] >= utility["min_recall"]
        and utility["min_acceptance_rate"]
        <= metrics["validation_acceptance_rate"]
        <= utility["max_acceptance_rate"]
    )


def specifications(cfg: dict) -> list[tuple[dict, float, dict]]:
    """Reject excessive or unsafe search definitions before fitting anything."""
    settings = cfg["qualification"]
    limit = settings["max_trials"]
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 30:
        raise ValueError("Qualification search must be bounded to 1–30 trials.")
    allowed = {"n_estimators", "num_leaves", "learning_rate", "min_child_samples", "reg_lambda"}
    specs = []
    profiles = settings.get("proxy_profiles", [{"name": "unchanged", "columns": []}])
    if not profiles or any(
        set(profile) != {"name", "columns"}
        or not isinstance(profile["name"], str)
        or not isinstance(profile["columns"], list)
        or not all(isinstance(column, str) for column in profile["columns"])
        for profile in profiles
    ):
        raise ValueError("Invalid proxy profiles.")
    for parameters in settings["configurations"]:
        if set(parameters) - allowed:
            raise ValueError("Unsupported qualification hyperparameter.")
        for power in settings["reweighing_powers"]:
            if (
                isinstance(power, bool)
                or not isinstance(power, (int, float))
                or not 0 <= power <= 3
            ):
                raise ValueError("Invalid reweighing power.")
            for profile in profiles:
                specs.append((dict(parameters), float(power), copy.deepcopy(profile)))
    if not specs or len(specs) > limit:
        raise ValueError("Qualification trial count exceeds the configured bound.")
    return specs


def search(cfg: dict, output: Path) -> dict:
    """Track bounded candidates and freeze the best eligible validation policy."""
    specs = specifications(cfg)
    processed = project_path(cfg, "processed")
    frames = {
        name: pd.read_csv(processed / f"{name}.csv", index_col="SK_ID_CURR")
        for name in ("X_train", "y_train", "sensitive_train", "X_val", "y_val", "sensitive_val")
    }
    xt, xv = frames["X_train"], frames["X_val"]
    yt, yv = frames["y_train"].TARGET, frames["y_val"].TARGET
    st, sv = frames["sensitive_train"], frames["sensitive_val"]
    columns = json.loads((processed / "feature_columns.json").read_text())
    if list(xt) != columns or list(xv) != columns or EXCLUDED.intersection(columns):
        raise ValueError("Qualification must retain the frozen feature contract.")
    if any(set(profile["columns"]) - set(columns) for _, _, profile in specs):
        raise ValueError("Unknown proxy profile columns.")
    if not yt.index.equals(xt.index) or not st.index.equals(xt.index):
        raise ValueError("Training audit alignment differs.")
    if not yv.index.equals(xv.index) or not sv.index.equals(xv.index):
        raise ValueError("Validation audit alignment differs.")
    if not xt.index.is_unique or not xv.index.is_unique or len(xt.index.intersection(xv.index)):
        raise ValueError("Qualification train/validation must be disjoint.")
    weights = reweighing_weights(yt, intersection_groups(st))
    provenance, lock = data_provenance(ROOT, processed)
    expansion_root = cfg["qualification"].get("snapshot_root")
    if expansion_root:
        expansion_root = Path(expansion_root)
        descriptor = expansion_root / "processed.dvc"
        tracked = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
        provenance["qualification_processed_dvc_hash"] = tracked["outs"][0]["md5"]
        provenance["dvc_lock_scope"] = "original pipeline; expanded data in processed.dvc artifact"
    output.mkdir(parents=True, exist_ok=False)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment("credit-scoring-qualification")
    rows, best = [], None
    requirements = [
        f"{name}=={importlib.metadata.version(name)}"
        for name in ("lightgbm", "scikit-learn", "pandas", "numpy", "mlflow")
    ]
    for number, (parameters, power, profile) in enumerate(specs, 1):
        trial = copy.deepcopy(cfg)
        trial["train"]["lightgbm"].update(parameters)
        with mlflow.start_run(run_name=f"qualification-{output.name}-{number:02d}") as run:
            mlflow.set_tags(
                {
                    **provenance,
                    "selection_split": "validation",
                    "test_split": "untouched",
                    "sensitive_features": "audit_only; train reweighing",
                    "gate_status": "not_evaluated",
                }
            )
            mlflow.log_params(
                {
                    **trial["train"]["lightgbm"],
                    "reweighing_power": power,
                    "search_max_trials": len(specs),
                    "sample_frac": cfg["data"]["sample_frac"],
                    "proxy_profile": profile["name"],
                }
            )
            pipeline = build_pipeline("lightgbm", trial)
            if profile["columns"]:
                pipeline = Pipeline(
                    [("proxy_mask", ProxyMask(tuple(profile["columns"]))), *pipeline.steps]
                )
            start = time.perf_counter()
            pipeline.fit(xt, yt, model__sample_weight=weights**power)
            duration = time.perf_counter() - start
            scores = pipeline.predict_proba(xv)[:, 1]
            threshold, feasible = select_fair_threshold(yv, scores, sv, trial)
            decisions = (scores >= threshold).astype(int)
            fairness = audit_fairness(yv, decisions, sv, trial["fairness"], scores)
            metrics = performance_metrics(yv, scores, threshold, cfg["train"]["threshold"])
            metrics.update(
                training_seconds=duration,
                requires_sensitive_at_inference=False,
                randomized_decision=False,
            )
            mlflow.log_metrics(
                {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
            )
            mlflow.log_dict(fairness, "fairness_report.json")
            mlflow.log_dict(trial, "params.json")
            mlflow.log_dict(columns, "feature_columns.json")
            mlflow.log_dict(profile, "proxy_mask.json")
            mlflow.log_artifact(str(processed / "feature_encoder.pkl"), "features")
            if expansion_root:
                for filename in ("processed.dvc", "interim.dvc", "split_manifest.json"):
                    mlflow.log_artifact(str(expansion_root / filename), "provenance/expansion")
            with tempfile.TemporaryDirectory(prefix="credit-qualification-") as temporary:
                snapshot = Path(temporary) / "dvc.lock"
                snapshot.write_bytes(lock)
                mlflow.log_artifact(str(snapshot), "provenance")
            usable = feasible and eligible(metrics, fairness, trial)
            row = {
                "trial": number,
                "run_id": run.info.run_id,
                "parameters": parameters,
                "power": power,
                "proxy_profile": profile["name"],
                "threshold": threshold,
                "validation_auc": metrics["validation_auc"],
                "eligible": bool(usable),
                "fair_threshold_feasible": feasible,
                "age_di": fairness["attributes"]["AGE_GROUP"]["disparate_impact"],
                "age_eo": fairness["attributes"]["AGE_GROUP"]["equal_opportunity_difference"],
                "gender_di": fairness["attributes"]["CODE_GENDER"]["disparate_impact"],
                "gender_eo": fairness["attributes"]["CODE_GENDER"]["equal_opportunity_difference"],
                "validation_recall": metrics["validation_recall"],
                "validation_acceptance_rate": metrics["validation_acceptance_rate"],
            }
            if usable and (best is None or row["validation_auc"] > best["validation_auc"]):
                logged = mlflow.sklearn.log_model(
                    pipeline,
                    name="model",
                    input_example=xt.head(3),
                    metadata={"threshold": threshold},
                    pyfunc_predict_fn="predict_proba",
                    pip_requirements=requirements,
                )
                bundle = {
                    "pipeline": pipeline,
                    "policy": ScoringPolicy(pipeline, threshold),
                    "threshold": threshold,
                    "columns": columns,
                    "metrics": metrics,
                    "fairness_report": fairness,
                    "run_id": run.info.run_id,
                    "model_uri": logged.model_uri,
                }
                (output / "candidate.pkl").write_bytes(pickle.dumps(bundle))
                write_json(
                    output / "selected_params.json",
                    {"parameters": parameters, "power": power, "proxy_profile": profile},
                )
                best = row
            rows.append(row)
            write_json(
                output / "search.json", {"trials": rows, "best": best, "test_split": "untouched"}
            )
            print(json.dumps(row), flush=True)
    result = {
        "trials": rows,
        "best": best,
        "test_split": "untouched",
        "status": "candidate_frozen_not_promoted" if best else "no_eligible_candidate",
        "output": str(output),
    }
    write_json(output / "search.json", result)
    return result


def main() -> None:
    """Run qualification into a new private directory without changing production aliases."""
    # MLflow writes Unicode links even when PowerShell exposes a legacy code page.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, default=ROOT / "params.yaml")
    args = parser.parse_args()
    result = search(load_config(args.params), ROOT / "runtime/qualification" / uuid4().hex[:12])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
