"""Prepare, fit, qualify and optionally promote a single frozen calibration candidate."""

import argparse
import copy
import importlib.metadata
import json
import pickle
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from src.config import ROOT, load_config, project_path, write_json
from src.data.calibration_cohort import prepare_cohort, sha256
from src.models.calibrated_explain import create_calibrated_shap
from src.models.calibration import CalibratedMixture, calibration_metrics, fit_sigmoid
from src.models.evaluate import performance_metrics
from src.models.fairness import audit_fairness
from src.models.policies import ScoringPolicy
from src.models.promote import SHAP_FILES, candidate_evidence, quality_gate, register_and_promote
from src.models.qualification_protocol import protocol_context
from src.models.qualify_holdout import reserve_holdout


def context(settings: dict):
    """Check frozen allocations, source and feature files before any evaluation."""
    output = ROOT / settings["output"]
    allocation = json.loads((output / "allocation.json").read_text())
    prepared = json.loads((output / "prepared.json").read_text())
    if allocation["settings"] != settings:
        raise ValueError("Calibration settings differ from the prospective allocation.")
    if sha256(output / "allocation.json") != prepared["allocation_sha256"]:
        raise ValueError("The prospective allocation changed.")
    if sha256(ROOT / settings["source_candidate"]) != settings["source_candidate_sha256"]:
        raise ValueError("The frozen source candidate changed.")
    for name, expected in prepared["data_sha256"].items():
        if sha256(output / "processed" / name) != expected:
            raise ValueError("Calibration or test data changed after preparation.")
    for name, expected in allocation["historical_membership_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError("Historical allocation changed.")
    for name, expected in allocation["membership_sha256"].items():
        if sha256(output / "memberships" / f"{name}.csv") != expected:
            raise ValueError("Calibration allocation changed.")
    cfg = load_config(ROOT / settings["source_params"])
    protocol_context(cfg)
    if sha256(project_path(cfg, "processed") / "feature_encoder.pkl") != prepared["encoder_sha256"]:
        raise ValueError("The frozen feature encoder changed.")
    return output, cfg, prepared


def read_split(directory: Path, name: str):
    """Require exact alignment and unique client identities."""
    x = pd.read_csv(directory / f"X_{name}.csv", index_col="SK_ID_CURR")
    y = pd.read_csv(directory / f"y_{name}.csv", index_col="SK_ID_CURR").TARGET
    sensitive = pd.read_csv(directory / f"sensitive_{name}.csv", index_col="SK_ID_CURR")
    if not x.index.is_unique or not x.index.equals(y.index) or not x.index.equals(sensitive.index):
        raise ValueError("Misaligned calibration data.")
    return x, y, sensitive


def fit(settings: dict) -> dict:
    """Fit only the calibration half and freeze the complete model before test access."""
    output, cfg, prepared = context(settings)
    if (output / "candidate.pkl").exists():
        raise ValueError("A fitted candidate already exists; use evaluate, never refit.")
    # Exclusive creation prevents adaptive retries after a completed or interrupted fit.
    with (output / "fit_started.json").open("x") as stream:
        json.dump({"allocation_sha256": prepared["allocation_sha256"], "max_fits": 1}, stream)
    original = pickle.loads((ROOT / settings["source_candidate"]).read_bytes())
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    source_run = mlflow.MlflowClient().get_run(original["run_id"])
    provenance = {
        key: value
        for key, value in source_run.data.tags.items()
        if key.startswith(("data_sha256.", "raw_dvc_hash", "training_data_sha256", "dvc_lock_"))
    }
    for name in ("X_val.csv", "y_val.csv"):
        if sha256(project_path(cfg, "processed") / name) != provenance.get(f"data_sha256.{name}"):
            raise ValueError("Source validation files differ from their tracked provenance.")
    x, y, _ = read_split(output / "processed", "calibration")
    if list(x) != original["columns"]:
        raise ValueError("Calibration feature contract differs from the source model.")
    scores = original["pipeline"].predict_proba(x)[:, 1]
    slope, intercept = fit_sigmoid(scores, y.to_numpy())
    model = CalibratedMixture(original["pipeline"][-1], slope, intercept)
    threshold = float(model.calibrate(original["threshold"]))
    pipeline = Pipeline(original["pipeline"].steps[:-1] + [("model", model)])
    coefficients = {
        "slope": slope,
        "intercept": intercept,
        "raw_threshold": original["threshold"],
        "calibrated_threshold": threshold,
        "formula": "PD = sigmoid(slope * raw_score + intercept)",
        "calibration_rows": len(x),
        "source_candidate_sha256": settings["source_candidate_sha256"],
    }
    write_json(output / "coefficients.json", coefficients)
    # The original validation is reused for the production comparator only; no tuning here.
    validation, labels, sensitive = read_split(project_path(cfg, "processed"), "val")
    predicted = pipeline.predict_proba(validation)[:, 1]
    raw = original["pipeline"].predict_proba(validation)[:, 1]
    np.testing.assert_array_equal(predicted >= threshold, raw >= original["threshold"])
    metrics = performance_metrics(labels, predicted, threshold, cfg["train"]["threshold"])
    metrics.update(requires_sensitive_at_inference=False, randomized_decision=False)
    fairness = audit_fairness(
        labels, (predicted >= threshold).astype(int), sensitive, cfg["fairness"], predicted
    )
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment("credit-scoring-calibration")
    with mlflow.start_run(run_name="sigmoid-frozen-mixture-v1") as run:
        mlflow.set_tags(
            {
                **provenance,
                "parent_run_id": original["run_id"],
                "calibration_method": settings["method"],
                "calibration_allocation_sha256": prepared["allocation_sha256"],
                "independent_test_required": "true",
            }
        )
        mlflow.log_params(coefficients)
        mlflow.log_metrics({k: v for k, v in metrics.items() if not isinstance(v, bool)})
        mlflow.log_dict(fairness, "fairness_report.json")
        for name in ("feature_columns.json",):
            mlflow.log_artifact(str(project_path(cfg, "processed") / name))
        for name in ("feature_encoder.pkl", "category_vocabulary.json"):
            mlflow.log_artifact(str(project_path(cfg, "processed") / name), "features")
        for name in ("allocation.json", "prepared.json", "coefficients.json"):
            mlflow.log_artifact(str(output / name), "calibration")
        example = validation.head(3).astype(float)
        info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="calibrated_probability_ensemble",
            input_example=example,
            signature=infer_signature(example, pipeline.predict_proba(example)),
            pyfunc_predict_fn="predict_proba",
            pip_requirements=[
                f"{package}=={importlib.metadata.version(package)}"
                for package in ("scikit-learn", "numpy", "pandas", "lightgbm", "mlflow", "scipy")
            ],
            metadata={"threshold": threshold, "positive_label": 1, "calibrated": True},
        )
        bundle = {
            "pipeline": pipeline,
            "policy": ScoringPolicy(pipeline, threshold),
            "columns": original["columns"],
            "threshold": threshold,
            "run_id": run.info.run_id,
            "model_uri": info.model_uri,
            "metrics": metrics,
            "fairness_report": fairness,
            "independent_test_required": True,
            "calibration": coefficients,
        }
        (output / "candidate.pkl").write_bytes(pickle.dumps(bundle, pickle.HIGHEST_PROTOCOL))
        create_calibrated_shap(
            bundle,
            validation,
            output / "shap",
            settings["shap_sample_size"],
            settings["random_state"],
        )
        mlflow.log_artifacts(str(output / "shap"), "shap")
    return finalize(settings)


def finalize(settings: dict) -> dict:
    """Recover completed uploads after a tracking/console failure without any refitting."""
    output, cfg, _ = context(settings)
    if (output / "frozen_candidate.json").exists():
        raise ValueError("The candidate is already frozen; its identity cannot be rewritten.")
    bundle = pickle.loads((output / "candidate.pkl").read_bytes())
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    client = mlflow.MlflowClient()
    run = client.get_run(bundle["run_id"])
    if run.info.status not in {"RUNNING", "FINISHED"}:
        raise ValueError("Failed or killed runs cannot be finalized.")
    for name in SHAP_FILES:
        remote = Path(client.download_artifacts(bundle["run_id"], f"shap/{name}"))
        if sha256(remote) != sha256(output / "shap" / name):
            raise ValueError("SHAP uploads are incomplete or inconsistent.")
    for key in ("validation_auc", "validation_recall", "validation_acceptance_rate"):
        if bundle["metrics"][key] != run.data.metrics.get(key):
            raise ValueError("Uploaded validation metrics differ from the frozen candidate.")
    if run.info.status == "RUNNING":
        client.set_terminated(bundle["run_id"], status="FINISHED")
    candidate_evidence(bundle, client)
    write_json(
        output / "frozen_candidate.json",
        {
            "candidate_sha256": sha256(output / "candidate.pkl"),
            "source_sha256": {
                path.relative_to(ROOT).as_posix(): sha256(path)
                for path in [
                    Path(__file__).resolve(),
                    ROOT / "src/models/calibration.py",
                    ROOT / "src/models/calibrated_explain.py",
                    ROOT / "src/models/evaluate.py",
                    ROOT / "src/models/fairness.py",
                    ROOT / "src/models/promote.py",
                    ROOT / "src/models/explain.py",
                    ROOT / "src/models/probability_mixture.py",
                    ROOT / "src/models/qualification_protocol.py",
                    ROOT / settings["source_params"],
                ]
            },
        },
    )
    return {"run_id": bundle["run_id"], **bundle["calibration"], "test_evaluated": False}


def evaluate(settings: dict, promote: bool = False) -> dict:
    """Evaluate once; promote only if calibration, performance and fairness all pass."""
    output, cfg, prepared = context(settings)
    frozen = json.loads((output / "frozen_candidate.json").read_text())
    if sha256(output / "candidate.pkl") != frozen["candidate_sha256"]:
        raise ValueError("The frozen calibration candidate changed.")
    for name, expected in frozen["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError("Frozen evaluation source changed.")
    bundle = pickle.loads((output / "candidate.pkl").read_bytes())
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    client = mlflow.MlflowClient()
    evidence = candidate_evidence(bundle, client)
    diagnostic = copy.deepcopy(cfg)
    diagnostic["qualification"]["require_independent_test"] = False
    pending = {
        k: v
        for k, v in evidence.items()
        if k not in ("independent_test_required", "independent_test_report")
    }
    if not quality_gate(pending, None, diagnostic)[0]:
        raise ValueError("Validation/SHAP gate failed; the fresh test remains unopened.")
    identity = {
        "candidate_sha256": frozen["candidate_sha256"],
        "allocation_sha256": prepared["allocation_sha256"],
        "run_id": bundle["run_id"],
        "data_sha256": prepared["data_sha256"],
    }
    ledger = output / "holdout_ledger.json"
    report = reserve_holdout(ledger, identity)
    if report is None:
        x, y, sensitive = read_split(output / "processed", "test")
        expected_ids = pd.read_csv(output / "memberships/test.csv").SK_ID_CURR
        calibration_ids = pd.read_csv(output / "memberships/calibration.csv").SK_ID_CURR
        if set(x.index) != set(expected_ids) or set(x.index).intersection(calibration_ids):
            raise ValueError("Test allocation is inconsistent or overlaps calibration.")
        if list(x) != bundle["columns"]:
            raise ValueError("Test feature contract differs.")
        p = bundle["pipeline"].predict_proba(x)[:, 1]
        model = bundle["pipeline"][-1]
        raw = model.mixture.predict_proba(bundle["pipeline"][:-1].transform(x))[:, 1]
        decision = p >= bundle["threshold"]
        unchanged = bool(np.array_equal(decision, raw >= bundle["calibration"]["raw_threshold"]))
        metrics = performance_metrics(y, p, bundle["threshold"], cfg["train"]["threshold"])
        raw_metrics = performance_metrics(
            y, raw, bundle["calibration"]["raw_threshold"], cfg["train"]["threshold"]
        )
        metrics.update(requires_sensitive_at_inference=False, randomized_decision=False)
        fairness = audit_fairness(y, decision.astype(int), sensitive, cfg["fairness"], p)
        baseline = calibration_metrics(y, raw, settings["reliability_bins"])
        calibrated = calibration_metrics(y, p, settings["reliability_bins"])
        calibration_rules = {
            "lower_test_brier": calibrated["brier"] < baseline["brier"],
            "lower_test_log_loss": calibrated["log_loss"] < baseline["log_loss"],
            "unchanged_test_decisions": unchanged,
            "unchanged_test_auc": abs(metrics["validation_auc"] - raw_metrics["validation_auc"])
            < 1e-12,
        }
        test_passed, rules = quality_gate(
            {**pending, "metrics": metrics, "fairness_report": fairness}, None, diagnostic
        )
        report = {
            "run_id": bundle["run_id"],
            "evaluation_split": "test",
            "rows": len(x),
            "frozen_threshold": bundle["threshold"],
            "passed": test_passed and all(calibration_rules.values()),
            "rules": rules,
            "metrics": {k.replace("validation_", "test_"): v for k, v in metrics.items()},
            "fairness_report": fairness,
            "data_sha256": {
                name: prepared["data_sha256"][name]
                for name in ("X_test.csv", "y_test.csv", "sensitive_test.csv")
            },
            "calibration": {"before": baseline, "after": calibrated, "rules": calibration_rules},
            "selection_or_retuning_allowed": False,
        }
        write_json(ledger, {"identity": identity, "status": "evaluated", "report": report})
    write_json(output / "holdout_report.json", report)
    client.log_dict(bundle["run_id"], report, "qualification/holdout_report.json")
    client.log_dict(bundle["run_id"], report, "calibration/report.json")
    for name, value in report["metrics"].items():
        if name.startswith("test_"):
            client.log_metric(bundle["run_id"], name, value)
    for stage in ("before", "after"):
        for name in ("brier", "log_loss", "ece"):
            client.log_metric(
                bundle["run_id"], f"test_{stage}_{name}", report["calibration"][stage][name]
            )
    client.set_tag(
        bundle["run_id"], "independent_test_status", "passed" if report["passed"] else "failed"
    )
    result = {
        "run_id": bundle["run_id"],
        "passed": report["passed"],
        "calibration": report["calibration"],
        "test_metrics": report["metrics"],
    }
    if promote and report["passed"]:
        evidence = candidate_evidence(bundle, client)
        result["promotion"] = register_and_promote(evidence, cfg, client, expected_pass=True)
        write_json(output / "promotion.json", result["promotion"])
    elif promote:
        client.set_tag(bundle["run_id"], "gate_status", "rejected")
        result["promotion"] = {
            "passed": False,
            "reason": "Calibration or quality gate failed; champion preserved.",
        }
    return result


def main():
    """Explicit stages make the frozen model and evaluation independently reviewable."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "fit", "finalize", "evaluate", "promote"])
    parser.add_argument("--config", type=Path, default=ROOT / "config/calibration.yaml")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    if (
        settings["method"] != "sigmoid_on_raw_score"
        or settings["max_fits"] != 1
        or settings["test_evaluations"] != 1
        or settings["preserve_decisions"] is not True
    ):
        raise ValueError(
            "Only the predeclared single-fit, decision-preserving protocol is supported."
        )
    if args.stage == "prepare":
        result = prepare_cohort(settings)
    elif args.stage == "fit":
        result = fit(settings)
    elif args.stage == "finalize":
        result = finalize(settings)
    else:
        result = evaluate(settings, promote=args.stage == "promote")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
