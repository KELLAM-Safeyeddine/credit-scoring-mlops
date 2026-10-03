"""Package frozen development oracles, explain them and freeze one evaluation plan."""

import copy
import hashlib
import importlib.metadata
import json
import pickle
import shutil
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from src.config import ROOT, load_config, project_path, write_json
from src.models.constrained_study import deterministic_scores
from src.models.explain import create_shap_artifacts
from src.models.policies import ScoringPolicy
from src.models.probability_mixture import ProbabilityMixture
from src.models.promote import candidate_evidence, quality_gate
from src.models.qualification_protocol import protocol_context, verify_memberships


def prepare() -> dict:
    """Create SHAP and an immutable plan without reading test observations or targets."""
    settings = load_config()["ensemble_qualification"]
    source = ROOT / settings["source_params"]
    cfg = load_config(source)
    ledger, parent = protocol_context(cfg)
    if parent is None or ledger.exists():
        raise ValueError("Parent holdout must remain unopened.")
    snapshot = ROOT / settings["snapshot"]
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != settings["snapshot_sha256"]:
        raise ValueError("Development snapshot differs from the selected frozen model.")
    original = pickle.loads(snapshot.read_bytes())
    if original["mode"] != "probability_average":
        raise ValueError("Only the selected probability ensemble can be explained here.")
    output = ROOT / settings["output"]
    if output.exists():
        raise ValueError("Preserve the existing preparation and evaluation plan.")
    processed = project_path(cfg, "processed")
    train = pd.read_csv(processed / "X_train.csv", index_col="SK_ID_CURR")
    validation = pd.read_csv(processed / "X_val.csv", index_col="SK_ID_CURR")
    verify_memberships(cfg, parent, {"train": train, "val": validation})
    for name in ("X_train.csv", "y_train.csv", "X_val.csv", "y_val.csv", "feature_columns.json"):
        if (
            hashlib.sha256((processed / name).read_bytes()).hexdigest()
            != original["provenance"][f"data_sha256.{name}"]
        ):
            raise ValueError("Development inputs changed after candidate selection.")
    if list(train) != original["columns"] or list(validation) != original["columns"]:
        raise ValueError("Frozen ordered feature contract differs.")
    background = train.sample(
        n=settings["background_rows"], random_state=cfg["data"]["random_state"]
    )
    active = original["reduction"].weights_
    mixture = ProbabilityMixture(
        tuple(original["reduction"].predictors_.loc[active.index]),
        tuple(active),
        original["imputer"].transform(background),
    )
    pipeline = Pipeline([("imputer", original["imputer"]), ("model", mixture)])
    expected = deterministic_scores(
        original["reduction"], original["imputer"].transform(validation), original["mode"]
    )
    np.testing.assert_allclose(
        pipeline.predict_proba(validation)[:, 1], expected, atol=1e-12, rtol=0
    )
    output.mkdir(parents=True)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment("credit-scoring-constrained-development")
    with mlflow.start_run(run_name="frozen-ensemble-shap-qualification") as run:
        mlflow.set_tags(
            {
                **original["provenance"],
                "parent_run_id": original["run_id"],
                "test_state": "untouched",
                "qualification_state": "shap_preparation_only",
                "development_snapshot_sha256": settings["snapshot_sha256"],
            }
        )
        metrics = {
            **original["metrics"],
            "requires_sensitive_at_inference": False,
            "randomized_decision": False,
        }
        mlflow.log_metrics(
            {key: value for key, value in metrics.items() if not isinstance(value, bool)}
        )
        mlflow.log_dict(original["fairness"], "fairness_report.json")
        mlflow.log_artifact(str(processed / "feature_columns.json"))
        for name in ("feature_encoder.pkl", "category_vocabulary.json"):
            mlflow.log_artifact(str(processed / name), "features")
        mlflow.log_artifact(str(processed.parent / "processed.dvc"), "provenance")
        requirements = [
            f"{package}=={importlib.metadata.version(package)}"
            for package in ("scikit-learn", "numpy", "pandas", "lightgbm", "mlflow")
        ]
        example = validation.head(3).astype(float)
        info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="frozen_probability_ensemble",
            input_example=example,
            signature=infer_signature(example, pipeline.predict_proba(example)),
            pyfunc_predict_fn="predict_proba",
            pip_requirements=requirements,
            metadata={
                "threshold": original["threshold"],
                "positive_label": 1,
                "score_semantics": "uncalibrated weighted oracle probabilities",
            },
        )
        bundle = {
            "pipeline": pipeline,
            "policy": ScoringPolicy(pipeline, original["threshold"]),
            "columns": original["columns"],
            "threshold": original["threshold"],
            "run_id": run.info.run_id,
            "model_uri": info.model_uri,
            "metrics": metrics,
            "fairness_report": original["fairness"],
            "independent_test_required": True,
            "selected_training_run_id": original["run_id"],
        }
        report = create_shap_artifacts(
            bundle, validation, output / "shap", settings["shap_rows"], cfg["data"]["random_state"]
        )
        mlflow.log_artifacts(str(output / "shap"), "shap")
        mlflow.set_tag("shap_artifacts_present", "true")
        mlflow.set_tag("independent_test_required", "true")
    content = pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
    (output / "candidate.pkl").write_bytes(content)
    client = mlflow.MlflowClient()
    evidence = candidate_evidence(bundle, client)
    validation_evidence = {
        key: value for key, value in evidence.items() if key != "independent_test_required"
    }
    passed, rules = quality_gate(validation_evidence, None, cfg)
    if not passed:
        raise ValueError("Validation and actual SHAP evidence must pass before plan freezing.")
    pending_passed, pending_rules = quality_gate(evidence, None, cfg)
    assert not pending_passed and not pending_rules["independent_test"]["passed"]
    write_json(output / "validation_gate.json", {"passed": passed, "rules": rules})
    write_json(
        output / "promotion_gate_pending.json", {"passed": pending_passed, "rules": pending_rules}
    )
    frozen = copy.deepcopy(cfg)
    frozen["qualification"]["protocol_manifest"] = str(
        (output / "protocol.json").relative_to(ROOT)
    ).replace("\\", "/")
    frozen["qualification"]["require_independent_test"] = True
    frozen["paths"]["audit_models"] = settings["output"]
    frozen["paths"]["reports"] = settings["output"] + "/reports"
    parent_path = ROOT / cfg["qualification"]["protocol_manifest"]
    shutil.copytree(parent_path.parent / "memberships", output / "memberships")
    (output / "params.yaml").write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    sources = [
        "src/models/probability_mixture.py",
        "src/models/explain.py",
        "src/models/promote.py",
        "src/models/qualify_holdout.py",
        "src/models/qualification_protocol.py",
    ]
    plan = {
        "protocol_id": settings["plan_id"],
        "scope": "one_frozen_candidate_after_development_before_test",
        "evaluation_parent_params": settings["source_params"],
        "evaluation_parent_manifest_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "membership_sha256": parent["membership_sha256"],
        "prospective_config_sha256": hashlib.sha256(
            json.dumps(frozen, sort_keys=True).encode()
        ).hexdigest(),
        "frozen_candidate_sha256": hashlib.sha256(content).hexdigest(),
        "frozen_source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources
        },
        "candidate_run_id": bundle["run_id"],
        "frozen_threshold": bundle["threshold"],
        "source_snapshot_sha256": settings["snapshot_sha256"],
        "processed_dvc_hash": yaml.safe_load((processed.parent / "processed.dvc").read_text())[
            "outs"
        ][0]["md5"],
        "test_evaluations_allowed": 1,
        "test_retuning_allowed": False,
        "test_evaluated": False,
        "original_six_trial_protocol_changed": False,
        "shared_holdout_ledger": str(ledger.relative_to(ROOT)).replace("\\", "/"),
    }
    write_json(output / "protocol.json", plan)
    resolved, _ = protocol_context(frozen)
    assert resolved == ledger and not ledger.exists()
    client.log_dict(bundle["run_id"], plan, "qualification/evaluation_plan.json")
    result = {
        "run_id": bundle["run_id"],
        "model_uri": bundle["model_uri"],
        "validation_shap_gate_passed": True,
        "promotion_gate_passed": False,
        "test_evaluated": False,
        "shared_holdout_ledger": plan["shared_holdout_ledger"],
        "background_rows": len(background),
        "shap_rows": report["sample_rows"],
        "shap_reconstruction_error": report["max_probability_reconstruction_error"],
    }
    write_json(output / "execution.json", result)
    return result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(prepare(), indent=2))
