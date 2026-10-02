"""Verify real mitigation decisions, SHAP additivity and persisted MLflow artifacts."""

import json
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.models.fairness import audit_fairness
from src.models.policies import ScoringPolicy


def main() -> None:
    """Require at least one useful group-free mitigation within two AUC points."""
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    client = mlflow.MlflowClient()
    reports = project_path(cfg, "audit_reports")
    comparison = json.loads((reports / "mitigation_comparison.json").read_text(encoding="utf-8"))
    processed = project_path(cfg, "processed")
    x = pd.read_csv(processed / "X_val.csv", index_col="SK_ID_CURR")
    y = pd.read_csv(processed / "y_val.csv", index_col="SK_ID_CURR").TARGET
    sensitive = pd.read_csv(processed / "sensitive_val.csv", index_col="SK_ID_CURR")
    for row in comparison["results"]:
        run = client.get_run(row["run_id"])
        assert run.info.status == "FINISHED"
        assert run.data.tags["test_split"] == "untouched"
        assert run.data.metrics["validation_auc"] == row["validation_auc"]
        bundle = pickle.loads(
            (project_path(cfg, "audit_models") / f"{row['method']}.pkl").read_bytes()
        )
        policy = bundle["policy"]
        decisions = policy.predict(x, sensitive)
        np.testing.assert_array_equal(decisions, policy.predict(x, sensitive))
        report = audit_fairness(y, decisions, sensitive, cfg["fairness"], policy.scores(x))
        assert report == bundle["fairness_report"]
        assert client.list_artifacts(row["run_id"], "policy")
        if row["model_uri"]:
            loaded = mlflow.sklearn.load_model(row["model_uri"])
            np.testing.assert_allclose(
                loaded.predict_proba(x.head(20))[:, 1], policy.scores(x.head(20)), atol=1e-12
            )
    selected = comparison["selected_group_free_candidate"]
    assert selected is not None, "No group-free candidate meets phase-five validation."
    row = next(row for row in comparison["results"] if row["method"] == selected)
    assert row["passes_phase5"] and row["utility_passes"]
    assert row["auc_loss_vs_baseline"] < cfg["fairness"]["max_auc_loss_mitigation"]
    bundle = pickle.loads((project_path(cfg, "audit_models") / "candidate.pkl").read_bytes())
    assert isinstance(bundle["policy"], ScoringPolicy)
    np.testing.assert_array_equal(
        bundle["policy"].predict(x), bundle["policy"].predict(x, sensitive)
    )
    for name in ("baseline", "candidate"):
        local = json.loads(
            (project_path(cfg, "shap_reports") / name / "shap_report.json").read_text(
                encoding="utf-8"
            )
        )
        assert local["max_probability_reconstruction_error"] < 1e-6
        filenames = {
            artifact.path.rsplit("/", 1)[-1]
            for artifact in client.list_artifacts(local["model_run_id"], "shap")
        }
        assert {
            "summary.png",
            "bar.png",
            "waterfall_accepted.png",
            "waterfall_refused.png",
            "waterfall_borderline.png",
            "shap_report.json",
        }.issubset(filenames)
    print(
        f"Verified {len(comparison['results'])} policies; selected {selected}, "
        f"AUC loss={row['auc_loss_vs_baseline']:.4f}; SHAP artifacts present."
    )


if __name__ == "__main__":
    main()
