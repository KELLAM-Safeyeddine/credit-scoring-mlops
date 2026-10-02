"""Verify the real Registry decisions against local trusted model evidence."""

import json
import pickle
from pathlib import Path

import mlflow

from src.config import load_config, project_path
from src.models.promote import candidate_evidence, champion_version, quality_gate


def main() -> None:
    """Check decisions, stored artifacts and champion eligibility without mutation."""
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    client = mlflow.MlflowClient()
    name = cfg["mlflow"]["registered_model_name"]
    versions = list(client.search_model_versions(f"name='{name}'"))
    rows = []
    for local in ("baseline", "candidate"):
        bundle = pickle.loads((project_path(cfg, "audit_models") / f"{local}.pkl").read_bytes())
        matches = [version for version in versions if version.run_id == bundle["run_id"]]
        assert len(matches) == 1, "A retry must reuse the registered version."
        version = matches[0]
        evidence = candidate_evidence(bundle, client)
        path = client.download_artifacts(bundle["run_id"], "promotion/quality_gate.json")
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        assert version.tags["gate_status"] == result["gate_status"]
        assert json.loads(version.tags["gate_failed_rules"]) == result["failed_rules"]
        # Current phase-six candidates were evaluated before any champion existed.
        if (
            result["rules"]["auc_vs_production"]["requirement"]
            == "first promotion: no production comparator"
        ):
            passed, rules = quality_gate(evidence, None, cfg)
            assert passed == result["passed"] and rules == result["rules"]
        assert evidence["shap_verified"]
        rows.append(
            {
                "version": str(version.version),
                "candidate": local,
                "gate_status": result["gate_status"],
                "failed_rules": result["failed_rules"],
            }
        )
    champion = champion_version(client, name)
    if champion:
        assert champion.tags["gate_status"] == "accepted"
    print(
        json.dumps(
            {
                "registered_model": name,
                "versions": rows,
                "champion": str(champion.version) if champion else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
