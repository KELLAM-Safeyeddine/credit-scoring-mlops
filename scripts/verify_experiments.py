"""Verify persisted MLflow runs, provenance and model probability round-trips."""

import hashlib
import json
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.config import load_config, project_path


def main() -> None:
    """Check the latest comparison against the live tracking server."""
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    client = mlflow.MlflowClient()
    comparison = json.loads(
        (project_path(cfg, "reports") / "model_comparison.json").read_text(encoding="utf-8")
    )
    rows = comparison["results"]
    assert {row["model"] for row in rows} == set(cfg["train"]["candidate_models"])
    x = pd.read_csv(project_path(cfg, "processed") / "X_val.csv", index_col="SK_ID_CURR", nrows=25)
    for row in rows:
        run = client.get_run(row["run_id"])
        assert run.info.status == "FINISHED"
        assert run.data.tags["gate_status"] == "not_evaluated"
        assert run.data.tags["test_split"] == "untouched"
        for key in ("validation_auc", "validation_ks", "threshold", "training_seconds"):
            assert run.data.metrics[key] == row[key]
        lock_path = client.download_artifacts(row["run_id"], "provenance/dvc.lock")
        assert (
            hashlib.sha256(Path(lock_path).read_bytes()).hexdigest()
            == run.data.tags["dvc_lock_sha256"]
        )
        loaded = mlflow.sklearn.load_model(row["model_uri"])
        bundle = pickle.loads((project_path(cfg, "models") / f"{row['model']}.pkl").read_bytes())
        np.testing.assert_allclose(
            loaded.predict_proba(x), bundle["pipeline"].predict_proba(x), atol=1e-12
        )
        pyfunc = mlflow.pyfunc.load_model(row["model_uri"])
        np.testing.assert_allclose(pyfunc.predict(x), loaded.predict_proba(x), atol=1e-12)
        assert client.list_artifacts(row["run_id"], "features")
    assert comparison["selected_model"] == max(rows, key=lambda row: row["validation_auc"])["model"]
    print(f"Verified {len(rows)} finished runs, lock hashes, encoders and probability round-trips.")


if __name__ == "__main__":
    main()
