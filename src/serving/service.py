"""Freeze accepted artifacts and reuse training transformations at inference."""

import hashlib
import hmac
import json
import pickle
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import mlflow.models
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts

from src.features.contracts import EXCLUDED
from src.models.explain import CreditExplainer
from src.models.promote import SHAP_FILES, champion_version, quality_gate


@dataclass(frozen=True)
class Snapshot:
    """One coherent model version for an entire request."""

    version: str
    run_id: str
    created_at: str
    pipeline: Any
    encoder: Any
    columns: list[str]
    threshold: float
    evidence: dict
    explainer: CreditExplainer

    def transform(self, inputs: dict) -> tuple[pd.DataFrame, list[str]]:
        """Supply missing inputs without fitting or fabricating history."""
        values = dict(inputs)
        employed = values.get("DAYS_EMPLOYED")
        values["EMPLOYED_ANOMALY"] = int(employed == 365243)
        if employed == 365243:
            values["DAYS_EMPLOYED"] = None
        for output, numerator, denominator in (
            ("CREDIT_INCOME_RATIO", "AMT_CREDIT", "AMT_INCOME_TOTAL"),
            ("ANNUITY_INCOME_RATIO", "AMT_ANNUITY", "AMT_INCOME_TOTAL"),
            ("CREDIT_TERM", "AMT_CREDIT", "AMT_ANNUITY"),
        ):
            values[output] = values[numerator] / values[denominator]
        known = [
            values[f"EXT_SOURCE_{i}"]
            for i in (1, 2, 3)
            if values.get(f"EXT_SOURCE_{i}") is not None
        ]
        values["EXT_SOURCE_MEAN"] = float(np.mean(known)) if known else None
        names = self.encoder.numeric_columns + self.encoder.categorical_columns
        missing = [name for name in names if values.get(name) is None]
        frame = pd.DataFrame([{name: values.get(name) for name in names}])
        for name in self.encoder.numeric_columns:
            frame[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
        transformed = self.encoder.transform(frame)
        if list(transformed.columns) != self.columns:
            raise ValueError("Inference columns differ from the registered contract.")
        return transformed, missing


class RegistryLoader:
    """Load only a champion with accepted, consistent promotion evidence."""

    def __init__(self, uri: str, name: str) -> None:
        self.client = MlflowClient(tracking_uri=uri)
        self.name = name

    def load(self) -> Snapshot:
        """Resolve the alias once and load its explicit source model."""
        version = champion_version(self.client, self.name)
        if version is None or version.tags.get("gate_status") != "accepted":
            raise ValueError("No accepted champion is available.")
        run_id = version.run_id
        run = self.client.get_run(run_id)
        if run.info.status != "FINISHED":
            raise ValueError("Champion source run is not finished.")

        def artifact(name: str) -> Path:
            return Path(self.client.download_artifacts(run_id, name))

        evidence = json.loads(artifact("promotion/evidence.json").read_text(encoding="utf-8"))
        gate = json.loads(artifact("promotion/quality_gate.json").read_text(encoding="utf-8"))
        cfg = json.loads(artifact("promotion/params.json").read_text(encoding="utf-8"))
        if (
            evidence.get("run_id") != run_id
            or gate.get("run_id") != run_id
            or gate.get("version") != str(version.version)
            or gate.get("passed") is not True
            or gate.get("gate_status") != "accepted"
            or gate.get("registered_model_name") != self.name
            or not gate.get("rules")
            or not all(rule.get("passed") is True for rule in gate["rules"].values())
            or not quality_gate(evidence, None, cfg)[0]
        ):
            raise ValueError("Champion promotion evidence is inconsistent or rejected.")
        columns = json.loads(artifact("feature_columns.json").read_text(encoding="utf-8"))
        encoder = pickle.loads(artifact("features/feature_encoder.pkl").read_bytes())
        if columns != encoder.columns or EXCLUDED.intersection(columns):
            raise ValueError("Unsafe or inconsistent feature contract.")
        files = {
            item.path.rsplit("/", 1)[-1]
            for item in self.client.list_artifacts(run_id, "shap")
            if not item.is_dir and item.file_size and item.file_size > 0
        }
        if not SHAP_FILES.issubset(files):
            raise ValueError("Champion SHAP artifacts are missing.")
        report = json.loads(artifact("shap/shap_report.json").read_text(encoding="utf-8"))
        if (
            report.get("model_run_id") != run_id
            or report.get("feature_count") != len(columns)
            or not 0 <= report.get("max_probability_reconstruction_error", float("inf")) < 1e-6
        ):
            raise ValueError("Champion SHAP evidence is inconsistent.")
        model_id = evidence["model_uri"].removeprefix("models:/")
        if model_id.startswith("m-"):
            model = self.client.get_logged_model(model_id)
            if model.source_run_id != run_id:
                raise ValueError("Logged model does not belong to the promoted run.")
            model_path = model.artifact_location
        else:
            model_path = version.source
        local = download_artifacts(artifact_uri=model_path, tracking_uri=self.client.tracking_uri)
        metadata = mlflow.models.Model.load(str(Path(local) / "MLmodel"))
        pipeline = mlflow.sklearn.load_model(local)
        threshold = float(metadata.metadata["threshold"])
        if (
            not 0 < threshold < 1
            or pipeline.n_features_in_ != len(columns)
            or list(pipeline.classes_) != [0, 1]
            or evidence["metrics"].get("threshold") != threshold
        ):
            raise ValueError("Invalid threshold or trained dimension.")
        parent_id = run.data.tags.get("parent_run_id")
        parent = self.client.get_run(parent_id) if parent_id else run
        evidence = {**evidence, "dvc_lock_sha256": parent.data.tags.get("dvc_lock_sha256")}
        return Snapshot(
            str(version.version),
            run_id,
            datetime.fromtimestamp(version.creation_timestamp / 1000, UTC).isoformat(),
            pipeline,
            encoder,
            columns,
            threshold,
            evidence,
            CreditExplainer({"pipeline": pipeline, "columns": columns}),
        )


class ScoringService:
    """Swap snapshots atomically and persist model-aligned events."""

    def __init__(self, loader: RegistryLoader, log_path: Path, key: str) -> None:
        if len(key) < 32:
            raise ValueError("API_LOG_HMAC_KEY must contain at least 32 characters.")
        self.loader, self.log_path, self.key = loader, log_path, key.encode()
        self.snapshot: Snapshot | None = None
        self.error: str | None = None
        self.lock = threading.RLock()

    def reload(self) -> bool:
        """Keep the last verified model when the new alias cannot be loaded."""
        with self.lock:
            try:
                candidate = self.loader.load()
            except Exception:
                self.error = "champion_unavailable_or_invalid"
                return False
            self.snapshot, self.error = candidate, None
            return True

    def score(self, customer_id: str, inputs: dict, explain: bool) -> dict:
        """Use the same snapshot for prediction, explanation and logging."""
        snapshot = self.snapshot
        if snapshot is None:
            raise LookupError("No accepted champion is available.")
        x, missing = snapshot.transform(inputs)
        probability = float(snapshot.pipeline.predict_proba(x)[0, 1])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid model output.")
        decision = int(probability >= snapshot.threshold)
        result = {
            "request_id": str(uuid4()),
            "model_version": snapshot.version,
            "default_score": probability,
            "predicted_default": decision,
            "decision": "refus" if decision else "acceptation",
            "threshold": snapshot.threshold,
            "missing_inputs": missing,
        }
        if explain:
            result["reasons"] = snapshot.explainer.top_reasons(x, min(3, len(x.columns)))
        event = {
            **result,
            "timestamp": datetime.now(UTC).isoformat(),
            "customer_pseudonym": hmac.new(
                self.key, customer_id.encode(), hashlib.sha256
            ).hexdigest(),
            "features": {
                name: None if pd.isna(value) else float(value) for name, value in x.iloc[0].items()
            },
            "model_run_id": snapshot.run_id,
        }
        with self.lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False, ensure_ascii=False) + "\n")
        return result
