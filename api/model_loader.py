"""Environment adapter for the Registry inference service."""

import os
from pathlib import Path

from src.config import ROOT, load_config
from src.serving.service import RegistryLoader, ScoringService


def create_service() -> ScoringService:
    """Read Registry configuration and require a private pseudonymization key."""
    cfg = load_config()
    return ScoringService(
        RegistryLoader(
            os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]),
            cfg["mlflow"]["registered_model_name"],
        ),
        Path(os.getenv("API_LOG_PATH", str(ROOT / cfg["api"]["log_path"]))),
        os.getenv("API_LOG_HMAC_KEY", ""),
    )
