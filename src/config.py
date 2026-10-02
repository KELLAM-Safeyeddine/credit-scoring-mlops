"""Load settings and resolve paths independently of the working directory."""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load YAML and check split and audit-only invariants."""
    cfg = yaml.safe_load((path or ROOT / "params.yaml").read_text(encoding="utf-8-sig"))
    data = cfg["data"]
    if not 0 < data["sample_frac"] <= 1 or not 0 < data["production_size"] < 1:
        raise ValueError("Invalid sample_frac or production_size.")
    if not 0 < data["test_size"] + data["val_size"] < 1:
        raise ValueError("Validation/test fractions must sum to less than one.")
    if min(data["test_size"], data["val_size"]) <= 0:
        raise ValueError("Validation and test sizes must be positive.")
    if not cfg["features"]["drop_sensitive"]:
        raise ValueError("Sensitive variables are audit-only: drop_sensitive must be true.")
    if cfg["features"]["mode"] != "numeric_baseline":
        raise ValueError("Only numeric_baseline is implemented in phase 2.")
    if cfg["train"]["model"] != "logistic_regression":
        raise ValueError("Multi-model experiments will be implemented in phase 4.")
    threshold = cfg["train"]["threshold"]
    if threshold["criterion"] != "min_business_cost":
        raise ValueError("Unsupported decision-threshold criterion.")
    if not 0 < threshold["min"] < threshold["max"] < 1 or threshold["steps"] < 2:
        raise ValueError("Invalid threshold search grid.")
    if min(threshold["false_negative_cost"], threshold["false_positive_cost"]) <= 0:
        raise ValueError("Business error costs must be positive.")
    if cfg["monitoring"]["n_batches"] < 1 or cfg["monitoring"]["label_delay_batches"] < 0:
        raise ValueError("Invalid production batch count or ground-truth delay.")
    return cfg


def project_path(cfg: dict[str, Any], name: str) -> Path:
    """Resolve a configured path relative to the repository."""
    return ROOT / cfg["paths"][name]


def write_json(path: Path, value: Any) -> None:
    """Write deterministic JSON without non-standard NaN values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def parse_params() -> Path:
    """Read the shared command-line configuration option."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, default=ROOT / "params.yaml")
    return parser.parse_args().params
