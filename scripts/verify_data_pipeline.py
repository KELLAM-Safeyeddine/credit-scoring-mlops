"""Verify produced contracts and snapshot generated-file hashes for reproducibility."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.extract_data import sha256_file
from src.config import ROOT, load_config, project_path, write_json
from src.features.build_features import EXCLUDED


def snapshot(cfg: dict) -> dict[str, str]:
    """Hash every generated file, excluding all raw data."""
    result = {}
    for key in ("interim", "processed", "production", "models", "reports"):
        directory = project_path(cfg, key)
        for file in sorted(directory.rglob("*")):
            if file.is_file():
                result[str(file.relative_to(ROOT)).replace("\\", "/")] = sha256_file(file)
    return result


def verify(cfg: dict) -> dict:
    """Check IDs, features, labels, sensitive data and production contracts."""
    processed = project_path(cfg, "processed")
    columns = json.loads((processed / "feature_columns.json").read_text(encoding="utf-8"))
    feature_manifest = json.loads((processed / "feature_manifest.json").read_text(encoding="utf-8"))
    if EXCLUDED.intersection(columns):
        raise ValueError("Forbidden model columns.")
    sets = {}
    counts = {}
    for name in ("train", "val", "test", "production"):
        x = pd.read_csv(processed / f"X_{name}.csv", index_col="SK_ID_CURR")
        y = pd.read_csv(processed / f"y_{name}.csv", index_col="SK_ID_CURR")
        sensitive = pd.read_csv(processed / f"sensitive_{name}.csv", index_col="SK_ID_CURR")
        if list(x.columns) != columns or not x.index.equals(y.index):
            raise ValueError(f"Feature/label contract failure in {name}.")
        if not x.index.equals(sensitive.index) or not x.index.is_unique:
            raise ValueError(f"Audit alignment/ID failure in {name}.")
        if not set(y.TARGET.unique()) <= {0, 1}:
            raise ValueError(f"Unexpected labels in {name}.")
        if np.isinf(x.to_numpy()).any():
            raise ValueError(f"Infinite feature values in {name}.")
        expected_missing = feature_manifest.get("missing_values", {}).get(name)
        if expected_missing is not None:
            actual_missing = {key: int(value) for key, value in x.isna().sum().items() if value}
            if actual_missing != expected_missing:
                raise ValueError(f"Unexpected missing values in {name}.")
        sets[name] = set(x.index)
        counts[name] = len(x)
    for first, second in combinations(sets, 2):
        if sets[first] & sets[second]:
            raise ValueError(f"ID leakage: {first}/{second}.")
    output = project_path(cfg, "production")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    production_ids = set()
    for batch in manifest["batches"]:
        label = batch["batch_id"]
        features = pd.read_csv(output / f"X_{label}.csv", index_col="SK_ID_CURR")
        predictions = pd.read_csv(output / f"predictions_{label}.csv", index_col="SK_ID_CURR")
        audit = pd.read_csv(output / f"sensitive_{label}.csv", index_col="SK_ID_CURR")
        truth = pd.read_csv(output / f"ground_truth/y_{label}.csv", index_col="SK_ID_CURR")
        if list(features.columns) != columns:
            raise ValueError(f"Batch {label}: inconsistent feature schema.")
        if any(not features.index.equals(item.index) for item in (predictions, audit, truth)):
            raise ValueError(f"Batch {label}: alignment failure.")
        if production_ids & set(features.index):
            raise ValueError("Repeated production IDs across batches.")
        production_ids.update(features.index)
    if production_ids != sets["production"]:
        raise ValueError("Production batches do not cover the reserved population exactly.")
    result = {
        "rows": counts,
        "feature_count": len(columns),
        "production_batches": len(manifest["batches"]),
        "checks": "disjoint IDs, aligned X/y/audit, identical columns, complete production reserve",
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    cfg = load_config()
    verify(cfg)
    hashes = snapshot(cfg)
    if args.snapshot:
        write_json(args.snapshot, hashes)
    if args.compare:
        before = json.loads(args.compare.read_text(encoding="utf-8"))
        if before != hashes:
            changed = sorted(
                key for key in before.keys() | hashes.keys() if before.get(key) != hashes.get(key)
            )
            raise SystemExit(f"Reproducibility failure: {changed}")
        print(f"Reproducibility passed: {len(hashes)} files identical by SHA-256.")
