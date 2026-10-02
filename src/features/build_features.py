"""Build a numeric baseline contract without fitting transformations."""

from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, parse_params, project_path, write_json

AUDIT_ONLY = {"CODE_GENDER", "DAYS_BIRTH", "AGE_YEARS", "AGE_GROUP"}
EXCLUDED = AUDIT_ONLY | {"TARGET", "SK_ID_CURR"}


def numeric_features(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Use a train-defined column order and reject sensitive or missing columns."""
    if columns is None:
        columns = [
            name
            for name in frame.select_dtypes(include=["number", "bool"]).columns
            if name not in EXCLUDED
        ]
    if EXCLUDED.intersection(columns):
        raise ValueError("Sensitive variables, identifiers and TARGET are forbidden features.")
    if missing := sorted(set(columns) - set(frame.columns)):
        raise ValueError(f"Missing model columns: {missing}")
    return frame.loc[:, columns].replace([np.inf, -np.inf], np.nan).astype("float32")


def build_dataset(cfg: dict[str, Any]) -> dict[str, int]:
    """Export aligned X/y/audit files with the same schema for all splits."""
    output = project_path(cfg, "processed")
    output.mkdir(parents=True, exist_ok=True)
    (output / ".gitkeep").unlink(missing_ok=True)
    columns = None
    counts = {}
    for split in ("train", "val", "test", "production"):
        frame = pd.read_csv(project_path(cfg, "interim") / f"{split}.csv", index_col="SK_ID_CURR")
        features = numeric_features(frame, columns)
        if columns is None:
            columns = list(features.columns)
            write_json(output / "feature_columns.json", columns)
        features.to_csv(output / f"X_{split}.csv", index_label="SK_ID_CURR", lineterminator="\n")
        frame[["TARGET"]].to_csv(output / f"y_{split}.csv", lineterminator="\n")
        frame[["CODE_GENDER", "AGE_YEARS", "AGE_GROUP"]].to_csv(
            output / f"sensitive_{split}.csv", lineterminator="\n"
        )
        counts[split] = len(frame)
    write_json(
        output / "feature_manifest.json",
        {
            "phase": 2,
            "kind": "numeric baseline; business ratios/aggregates pending phase 3",
            "excluded": sorted(EXCLUDED),
            "feature_count": len(columns or []),
            "rows": counts,
            "fit_policy": "Imputation/scaling fit only on train in the train stage.",
        },
    )
    return counts


if __name__ == "__main__":
    print(build_dataset(load_config(parse_params())))
