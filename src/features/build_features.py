"""Build financial ratios, historical aggregates and a train-fitted feature contract."""

import pickle
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, parse_params, project_path, write_json
from src.features.aggregations import (
    aggregate_bureau,
    aggregate_installments,
    aggregate_previous_application,
)
from src.features.contracts import EXCLUDED as EXCLUDED
from src.features.contracts import numeric_features as numeric_features
from src.features.encoding import FeatureEncoder


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return missing for zero/missing denominators, never infinity."""
    return numerator.div(denominator.where(denominator.ne(0))).replace([np.inf, -np.inf], np.nan)


def add_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute row-local ratios; the age-derived ratio remains audit-only."""
    result = frame.copy()
    result["CREDIT_INCOME_RATIO"] = safe_ratio(result.AMT_CREDIT, result.AMT_INCOME_TOTAL)
    result["ANNUITY_INCOME_RATIO"] = safe_ratio(result.AMT_ANNUITY, result.AMT_INCOME_TOTAL)
    result["CREDIT_TERM"] = safe_ratio(result.AMT_CREDIT, result.AMT_ANNUITY)
    result["EMPLOYED_AGE_RATIO"] = safe_ratio(result.DAYS_EMPLOYED, result.DAYS_BIRTH)
    sources = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    result["EXT_SOURCE_MEAN"] = result[sources].mean(axis=1)
    return result


def build_dataset(cfg: dict[str, Any]) -> dict[str, int]:
    """Join client histories and export X/y/audit using the train-fitted schema."""
    output = project_path(cfg, "processed")
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        name: pd.read_csv(project_path(cfg, "interim") / f"{name}.csv", index_col="SK_ID_CURR")
        for name in ("train", "val", "test", "production")
    }
    ids = pd.Index(
        sorted(set().union(*(set(frame.index) for frame in frames.values()))), name="SK_ID_CURR"
    )
    settings = cfg["features"]
    aggregations = []
    for enabled, filename, function in (
        ("use_bureau", "bureau.csv", aggregate_bureau),
        ("use_previous_application", "previous_application.csv", aggregate_previous_application),
        ("use_installments", "installments_payments.csv", aggregate_installments),
    ):
        if settings[enabled]:
            options = {
                "chunksize": settings["chunksize"],
                "cutoff": settings["history_cutoff_days"],
            }
            if enabled == "use_installments":
                options["partitions"] = settings["installment_partitions"]
            print(f"Aggregating {filename} for {len(ids):,} client IDs", flush=True)
            aggregations.append(function(project_path(cfg, "raw") / filename, ids, **options))
    histories = pd.concat(aggregations, axis=1) if aggregations else pd.DataFrame(index=ids)
    enriched = {
        name: add_ratios(frame).join(histories, how="left", validate="one_to_one")
        for name, frame in frames.items()
    }
    encoder = FeatureEncoder().fit(enriched["train"])
    counts = {}
    missingness = {}
    for split, frame in enriched.items():
        features = encoder.transform(frame)
        features.to_csv(output / f"X_{split}.csv", index_label="SK_ID_CURR", lineterminator="\n")
        frame[["TARGET"]].to_csv(output / f"y_{split}.csv", lineterminator="\n")
        frame[["CODE_GENDER", "AGE_YEARS", "AGE_GROUP", "EMPLOYED_AGE_RATIO"]].to_csv(
            output / f"sensitive_{split}.csv", lineterminator="\n"
        )
        counts[split] = len(frame)
        missingness[split] = {
            name: int(value) for name, value in features.isna().sum().items() if value
        }
    write_json(output / "feature_columns.json", encoder.columns)
    write_json(
        output / "category_vocabulary.json",
        {
            name: list(values)
            for name, values in zip(
                encoder.categorical_columns,
                encoder.encoder.categories_ if encoder.categorical_columns else [],
                strict=True,
            )
        },
    )
    (output / "feature_encoder.pkl").write_bytes(
        pickle.dumps(encoder, protocol=pickle.HIGHEST_PROTOCOL)
    )
    write_json(
        output / "feature_manifest.json",
        {
            "phase": 3,
            "excluded": sorted(EXCLUDED),
            "feature_count": len(encoder.columns),
            "numeric_count": len(encoder.numeric_columns),
            "categorical_inputs": encoder.categorical_columns,
            "rows": counts,
            "missing_values": missingness,
            "missing_policy": "Source gaps retained; model imputer fits only on train.",
            "history_policy": "Per-client history through day 0, without population fitting.",
            "encoding_policy": "Train-only category vocabulary, unknown categories ignored.",
            "age_policy": "EMPLOYED_AGE_RATIO computed for audit only, excluded from model.",
        },
    )
    return counts


if __name__ == "__main__":
    print(build_dataset(load_config(parse_params())))
