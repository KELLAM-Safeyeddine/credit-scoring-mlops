"""Create controlled drift scenarios from the disjoint production reserve."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, parse_params, project_path, write_json


def monitor_model_path(cfg: dict) -> Path:
    """Use the explicitly configured diagnostic model, never imply deployment."""
    return project_path(cfg, "audit_models") / cfg["monitoring"]["model_filename"]


def scenario_features(
    frame: pd.DataFrame, sensitive: pd.DataFrame, number: int, settings: dict
) -> pd.DataFrame:
    """Perturb inputs and recompute row-local ratios; audit variables remain separate."""
    result = frame.copy()
    if number in (4, 5):
        result["AMT_INCOME_TOTAL"] *= settings["income_multiplier"]
        result["AMT_CREDIT"] *= settings["credit_multiplier"]
        result["CREDIT_INCOME_RATIO"] = result.AMT_CREDIT / result.AMT_INCOME_TOTAL
        result["ANNUITY_INCOME_RATIO"] = result.AMT_ANNUITY / result.AMT_INCOME_TOTAL
        result["CREDIT_TERM"] = result.AMT_CREDIT / result.AMT_ANNUITY.replace(0, np.nan)
    if number == 7:
        targeted = sensitive.CODE_GENDER == settings["targeted_group"]
        columns = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        result.loc[targeted, columns] *= settings["targeted_score_multiplier"]
        result["EXT_SOURCE_MEAN"] = result[columns].mean(axis=1)
    if not frame.columns.equals(result.columns):
        raise ValueError("Scenario changed the feature contract.")
    return result


def assign_batches(
    index: pd.Index, sensitive: pd.DataFrame, settings: dict, seed: int
) -> list[np.ndarray]:
    """Keep eight disjoint batches and enrich batch six with real young applicants."""
    if settings["n_batches"] != 8:
        raise ValueError("The demonstration requires exactly eight batches.")
    rng = np.random.default_rng(seed)
    initial = list(np.array_split(rng.permutation(index), 8))
    remaining = np.concatenate(initial[3:])
    young = remaining[sensitive.loc[remaining, "AGE_GROUP"].eq("<30").to_numpy()]
    others = remaining[~sensitive.loc[remaining, "AGE_GROUP"].eq("<30").to_numpy()]
    size = len(initial[5])
    count = int(size * settings["young_population_share"])
    if len(young) < count or len(others) < size - count:
        raise ValueError("Production reserve cannot support the requested population shift.")
    sixth = np.concatenate(
        [rng.choice(young, count, replace=False), rng.choice(others, size - count, replace=False)]
    )
    rng.shuffle(sixth)
    rest = rng.permutation(remaining[~np.isin(remaining, sixth)])
    fourth, fifth, seventh, eighth = np.array_split(rest, 4)
    batches = initial[:3] + [fourth, fifth, sixth, seventh, eighth]
    if len(set(np.concatenate(batches))) != len(index):
        raise ValueError("Batch assignment is not a disjoint partition.")
    return batches


def simulate(cfg: dict) -> dict:
    """Export features, frozen-model scores and separately delayed ground truth."""
    processed = project_path(cfg, "processed")
    output = project_path(cfg, "production")
    (output / "ground_truth").mkdir(parents=True, exist_ok=True)
    (output / ".gitkeep").unlink(missing_ok=True)
    x = pd.read_csv(processed / "X_production.csv", index_col="SK_ID_CURR")
    y = pd.read_csv(processed / "y_production.csv", index_col="SK_ID_CURR")
    sensitive = pd.read_csv(processed / "sensitive_production.csv", index_col="SK_ID_CURR")
    columns = json.loads((processed / "feature_columns.json").read_text(encoding="utf-8"))
    if (
        list(x.columns) != columns
        or not x.index.equals(y.index)
        or not x.index.equals(sensitive.index)
    ):
        raise ValueError("Production feature/label/audit alignment failure.")
    # This is our locally generated, DVC-versioned artifact, not an external pickle.
    bundle = pickle.loads(monitor_model_path(cfg).read_bytes())
    if columns != bundle["columns"]:
        raise ValueError("Diagnostic model and production columns differ.")
    batches = assign_batches(x.index, sensitive, cfg["monitoring"], cfg["data"]["random_state"])
    scenarios = [
        "normal",
        "normal",
        "normal",
        "financial_data_drift",
        "financial_data_drift",
        "young_population_drift",
        "targeted_group_input_degradation",
        "concept_drift",
    ]
    manifest = []
    for number, ids in enumerate(batches, 1):
        label = f"{number:02d}"
        current = scenario_features(x.loc[ids], sensitive.loc[ids], number, cfg["monitoring"])
        probabilities = bundle["pipeline"].predict_proba(current)[:, 1]
        predictions = pd.DataFrame(
            {
                "probability_default": probabilities,
                "predicted_default": (probabilities >= bundle["threshold"]).astype("int8"),
            },
            index=current.index,
        )
        truth = y.loc[ids].copy()
        if number == 8:
            # Preserve prevalence; invert the score/outcome relationship, not the score.
            count = int(truth.TARGET.sum())
            truth["TARGET"] = 0
            truth.loc[current.index[np.argsort(probabilities, kind="stable")[:count]], "TARGET"] = 1
        for filename, frame in {
            f"X_{label}.csv": current,
            f"sensitive_{label}.csv": sensitive.loc[ids],
            f"predictions_{label}.csv": predictions,
            f"ground_truth/y_{label}.csv": truth,
        }.items():
            frame.to_csv(output / filename, index_label="SK_ID_CURR", lineterminator="\n")
        manifest.append(
            {
                "batch_id": label,
                "rows": len(ids),
                "scenario": scenarios[number - 1],
                "modified_target": number == 8,
                "expected_alert": number >= 4,
                "labels_available_at_batch": number + cfg["monitoring"]["label_delay_batches"],
            }
        )
    result = {
        "phase": 10,
        "feature_columns": columns,
        "model_run_id": bundle["run_id"],
        "model_artifact": str(
            Path(cfg["paths"]["audit_models"]) / cfg["monitoring"]["model_filename"]
        ),
        "model_role": "frozen mitigated candidate for diagnosis; not deployed champion",
        "threshold": bundle["threshold"],
        "batches": manifest,
        "label_policy": "Consumers must honor labels_available_at_batch.",
        "limitations": (
            "Controlled synthetic shifts; age enrichment redistributes the finite reserve. "
            "No production label is used to fit the model. Batch 08 labels are synthetic."
        ),
    }
    write_json(output / "manifest.json", result)
    return result


def main() -> None:
    """Generate the configured eight diagnostic batches."""
    result = simulate(load_config(parse_params()))
    print(f"Wrote {len(result['batches'])} disjoint scenario batches.")


if __name__ == "__main__":
    main()
