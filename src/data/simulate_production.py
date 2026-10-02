"""Create normal production batches with separate audit tables and delayed labels."""

import json
import pickle

import numpy as np
import pandas as pd

from src.config import load_config, parse_params, project_path, write_json


def main() -> None:
    """Export normal batches; drift scenarios will be implemented in phase ten."""
    cfg = load_config(parse_params())
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
    bundle = pickle.loads((project_path(cfg, "models") / "baseline.pkl").read_bytes())
    generator = np.random.default_rng(cfg["data"]["random_state"])
    batches = np.array_split(generator.permutation(x.index), cfg["monitoring"]["n_batches"])
    manifest = []
    for number, ids in enumerate(batches, 1):
        label = f"{number:02d}"
        current = x.loc[ids]
        probabilities = bundle["pipeline"].predict_proba(current)[:, 1]
        predictions = pd.DataFrame(
            {
                "probability_default": probabilities,
                "predicted_default": (probabilities >= bundle["threshold"]).astype("int8"),
            },
            index=current.index,
        )
        for filename, frame in {
            f"X_{label}.csv": current,
            f"sensitive_{label}.csv": sensitive.loc[ids],
            f"predictions_{label}.csv": predictions,
            f"ground_truth/y_{label}.csv": y.loc[ids],
        }.items():
            frame.to_csv(output / filename, index_label="SK_ID_CURR", lineterminator="\n")
        manifest.append(
            {
                "batch_id": label,
                "rows": len(ids),
                "scenario": "normal_engineered_baseline",
                "labels_available_at_batch": number + cfg["monitoring"]["label_delay_batches"],
            }
        )
    write_json(
        output / "manifest.json",
        {
            "phase": 3,
            "feature_columns": columns,
            "batches": manifest,
            "label_policy": "Consumers must honor labels_available_at_batch.",
            "drift_scenarios": "Pending phase 10; no alert demonstration claimed here.",
        },
    )
    print(f"Wrote {len(batches)} normal batches ({len(x):,} distinct production rows).")


if __name__ == "__main__":
    main()
