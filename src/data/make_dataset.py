"""Prepare disjoint development splits and an independent production reserve."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config, parse_params, project_path, write_json
from src.data.validate import load_validated_application


def prepare_splits(frame: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Reserve production first; stratify remaining rows into train/val/test."""
    settings = cfg["data"]
    seed = settings["random_state"]
    frame = frame.copy()
    if settings["sample_frac"] < 1:
        frame, _ = train_test_split(
            frame, train_size=settings["sample_frac"], stratify=frame.TARGET, random_state=seed
        )
    frame["EMPLOYED_ANOMALY"] = frame.DAYS_EMPLOYED.eq(365243).astype("int8")
    frame["DAYS_EMPLOYED"] = frame.DAYS_EMPLOYED.mask(frame.EMPLOYED_ANOMALY.eq(1), np.nan)
    frame["AGE_YEARS"] = -frame.DAYS_BIRTH / 365.25
    frame["AGE_GROUP"] = pd.cut(
        frame.AGE_YEARS,
        bins=[0, *cfg["fairness"]["age_bins"], float("inf")],
        labels=cfg["fairness"]["age_labels"],
        right=False,
    )
    development, production = train_test_split(
        frame, test_size=settings["production_size"], stratify=frame.TARGET, random_state=seed
    )
    train, holdout = train_test_split(
        development,
        test_size=settings["test_size"] + settings["val_size"],
        stratify=development.TARGET,
        random_state=seed,
    )
    validation, test = train_test_split(
        holdout,
        test_size=settings["test_size"] / (settings["test_size"] + settings["val_size"]),
        stratify=holdout.TARGET,
        random_state=seed,
    )
    return {
        name: part.set_index("SK_ID_CURR").sort_index()
        for name, part in {
            "train": train,
            "val": validation,
            "test": test,
            "production": production,
        }.items()
    }


def main() -> None:
    """Validate the source and export deterministic interim CSVs."""
    cfg = load_config(parse_params())
    frame = load_validated_application(project_path(cfg, "raw") / "application_train.csv")
    destination = project_path(cfg, "interim")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / ".gitkeep").unlink(missing_ok=True)
    summary = {}
    for name, part in prepare_splits(frame, cfg).items():
        part.to_csv(destination / f"{name}.csv", index_label="SK_ID_CURR", lineterminator="\n")
        summary[name] = {"rows": len(part), "default_rate": float(part.TARGET.mean())}
    write_json(
        destination / "split_manifest.json",
        {
            "source": "application_train.csv",
            "source_rows": len(frame),
            "splits": summary,
            "data_config": cfg["data"],
            "split_policy": "Reserve production, then 60/20/20.",
        },
    )
    print(summary)


if __name__ == "__main__":
    main()
