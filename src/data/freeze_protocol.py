"""Freeze disjoint prospective memberships before any model is trained."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.config import ROOT, load_config, write_json

SPLITS = ("train", "val", "test", "production")


def predeclared_plan(settings: dict) -> dict | None:
    """Check an optional bounded selection plan before any outputs are written."""
    plan = settings.get("qualification_plan")
    if plan is not None and (
        not isinstance(plan, dict)
        or not isinstance(plan.get("max_trials"), int)
        or isinstance(plan["max_trials"], bool)
        or not 1 <= plan["max_trials"] <= 4
        or not isinstance(plan.get("selection_rule"), str)
        or not plan["selection_rule"]
    ):
        raise ValueError("Invalid predeclared qualification plan.")
    return copy.deepcopy(plan)


def prospective_memberships(source: pd.DataFrame, used: set[int], cfg: dict) -> dict:
    """Stratify unused clients only; labels are used exclusively for allocation."""
    if source.SK_ID_CURR.isna().any() or not source.SK_ID_CURR.is_unique:
        raise ValueError("Source customer IDs must be unique and non-null.")
    if not source.TARGET.isin([0, 1]).all() or not used.issubset(set(source.SK_ID_CURR)):
        raise ValueError("Invalid labels or previously used customer IDs.")
    settings = cfg["independent_protocol"]
    count = int(len(source) * settings["source_fraction"])
    available = source.loc[~source.SK_ID_CURR.isin(used)]
    if not 0 < settings["source_fraction"] <= 1 or not 0 < count <= len(available):
        raise ValueError("The unused pool cannot supply this protocol sample.")
    seed = settings["random_state"]
    if count < len(available):
        available, _ = train_test_split(
            available, train_size=count, stratify=available.TARGET, random_state=seed
        )
    development, production = train_test_split(
        available,
        test_size=cfg["data"]["production_size"],
        stratify=available.TARGET,
        random_state=seed,
    )
    train, holdout = train_test_split(
        development,
        test_size=cfg["data"]["test_size"] + cfg["data"]["val_size"],
        stratify=development.TARGET,
        random_state=seed,
    )
    val, test = train_test_split(
        holdout,
        test_size=cfg["data"]["test_size"] / (cfg["data"]["test_size"] + cfg["data"]["val_size"]),
        stratify=holdout.TARGET,
        random_state=seed,
    )
    return {
        name: part[["SK_ID_CURR"]].sort_values("SK_ID_CURR").reset_index(drop=True)
        for name, part in zip(SPLITS, (train, val, test, production), strict=True)
    }


def freeze_protocol(cfg: dict) -> dict:
    """Write ID-only allocations and fixed rules; refuse to overwrite an existing protocol."""
    settings = cfg["independent_protocol"]
    output = ROOT / settings["directory"]
    plan = predeclared_plan(settings)
    if output.exists():
        raise ValueError(
            "Protocol directory already exists; frozen memberships cannot be replaced."
        )
    source = pd.read_csv(ROOT / "data/raw/application_train.csv", usecols=["SK_ID_CURR", "TARGET"])
    used = set()
    old_hashes = {}
    for directory in settings["used_membership_roots"]:
        local_ids = set()
        for split in SPLITS:
            path = ROOT / directory / f"{split}.csv"
            ids = pd.read_csv(path, usecols=["SK_ID_CURR"]).SK_ID_CURR
            if not ids.is_unique or local_ids.intersection(ids):
                raise ValueError("Historical split memberships overlap.")
            local_ids.update(ids)
            old_hashes[f"{directory}/{split}.csv"] = hashlib.sha256(
                ids.sort_values().to_numpy().tobytes()
            ).hexdigest()
        used.update(local_ids)
    parts = prospective_memberships(source, used, cfg)
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, ids in parts.items():
        path = output / "memberships" / f"{name}.csv"
        path.parent.mkdir(exist_ok=True)
        ids.to_csv(path, index=False, lineterminator="\n")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    prospective_cfg = copy.deepcopy(cfg)
    prospective_cfg["data"].update(
        sample_frac=settings["source_fraction"], random_state=settings["random_state"]
    )
    prospective_cfg["paths"].update(
        interim=f"{settings['directory']}/interim",
        processed=f"{settings['directory']}/processed",
    )
    prospective_cfg["qualification"] = {
        "max_trials": 6,
        "reweighing_scopes": [["AGE_GROUP"]],
        "reweighing_powers": [1.5, 1.6, 1.75],
        "configurations": [
            cfg["qualification"]["configurations"][4],
            cfg["qualification"]["configurations"][0],
        ],
        "protocol_manifest": f"{settings['directory']}/protocol.json",
        "snapshot_root": settings["directory"],
        "selection_rule": "max_validation_auc; ties: higher_recall_then_trial_order",
    }
    if plan is not None:
        prospective_cfg["qualification"] = {
            **copy.deepcopy(plan),
            "protocol_manifest": f"{settings['directory']}/protocol.json",
            "snapshot_root": settings["directory"],
            "require_independent_test": True,
        }
    prospective_cfg["train"]["threshold"]["steps"] = 981
    (output / "params.yaml").write_text(
        yaml.safe_dump(prospective_cfg, sort_keys=False), encoding="utf-8"
    )
    raw = yaml.safe_load((ROOT / "data/raw.dvc").read_text(encoding="utf-8"))
    manifest = {
        "protocol_id": settings["id"],
        "status": "memberships_frozen_no_training",
        "source_raw_dvc_hash": raw["outs"][0]["md5"],
        "source_rows": len(source),
        "previously_used_rows": len(used),
        "available_before_sampling": len(source) - len(used),
        "rows": {name: len(ids) for name, ids in parts.items()},
        "membership_sha256": hashes,
        "historical_membership_sha256": old_hashes,
        "source_fraction": settings["source_fraction"],
        "random_state": settings["random_state"],
        "overlap_with_historical": 0,
        "selection_budget": prospective_cfg["qualification"]["max_trials"],
        "selection_plan": copy.deepcopy(prospective_cfg["qualification"]),
        "test_evaluations_permitted": 1,
        "adaptive_trials_permitted": False,
        "labels_used_for": "stratified allocation only; exported membership CSVs contain no TARGET",
        "gate": {"quality_gate": cfg["quality_gate"], "fairness": cfg["fairness"]},
        "prospective_params_sha256": hashlib.sha256(
            (output / "params.yaml").read_bytes()
        ).hexdigest(),
        "prospective_config_sha256": hashlib.sha256(
            json.dumps(prospective_cfg, sort_keys=True).encode()
        ).hexdigest(),
        "stop_rule": (
            "No validation pass: no test. Test failure: no promotion or alternate-candidate test."
        ),
        "limitations": (
            "Independent within this Kaggle snapshot, not temporal or external bank validation."
        ),
    }
    write_json(output / "protocol.json", manifest)
    return manifest


def main() -> None:
    """Freeze the requested configuration without training or evaluating models."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path)
    args = parser.parse_args()
    manifest = freeze_protocol(load_config(args.params))
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "protocol_id",
                    "rows",
                    "previously_used_rows",
                    "overlap_with_historical",
                    "selection_budget",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
