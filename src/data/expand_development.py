"""Expand development data without moving previously reserved customers."""

import argparse
import copy
import hashlib
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.config import ROOT, load_config, project_path, write_json
from src.data.make_dataset import clean_application, prepare_splits
from src.data.validate import load_validated_application


def expand_splits(frame: pd.DataFrame, memberships: dict, cfg: dict, fraction: float) -> dict:
    """Keep all existing memberships; stratify only previously unused customers."""
    reserved = set()
    for ids in memberships.values():
        if not ids.is_unique or reserved.intersection(ids):
            raise ValueError("Existing split IDs must be unique and disjoint.")
        reserved.update(ids)
    if not 0 < fraction <= 1 or not reserved.issubset(set(frame.SK_ID_CURR)):
        raise ValueError("Invalid expansion fraction or unknown reserved IDs.")
    count = int(len(frame) * fraction) - len(reserved)
    unused = frame.loc[~frame.SK_ID_CURR.isin(reserved)]
    if count <= 0 or count > len(unused):
        raise ValueError("Expansion must add previously unused observations.")
    if count < len(unused):
        unused, _ = train_test_split(
            unused,
            train_size=count,
            stratify=unused.TARGET,
            random_state=cfg["data"]["random_state"],
        )
    expanded_cfg = copy.deepcopy(cfg)
    expanded_cfg["data"]["sample_frac"] = 1.0
    additions = prepare_splits(unused, expanded_cfg)
    original = clean_application(frame.loc[frame.SK_ID_CURR.isin(reserved)], cfg)
    original = original.set_index("SK_ID_CURR")
    return {
        name: pd.concat([original.loc[ids], additions[name]]).sort_index()
        for name, ids in memberships.items()
    }


def main() -> None:
    """Create a separate reproducible campaign configuration and interim snapshot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.2)
    args = parser.parse_args()
    cfg = load_config()
    memberships = {
        name: pd.Index(
            pd.read_csv(
                project_path(cfg, "interim") / f"{name}.csv", usecols=["SK_ID_CURR"]
            ).SK_ID_CURR
        )
        for name in ("train", "val", "test", "production")
    }
    frame = load_validated_application(project_path(cfg, "raw") / "application_train.csv")
    splits = expand_splits(frame, memberships, cfg, args.fraction)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name, part in splits.items():
        path = output / "interim" / f"{name}.csv"
        path.parent.mkdir(exist_ok=True)
        part.to_csv(path, index_label="SK_ID_CURR", lineterminator="\n")
    cfg["data"]["sample_frac"] = args.fraction
    cfg["paths"]["interim"] = str(output / "interim")
    cfg["paths"]["processed"] = str(output / "processed")
    cfg["qualification"]["configurations"] = cfg["qualification"]["configurations"][4:6]
    cfg["qualification"]["max_trials"] = 6
    cfg["qualification"]["snapshot_root"] = str(output)
    (output / "params.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    manifest = {
        "sample_frac": args.fraction,
        "rows": {name: len(part) for name, part in splits.items()},
        "existing_membership_preserved": True,
        "original_membership_sha256": {
            name: hashlib.sha256(ids.to_numpy().tobytes()).hexdigest()
            for name, ids in memberships.items()
        },
        "source": str(ROOT / "data/raw/application_train.csv"),
        "policy": "Old assignments frozen; new IDs stratified 60/20/20 after production reserve.",
    }
    write_json(output / "split_manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
