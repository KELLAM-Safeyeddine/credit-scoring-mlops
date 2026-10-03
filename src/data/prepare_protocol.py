"""Materialize frozen prospective memberships without resampling customers."""

import pandas as pd
import yaml

from src.config import ROOT, load_config, parse_params, project_path
from src.data.make_dataset import clean_application
from src.data.validate import load_validated_application
from src.models.qualification_protocol import protocol_context, verify_memberships


def select_protocol_rows(source: pd.DataFrame, memberships: dict, cfg: dict) -> dict:
    """Apply fixed IDs and row-local cleaning; never call a random split."""
    if not source.SK_ID_CURR.is_unique or source.SK_ID_CURR.isna().any():
        raise ValueError("Protocol source IDs must be unique and non-null.")
    indexed = source.set_index("SK_ID_CURR")
    seen, frames = set(), {}
    for name, ids in memberships.items():
        if not ids.is_unique or seen.intersection(ids) or set(ids) - set(indexed.index):
            raise ValueError("Protocol memberships overlap or contain unknown IDs.")
        seen.update(ids)
        frames[name] = clean_application(indexed.loc[ids], cfg).sort_index()
    return frames


def main() -> None:
    """Validate raw provenance and export interim data for the frozen prospective plan."""
    cfg = load_config(parse_params())
    _, manifest = protocol_context(cfg)
    if manifest is None:
        raise ValueError("A frozen prospective protocol is required.")
    raw = yaml.safe_load((ROOT / "data/raw.dvc").read_text(encoding="utf-8"))
    if raw["outs"][0]["md5"] != manifest["source_raw_dvc_hash"]:
        raise ValueError("Raw DVC version differs from the frozen protocol.")
    destination = project_path(cfg, "interim")
    if destination.exists():
        raise ValueError("Protocol interim data already exists; use its DVC snapshot.")
    membership_dir = (ROOT / cfg["qualification"]["protocol_manifest"]).parent / "memberships"
    memberships = {
        name: pd.Index(pd.read_csv(membership_dir / f"{name}.csv").SK_ID_CURR)
        for name in ("train", "val", "test", "production")
    }
    verify_memberships(
        cfg, manifest, {name: pd.DataFrame(index=ids) for name, ids in memberships.items()}
    )
    source = load_validated_application(project_path(cfg, "raw") / "application_train.csv")
    frames = select_protocol_rows(source, memberships, cfg)
    destination.mkdir(parents=True, exist_ok=False)
    for name, frame in frames.items():
        frame.to_csv(destination / f"{name}.csv", index_label="SK_ID_CURR", lineterminator="\n")
    print({name: len(frame) for name, frame in frames.items()})


if __name__ == "__main__":
    main()
