"""Reserve unused clients and apply the existing feature encoder without refitting."""

import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ROOT, load_config, project_path, write_json
from src.data.make_dataset import clean_application
from src.features.aggregations import (
    aggregate_bureau,
    aggregate_installments,
    aggregate_previous_application,
)
from src.features.build_features import add_ratios


def sha256(path: Path) -> str:
    """Hash artifacts in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_cohort(settings: dict) -> dict:
    """Freeze allocation and criteria before fitting or evaluating calibration."""
    output = ROOT / settings["output"]
    if output.exists():
        raise ValueError("Calibration allocation already exists; it cannot be reset.")
    source = ROOT / settings["source_candidate"]
    if sha256(source) != settings["source_candidate_sha256"]:
        raise ValueError("The frozen source candidate has changed.")
    cfg = load_config(ROOT / settings["source_params"])
    raw = project_path(cfg, "raw") / "application_train.csv"
    labels = pd.read_csv(raw, usecols=["SK_ID_CURR", "TARGET"])
    if not labels.SK_ID_CURR.is_unique or set(labels.TARGET.unique()) != {0, 1}:
        raise ValueError("Invalid source identities or targets.")
    used: set[int] = set()
    history_hashes = {}
    for root in settings["historical_membership_roots"]:
        for name in ("train", "val", "test", "production"):
            path = ROOT / root / f"{name}.csv"
            history_hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
            used.update(pd.read_csv(path, usecols=["SK_ID_CURR"]).SK_ID_CURR)
    unused = labels.loc[~labels.SK_ID_CURR.isin(used)]
    if len(unused) < 10000:
        raise ValueError("Insufficient untouched clients for calibration and independent test.")
    calibration, test = train_test_split(
        unused,
        train_size=settings["calibration_fraction"],
        stratify=unused.TARGET,
        random_state=settings["random_state"],
    )
    output.mkdir(parents=True)
    membership = output / "memberships"
    membership.mkdir()
    for name, part in (("calibration", calibration), ("test", test)):
        part[["SK_ID_CURR"]].sort_values("SK_ID_CURR").to_csv(
            membership / f"{name}.csv", index=False, lineterminator="\n"
        )
    plan = {
        "settings": settings,
        "raw_sha256": sha256(raw),
        "historical_membership_sha256": history_hashes,
        "membership_sha256": {
            name: sha256(path)
            for name, path in (
                ("calibration", membership / "calibration.csv"),
                ("test", membership / "test.csv"),
            )
        },
        "rows": {"calibration": len(calibration), "test": len(test)},
        "previously_allocated_clients": len(used),
        "policy": "Single sigmoid; no class weights; no test-based selection or retuning.",
        "base_training": "Frozen source model, encoder and imputer; no tree refitting.",
        "threshold": "Strictly monotone mapping of the existing threshold; preserve decisions.",
    }
    write_json(output / "allocation.json", plan)
    print(json.dumps({"reserved": plan["rows"], "historical_excluded": len(used)}), flush=True)
    ids = pd.Index(sorted(set(unused.SK_ID_CURR)), name="SK_ID_CURR")
    parts = [chunk.loc[chunk.SK_ID_CURR.isin(ids)] for chunk in pd.read_csv(raw, chunksize=20000)]
    frame = clean_application(pd.concat(parts, ignore_index=True), cfg).set_index("SK_ID_CURR")
    enriched = add_ratios(frame)
    features = cfg["features"]
    for key, filename, aggregate in (
        ("use_bureau", "bureau.csv", aggregate_bureau),
        ("use_previous_application", "previous_application.csv", aggregate_previous_application),
        ("use_installments", "installments_payments.csv", aggregate_installments),
    ):
        if not features[key]:
            continue
        options = {"chunksize": features["chunksize"], "cutoff": features["history_cutoff_days"]}
        if key == "use_installments":
            options.update(partitions=features["installment_partitions"], temp_dir=output)
        print(f"Aggregating {filename} for {len(ids)} untouched clients", flush=True)
        history = aggregate(project_path(cfg, "raw") / filename, ids, **options)
        enriched = enriched.join(history, validate="one_to_one")
    encoder_path = project_path(cfg, "processed") / "feature_encoder.pkl"
    encoder = pickle.loads(encoder_path.read_bytes())
    processed = output / "processed"
    processed.mkdir()
    for name, part in (("calibration", calibration), ("test", test)):
        rows = enriched.loc[sorted(part.SK_ID_CURR)]
        encoder.transform(rows).to_csv(processed / f"X_{name}.csv", lineterminator="\n")
        rows[["TARGET"]].to_csv(processed / f"y_{name}.csv", lineterminator="\n")
        rows[["CODE_GENDER", "AGE_GROUP"]].to_csv(
            processed / f"sensitive_{name}.csv", lineterminator="\n"
        )
    write_json(
        output / "prepared.json",
        {
            "allocation_sha256": sha256(output / "allocation.json"),
            "encoder_sha256": sha256(encoder_path),
            "data_sha256": {p.name: sha256(p) for p in sorted(processed.glob("*.csv"))},
        },
    )
    return {"output": str(output), "rows": plan["rows"], "encoder_refitted": False}
