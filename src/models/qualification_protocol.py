"""Resolve a frozen protocol without resetting a previously evaluated holdout."""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from src.config import ROOT, load_config


def protocol_context(cfg: dict) -> tuple[Path, dict | None]:
    """Verify frozen configuration and return its own immutable holdout ledger."""
    configured = cfg["qualification"].get("protocol_manifest")
    if not configured:
        return ROOT / "runtime/qualification-holdout.json", None
    path = ROOT / configured
    manifest = json.loads(path.read_text(encoding="utf-8"))
    identity = manifest["protocol_id"]
    if not isinstance(identity, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", identity):
        raise ValueError("Invalid protocol identity.")
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    if digest != manifest["prospective_config_sha256"]:
        raise ValueError("Configuration differs from the frozen prospective protocol.")
    if "evaluation_parent_params" in manifest:
        parent_cfg = load_config(ROOT / manifest["evaluation_parent_params"])
        parent_path = ROOT / parent_cfg["qualification"]["protocol_manifest"]
        if (
            hashlib.sha256(parent_path.read_bytes()).hexdigest()
            != manifest["evaluation_parent_manifest_sha256"]
        ):
            raise ValueError("Evaluation parent protocol changed.")
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        if "evaluation_parent_params" in parent:
            raise ValueError("Nested evaluation plans are forbidden.")
        parent_ledger, _ = protocol_context(parent_cfg)
        if any(
            cfg[key] != parent_cfg[key]
            for key in ("fairness", "quality_gate", "train", "features", "data")
        ):
            raise ValueError("Evaluation rules differ from the parent protocol.")
        if (
            manifest["membership_sha256"] != parent["membership_sha256"]
            or cfg["paths"]["processed"] != parent_cfg["paths"]["processed"]
        ):
            raise ValueError("Evaluation plan changed the reserved data.")
        return parent_ledger, manifest
    return ROOT / "runtime/qualification-holdouts" / f"{identity}.json", manifest


def verify_frozen_candidate(cfg: dict, content: bytes) -> None:
    """Enforce the single candidate identity before any holdout reads."""
    _, manifest = protocol_context(cfg)
    if manifest and "frozen_candidate_sha256" in manifest:
        if hashlib.sha256(content).hexdigest() != manifest["frozen_candidate_sha256"]:
            raise ValueError("Candidate differs from the frozen evaluation plan.")
        for filename, expected in manifest.get("frozen_source_sha256", {}).items():
            if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != expected:
                raise ValueError("Evaluation source code differs from the frozen plan.")


def verify_memberships(cfg: dict, manifest: dict, frames: dict[str, pd.DataFrame]) -> None:
    """Check ID-only allocation hashes and exact membership without reading target files."""
    directory = (ROOT / cfg["qualification"]["protocol_manifest"]).parent / "memberships"
    for name, frame in frames.items():
        path = directory / f"{name}.csv"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest["membership_sha256"][name]:
            raise ValueError("Frozen protocol membership file was modified.")
        expected = pd.read_csv(path, usecols=["SK_ID_CURR"]).SK_ID_CURR
        if not frame.index.is_unique or not expected.is_unique or set(frame.index) != set(expected):
            raise ValueError(f"Processed {name} IDs differ from the frozen protocol.")
