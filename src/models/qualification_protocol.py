"""Resolve a frozen protocol without resetting a previously evaluated holdout."""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from src.config import ROOT


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
    return ROOT / "runtime/qualification-holdouts" / f"{identity}.json", manifest


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
