"""Prospective allocations are frozen, disjoint and inaccessible to configuration changes."""

import hashlib
import json

import pandas as pd
import pytest

from src.config import write_json
from src.data.freeze_protocol import prospective_memberships
from src.models.qualification_protocol import protocol_context, verify_memberships


def test_new_memberships_exclude_all_historical_clients(applications, config):
    used = set(applications.SK_ID_CURR[:120])
    source = applications[["SK_ID_CURR", "TARGET"]]
    parts = prospective_memberships(source, used, config)
    assert sum(map(len, parts.values())) == int(len(source) * 0.3)
    seen = set(used)
    repeated = prospective_memberships(source, used, config)
    for name, part in parts.items():
        assert list(part) == ["SK_ID_CURR"]  # No targets or audit attributes exported.
        assert not seen.intersection(part.SK_ID_CURR)
        seen.update(part.SK_ID_CURR)
        pd.testing.assert_frame_equal(part, repeated[name])


def test_protocol_refuses_insufficient_unused_pool(applications, config):
    with pytest.raises(ValueError, match="unused pool"):
        prospective_memberships(applications, set(applications.SK_ID_CURR[:550]), config)
    with pytest.raises(ValueError, match="previously used"):
        prospective_memberships(applications, {-1}, config)


def test_protocol_blocks_changed_config_or_memberships(tmp_path, config):
    path = tmp_path / "protocol.json"
    config["qualification"]["protocol_manifest"] = str(path)
    directory = tmp_path / "memberships"
    directory.mkdir()
    membership = directory / "test.csv"
    membership.write_text("SK_ID_CURR\n1\n2\n", encoding="utf-8")
    manifest = {
        "protocol_id": "home-credit-independent-v2",
        "prospective_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "membership_sha256": {"test": hashlib.sha256(membership.read_bytes()).hexdigest()},
    }
    write_json(path, manifest)
    ledger, restored = protocol_context(config)
    assert ledger.name == "home-credit-independent-v2.json"
    assert ledger.parent.name == "qualification-holdouts"
    frame = pd.DataFrame(index=pd.Index([1, 2], name="SK_ID_CURR"))
    verify_memberships(config, restored, {"test": frame})
    with pytest.raises(ValueError, match="IDs differ"):
        verify_memberships(config, restored, {"test": frame.iloc[:1]})
    membership.write_text("SK_ID_CURR\n3\n4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="modified"):
        verify_memberships(config, restored, {"test": frame})
    config["quality_gate"]["min_auc"] = 0.739
    with pytest.raises(ValueError, match="frozen"):
        protocol_context(config)
