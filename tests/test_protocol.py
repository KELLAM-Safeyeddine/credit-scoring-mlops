"""Prospective allocations are frozen, disjoint and inaccessible to configuration changes."""

import copy
import hashlib
import json

import pandas as pd
import pytest
import yaml

from src.config import write_json
from src.data.freeze_protocol import prospective_memberships
from src.models.qualification_protocol import (
    protocol_context,
    verify_frozen_candidate,
    verify_memberships,
)


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


def test_new_evaluation_plan_keeps_parent_ledger_and_candidate(tmp_path, config):
    parent_cfg = copy.deepcopy(config)
    parent_file = tmp_path / "parent.json"
    parent_cfg["qualification"]["protocol_manifest"] = str(parent_file)
    parent_params = tmp_path / "parent.yaml"
    parent_params.write_text(yaml.safe_dump(parent_cfg), encoding="utf-8")
    parent = {
        "protocol_id": "parent-test-unopened",
        "membership_sha256": {},
        "prospective_config_sha256": hashlib.sha256(
            json.dumps(parent_cfg, sort_keys=True).encode()
        ).hexdigest(),
    }
    write_json(parent_file, parent)
    child_cfg = copy.deepcopy(parent_cfg)
    child_file = tmp_path / "child.json"
    child_cfg["qualification"]["protocol_manifest"] = str(child_file)
    child = {
        "protocol_id": "single-frozen-candidate",
        "membership_sha256": {},
        "evaluation_parent_params": str(parent_params),
        "evaluation_parent_manifest_sha256": hashlib.sha256(parent_file.read_bytes()).hexdigest(),
        "prospective_config_sha256": hashlib.sha256(
            json.dumps(child_cfg, sort_keys=True).encode()
        ).hexdigest(),
        "frozen_candidate_sha256": hashlib.sha256(b"candidate").hexdigest(),
    }
    write_json(child_file, child)
    assert protocol_context(child_cfg)[0] == protocol_context(parent_cfg)[0]
    verify_frozen_candidate(child_cfg, b"candidate")
    with pytest.raises(ValueError, match="Candidate differs"):
        verify_frozen_candidate(child_cfg, b"alternate")
    child_cfg["quality_gate"]["min_auc"] = 0.73
    child["prospective_config_sha256"] = hashlib.sha256(
        json.dumps(child_cfg, sort_keys=True).encode()
    ).hexdigest()
    write_json(child_file, child)
    with pytest.raises(ValueError, match="rules differ"):
        protocol_context(child_cfg)
