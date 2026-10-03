"""A transfer must never publish a candidate lacking independent qualification."""

import pickle
from unittest.mock import Mock

import pytest

from src.models import transfer_champion


def test_transfer_rejects_same_server_before_loading_candidate(tmp_path, monkeypatch, config):
    uri = config["mlflow"]["tracking_uri"]
    monkeypatch.setattr(transfer_champion, "load_config", lambda path: config)
    with pytest.raises(ValueError, match="must differ"):
        transfer_champion.transfer(tmp_path / "params", tmp_path / "absent.pkl", uri)


def test_transfer_rejects_unqualified_source_before_target_mutation(tmp_path, monkeypatch, config):
    candidate = tmp_path / "candidate.pkl"
    candidate.write_bytes(pickle.dumps({"run_id": "unqualified"}))
    monkeypatch.setattr(transfer_champion, "load_config", lambda path: config)
    monkeypatch.setattr(transfer_champion, "candidate_evidence", lambda bundle, client: {})
    monkeypatch.setattr(transfer_champion, "quality_gate", lambda evidence, prod, cfg: (False, {}))
    client = Mock()
    monkeypatch.setattr(transfer_champion.mlflow, "MlflowClient", lambda **kwargs: client)
    set_tracking_uri = Mock()
    monkeypatch.setattr(transfer_champion.mlflow, "set_tracking_uri", set_tracking_uri)
    with pytest.raises(ValueError, match="independently qualified"):
        transfer_champion.transfer(tmp_path / "params", candidate, "http://target:5000")
    set_tracking_uri.assert_not_called()
    client.get_run.assert_not_called()
