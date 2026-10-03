"""Independent evaluation uses a frozen policy and cannot silently change candidates."""

import numpy as np
import pandas as pd
import pytest

from src.config import write_json
from src.models.promote import SHAP_FILES
from src.models.qualify_holdout import evaluate_holdout, reserve_holdout


def test_holdout_uses_frozen_threshold_and_never_fits(config, tmp_path):
    index = pd.Index(range(16), name="SK_ID_CURR")
    scores = np.tile([0.1, 0.6, 0.1, 0.6], 4)
    pd.DataFrame({"score": scores}, index=index).to_csv(tmp_path / "X_test.csv")
    pd.DataFrame({"TARGET": np.tile([0, 1], 8)}, index=index).to_csv(tmp_path / "y_test.csv")
    pd.DataFrame(
        {
            "CODE_GENDER": np.tile(["F", "F", "M", "M"], 4),
            "AGE_GROUP": np.repeat(config["fairness"]["age_labels"], 4),
        },
        index=index,
    ).to_csv(tmp_path / "sensitive_test.csv")
    for name, value in (("train", 100), ("val", 200)):
        pd.DataFrame({"SK_ID_CURR": [value]}).to_csv(tmp_path / f"X_{name}.csv", index=False)

    class FrozenPipeline:
        def fit(self, *args, **kwargs):
            raise AssertionError("Independent evaluation must never fit.")

        def predict_proba(self, x):
            return np.column_stack([1 - x.score, x.score])

    config["fairness"].update(min_group_size=1, min_class_count=1)
    bundle = {
        "pipeline": FrozenPipeline(),
        "columns": ["score"],
        "threshold": 0.7,
        "run_id": "frozen-fixture",
    }
    evidence = {"shap_verified": True, "shap_files": sorted(SHAP_FILES)}
    report = evaluate_holdout(bundle, config, evidence, tmp_path)
    assert report["metrics"]["test_auc"] == 1
    assert report["metrics"]["test_recall"] == 0  # No replacement by a better test threshold.
    assert report["frozen_threshold"] == 0.7
    assert not report["passed"] and "minimum_recall" in report["failed_rules"]
    assert not report["selection_or_retuning_allowed"]
    pd.DataFrame({"SK_ID_CURR": [0]}).to_csv(tmp_path / "X_train.csv", index=False)
    with pytest.raises(ValueError, match="overlaps"):
        evaluate_holdout(bundle, config, evidence, tmp_path)


def test_holdout_ledger_reuses_cache_and_rejects_changed_identity(tmp_path):
    ledger = tmp_path / "ledger.json"
    identity = {"candidate_sha256": "first", "data_sha256": "frozen"}
    assert reserve_holdout(ledger, identity) is None
    write_json(ledger, {"identity": identity, "report": {"passed": False}})
    assert reserve_holdout(ledger, identity) == {"passed": False}
    original = ledger.read_bytes()
    with pytest.raises(ValueError, match="no retuning"):
        reserve_holdout(ledger, {**identity, "candidate_sha256": "second"})
    assert ledger.read_bytes() == original
