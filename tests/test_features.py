"""Test the feature contract and train-only fitted transformations."""

import numpy as np
import pandas as pd
import pytest

from src.data.make_dataset import prepare_splits
from src.features.build_features import EXCLUDED, numeric_features
from src.models.train import fit_baseline


def test_same_columns_and_aligned_indices(applications: pd.DataFrame, config: dict) -> None:
    splits = prepare_splits(applications, config)
    train = numeric_features(splits["train"])
    assert not EXCLUDED.intersection(train.columns)
    for frame in splits.values():
        features = numeric_features(frame.assign(unexpected=999), list(train.columns))
        assert list(features.columns) == list(train.columns)
        assert "unexpected" not in features
        assert features.index.equals(frame.TARGET.index)
        assert features.index.equals(frame[["CODE_GENDER", "AGE_GROUP"]].index)
    with pytest.raises(ValueError, match="Missing model columns"):
        numeric_features(splits["val"].drop(columns=train.columns[0]), list(train.columns))


def test_audit_and_target_cannot_be_features(applications: pd.DataFrame) -> None:
    for name in EXCLUDED & set(applications.columns):
        with pytest.raises(ValueError, match="forbidden"):
            numeric_features(applications, [name])


def test_imputer_fit_does_not_use_validation(config: dict) -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]})
    validation = pd.DataFrame({"x": [1000.0, 2000.0, np.nan, 4000.0]})
    model, metrics = fit_baseline(
        train, pd.Series([0, 0, 1, 0, 1, 1]), validation, pd.Series([0, 1, 0, 1]), config
    )
    assert model["pipeline"].named_steps["imputer"].statistics_[0] == 4.0
    assert 0.01 <= model["threshold"] <= 0.99
    assert 0 <= metrics["validation_auc"] <= 1
    assert metrics["promotion_status"] == "not_evaluated_not_promoted"
