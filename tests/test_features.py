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


def test_ratios_and_missing_denominators() -> None:
    from src.features.build_features import add_ratios

    frame = pd.DataFrame(
        {
            "AMT_CREDIT": [200.0, 200.0, 200.0],
            "AMT_INCOME_TOTAL": [100.0, 0.0, np.nan],
            "AMT_ANNUITY": [10.0, 0.0, np.nan],
            "DAYS_EMPLOYED": [-1000.0, np.nan, -500.0],
            "DAYS_BIRTH": [-10000.0, -10000.0, -10000.0],
            "EXT_SOURCE_1": [0.2, np.nan, np.nan],
            "EXT_SOURCE_2": [0.4, 0.6, np.nan],
            "EXT_SOURCE_3": [0.6, np.nan, np.nan],
        }
    )
    result = add_ratios(frame)
    assert result.loc[0, "CREDIT_INCOME_RATIO"] == 2
    assert result.loc[0, "ANNUITY_INCOME_RATIO"] == 0.1
    assert result.loc[0, "CREDIT_TERM"] == 20
    assert result.loc[0, "EMPLOYED_AGE_RATIO"] == 0.1
    assert result.loc[0, "EXT_SOURCE_MEAN"] == pytest.approx(0.4)
    assert result.loc[1, "EXT_SOURCE_MEAN"] == 0.6
    assert pd.isna(result.loc[2, "EXT_SOURCE_MEAN"])
    assert result.loc[1:, ["CREDIT_INCOME_RATIO", "CREDIT_TERM"]].isna().all().all()
    numeric = numeric_features(result)
    assert "EMPLOYED_AGE_RATIO" not in numeric
    assert not np.isinf(numeric.to_numpy()).any()


def test_encoder_vocabulary_is_train_only_and_reusable() -> None:
    import pickle

    from src.features.encoding import FeatureEncoder

    train = pd.DataFrame(
        {
            "amount": [1.0, 2.0, np.nan],
            "job": ["A", "B", None],
            "CODE_GENDER": ["M", "F", "M"],
            "EMPLOYED_AGE_RATIO": [0.1, 0.2, 0.3],
        }
    )
    validation = pd.DataFrame(
        {"amount": [9999.0], "job": ["VAL_ONLY"], "CODE_GENDER": ["F"], "EMPLOYED_AGE_RATIO": [0.9]}
    )
    encoder = FeatureEncoder().fit(train)
    restored = pickle.loads(pickle.dumps(encoder))
    encoded = restored.transform(validation)
    assert not any("VAL_ONLY" in name for name in encoder.columns)
    assert encoded.filter(like="job_").to_numpy().sum() == 0
    assert encoded.loc[0, "amount"] == 9999
    assert list(encoded.columns) == list(encoder.transform(train).columns)
    assert not EXCLUDED.intersection(encoded.columns)
    assert "EMPLOYED_AGE_RATIO" not in encoder.numeric_columns
    with pytest.raises(ValueError, match="Unexpected feature inputs"):
        encoder.transform(validation.assign(unexpected=1))


def test_engineered_exports_use_one_schema_and_preserve_audit(
    tmp_path, applications: pd.DataFrame, config: dict
) -> None:
    import json
    import pickle

    from src.features.build_features import build_dataset

    interim = tmp_path / "interim"
    output = tmp_path / "processed"
    interim.mkdir()
    for name, frame in prepare_splits(applications, config).items():
        frame.to_csv(interim / f"{name}.csv")
    config["paths"]["interim"] = str(interim)
    config["paths"]["processed"] = str(output)
    for name in ("use_bureau", "use_previous_application", "use_installments"):
        config["features"][name] = False
    counts = build_dataset(config)
    assert counts == {"train": 288, "val": 96, "test": 96, "production": 120}
    columns = json.loads((output / "feature_columns.json").read_text())
    encoder = pickle.loads((output / "feature_encoder.pkl").read_bytes())
    assert encoder.columns == columns
    assert encoder.__class__.__module__ == "src.features.encoding"
    assert {"CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "EXT_SOURCE_MEAN"} <= set(
        columns
    )
    assert not EXCLUDED.intersection(columns)
    for name in counts:
        features = pd.read_csv(output / f"X_{name}.csv", index_col="SK_ID_CURR")
        labels = pd.read_csv(output / f"y_{name}.csv", index_col="SK_ID_CURR")
        audit = pd.read_csv(output / f"sensitive_{name}.csv", index_col="SK_ID_CURR")
        assert list(features.columns) == columns
        assert features.index.equals(labels.index) and features.index.equals(audit.index)
        assert "EMPLOYED_AGE_RATIO" in audit
        assert not np.isinf(features.to_numpy()).any()
