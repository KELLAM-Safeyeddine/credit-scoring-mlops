"""Test raw validation, safe extraction and split isolation."""

import zipfile
from itertools import combinations
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import pytest

from scripts.extract_data import REQUIRED_CSVS, extract_archive
from src.data.make_dataset import prepare_splits
from src.data.validate import validate_application


def test_valid_schema(applications: pd.DataFrame) -> None:
    assert len(validate_application(applications)) == len(applications)


@pytest.mark.parametrize(
    "column,value",
    [
        ("TARGET", 2),
        ("AMT_INCOME_TOTAL", 0),
        ("CODE_GENDER", "U"),
        ("DAYS_BIRTH", 0),
        ("SK_ID_CURR", None),
    ],
)
def test_invalid_schema(applications: pd.DataFrame, column: str, value: object) -> None:
    applications.loc[0, column] = value
    with pytest.raises(pa.errors.SchemaErrors):
        validate_application(applications)


def test_duplicate_ids_rejected(applications: pd.DataFrame) -> None:
    applications.loc[0, "SK_ID_CURR"] = applications.loc[1, "SK_ID_CURR"]
    with pytest.raises(pa.errors.SchemaErrors):
        validate_application(applications)


def test_unlabeled_application_test_rejected(applications: pd.DataFrame) -> None:
    with pytest.raises(pa.errors.SchemaErrors):
        validate_application(applications.drop(columns="TARGET"))


def test_splits_are_disjoint_and_stratified(applications: pd.DataFrame, config: dict) -> None:
    splits = prepare_splits(applications, config)
    assert {name: len(frame) for name, frame in splits.items()} == {
        "train": 288,
        "val": 96,
        "test": 96,
        "production": 120,
    }
    for first, second in combinations(splits.values(), 2):
        assert not set(first.index) & set(second.index)
    assert set().union(*(set(frame.index) for frame in splits.values())) == set(
        applications.SK_ID_CURR
    )
    for frame in splits.values():
        assert abs(frame.TARGET.mean() - applications.TARGET.mean()) < 0.02
        assert frame.index.name == "SK_ID_CURR"
        assert (frame.DAYS_EMPLOYED != 365243).all()
        assert frame.loc[frame.EMPLOYED_ANOMALY == 1, "DAYS_EMPLOYED"].isna().all()
        assert frame.AGE_GROUP.notna().all()
    repeated = prepare_splits(applications, config)
    for name, frame in splits.items():
        pd.testing.assert_frame_equal(frame, repeated[name])


def test_extraction_is_idempotent_and_repairs_missing_files(tmp_path: Path) -> None:
    with zipfile.ZipFile(tmp_path / "source.zip", "w") as archive:
        for name in REQUIRED_CSVS:
            archive.writestr(name, "a,b\n1,2\n")
    manifest = extract_archive(tmp_path)
    raw = tmp_path / "data/raw"
    before = {name: (raw / name).stat().st_mtime_ns for name in REQUIRED_CSVS}
    assert extract_archive(tmp_path) == manifest
    assert before == {name: (raw / name).stat().st_mtime_ns for name in REQUIRED_CSVS}
    (raw / "bureau.csv").unlink()
    extract_archive(tmp_path)
    assert (raw / "bureau.csv").read_text() == "a,b\n1,2\n"


@pytest.mark.parametrize("member", ["../outside.csv", "folder/file.csv"])
def test_unsafe_archive_rejected(tmp_path: Path, member: str) -> None:
    with zipfile.ZipFile(tmp_path / "source.zip", "w") as archive:
        archive.writestr(member, "a\n1\n")
    with pytest.raises(ValueError):
        extract_archive(tmp_path)
    assert not (tmp_path.parent / "outside.csv").exists()


def test_missing_tables_rejected(tmp_path: Path) -> None:
    with zipfile.ZipFile(tmp_path / "source.zip", "w") as archive:
        archive.writestr("application_train.csv", "a\n1\n")
    with pytest.raises(ValueError, match="Missing required"):
        extract_archive(tmp_path)


def test_cleaning_age_boundaries_and_sentinel_is_idempotent(config: dict) -> None:
    from src.data.make_dataset import clean_application

    frame = pd.DataFrame(
        {
            "DAYS_BIRTH": -365.25 * pd.Series([29.9, 30.0, 45.0, 60.0]),
            "DAYS_EMPLOYED": [365243.0, -100.0, -200.0, -300.0],
        }
    )
    cleaned = clean_application(frame, config)
    assert cleaned.AGE_GROUP.astype(str).tolist() == ["<30", "30-45", "45-60", "60+"]
    assert cleaned.loc[0, "EMPLOYED_ANOMALY"] == 1
    assert pd.isna(cleaned.loc[0, "DAYS_EMPLOYED"])
    pd.testing.assert_frame_equal(clean_application(cleaned, config), cleaned)
