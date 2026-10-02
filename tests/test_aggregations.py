"""Test financial aggregates, historical cutoffs and fractional payments."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.aggregations import (
    aggregate_bureau,
    aggregate_installments,
    aggregate_previous_application,
)


def test_bureau_aggregates_across_chunks(tmp_path: Path) -> None:
    path = tmp_path / "bureau.csv"
    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2, 1, 99],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Active", "Active"],
            "DAYS_CREDIT": [-1, -2, -3, 1, -1],
            "AMT_CREDIT_SUM": [100, 200, np.nan, 999, 999],
            "AMT_CREDIT_SUM_DEBT": [20, np.nan, np.nan, 999, 999],
            "CREDIT_DAY_OVERDUE": [5, 0, 0, 999, 999],
        }
    ).to_csv(path, index=False)
    ids = pd.Index([1, 2, 3], name="SK_ID_CURR")
    reference = aggregate_bureau(path, ids, chunksize=100)
    for size in (1, 2):
        pd.testing.assert_frame_equal(aggregate_bureau(path, ids, chunksize=size), reference)
    assert reference.loc[1, "BUREAU_COUNT"] == 2
    assert reference.loc[1, "BUREAU_ACTIVE_RATE"] == 0.5
    assert reference.loc[1, "BUREAU_CREDIT_SUM"] == 300
    assert reference.loc[1, "BUREAU_DEBT_SUM"] == 20
    assert reference.loc[1, "BUREAU_OVERDUE_MAX"] == 5
    assert reference.loc[3, "BUREAU_COUNT"] == 0
    assert pd.isna(reference.loc[2, "BUREAU_DEBT_SUM"])
    assert pd.isna(reference.loc[3, "BUREAU_ACTIVE_RATE"])


def test_previous_means_are_weighted_by_valid_rows(tmp_path: Path) -> None:
    path = tmp_path / "previous.csv"
    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 1, 2],
            "DAYS_DECISION": [-1, -2, -3, 1, -1],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Refused", "Refused", "Approved"],
            "AMT_APPLICATION": [100, 300, np.nan, 99999, 50],
            "AMT_CREDIT": [80, 260, 200, 99999, 40],
        }
    ).to_csv(path, index=False)
    ids = pd.Index([1, 2, 3], name="SK_ID_CURR")
    result = aggregate_previous_application(path, ids, chunksize=2)
    pd.testing.assert_frame_equal(result, aggregate_previous_application(path, ids, chunksize=100))
    assert result.loc[1, "PREV_COUNT"] == 3
    assert result.loc[1, "PREV_REFUSAL_RATE"] == pytest.approx(2 / 3)
    assert result.loc[1, "PREV_APPLICATION_MEAN"] == 200
    assert result.loc[1, "PREV_CREDIT_MEAN"] == 180
    assert result.loc[3, "PREV_COUNT"] == 0
    assert pd.isna(result.loc[3, "PREV_CREDIT_MEAN"])


def test_fractional_payments_are_combined_across_chunks(tmp_path: Path) -> None:
    path = tmp_path / "installments.csv"
    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 1, 2],
            "SK_ID_PREV": [10, 10, 10, 10, 20],
            "NUM_INSTALMENT_VERSION": [1, 1, 1, 1, 1],
            "NUM_INSTALMENT_NUMBER": [1, 2, 1, 3, 1],
            "DAYS_INSTALMENT": [-10, -30, -10, -1, 1],
            "DAYS_ENTRY_PAYMENT": [-12, -31, -8, 1, -1],
            "AMT_INSTALMENT": [100, 80, 100, 100, 100],
            "AMT_PAYMENT": [40, 70, 60, 999, 100],
        }
    ).to_csv(path, index=False)
    ids = pd.Index([1, 2, 3], name="SK_ID_CURR")
    result = aggregate_installments(path, ids, chunksize=1, partitions=2)
    scratch = tmp_path / "worker-scratch"
    scratch.mkdir()
    other = aggregate_installments(path, ids, chunksize=100, partitions=3, temp_dir=scratch)
    pd.testing.assert_frame_equal(result, other)
    assert list(scratch.iterdir()) == []
    assert result.loc[1, "INSTALLMENTS_COUNT"] == 2
    assert result.loc[1, "INSTALLMENTS_PAYMENT_COUNT"] == 3
    assert result.loc[1, "INSTALLMENTS_DELAY_MEAN"] == 1
    assert result.loc[1, "INSTALLMENTS_LATE_RATE"] == 0.5
    assert result.loc[1, "INSTALLMENTS_PAYMENT_GAP_MEAN"] == 5
    assert result.loc[2, "INSTALLMENTS_COUNT"] == 0
    assert pd.isna(result.loc[3, "INSTALLMENTS_DELAY_MEAN"])


@pytest.mark.parametrize(
    "function,columns",
    [
        (
            aggregate_bureau,
            [
                "SK_ID_CURR",
                "CREDIT_ACTIVE",
                "DAYS_CREDIT",
                "AMT_CREDIT_SUM",
                "AMT_CREDIT_SUM_DEBT",
                "CREDIT_DAY_OVERDUE",
            ],
        ),
        (
            aggregate_previous_application,
            [
                "SK_ID_CURR",
                "DAYS_DECISION",
                "NAME_CONTRACT_STATUS",
                "AMT_APPLICATION",
                "AMT_CREDIT",
            ],
        ),
        (
            aggregate_installments,
            [
                "SK_ID_CURR",
                "SK_ID_PREV",
                "NUM_INSTALMENT_VERSION",
                "NUM_INSTALMENT_NUMBER",
                "DAYS_INSTALMENT",
                "DAYS_ENTRY_PAYMENT",
                "AMT_INSTALMENT",
                "AMT_PAYMENT",
            ],
        ),
    ],
)
def test_no_history_is_explicitly_missing(tmp_path: Path, function, columns: list[str]) -> None:
    path = tmp_path / "empty.csv"
    pd.DataFrame(columns=columns).to_csv(path, index=False)
    result = function(path, pd.Index([1], name="SK_ID_CURR"))
    for name in result.columns:
        if name.endswith("COUNT"):
            assert result.loc[1, name] == 0
        else:
            assert pd.isna(result.loc[1, name])
