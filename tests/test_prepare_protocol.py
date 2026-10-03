"""Preparing prospective data must use exactly the frozen IDs without resplitting."""

import pandas as pd
import pytest

from src.data.prepare_protocol import select_protocol_rows


def test_preparation_keeps_exact_reservations_and_source_labels(applications, config):
    ids = applications.SK_ID_CURR
    memberships = {
        "train": pd.Index(ids[:20]),
        "val": pd.Index(ids[20:30]),
        "test": pd.Index(ids[30:40]),
        "production": pd.Index(ids[40:50]),
    }
    prepared = select_protocol_rows(applications, memberships, config)
    for name, frame in prepared.items():
        assert set(frame.index) == set(memberships[name])
        expected = applications.set_index("SK_ID_CURR").loc[frame.index, "TARGET"]
        pd.testing.assert_series_equal(frame.TARGET, expected)
    assert prepared["train"].EMPLOYED_ANOMALY.iloc[0] == 1
    assert pd.isna(prepared["train"].DAYS_EMPLOYED.iloc[0])
    assert applications.DAYS_EMPLOYED.iloc[0] == 365243


@pytest.mark.parametrize("failure", ["overlap", "unknown", "duplicate"])
def test_preparation_rejects_invalid_reservations(applications, config, failure):
    first = applications.SK_ID_CURR.iloc[0]
    second = applications.SK_ID_CURR.iloc[1]
    memberships = {"train": pd.Index([first]), "test": pd.Index([second])}
    if failure == "overlap":
        memberships["test"] = pd.Index([first])
    elif failure == "unknown":
        memberships["test"] = pd.Index([-1])
    else:
        memberships["train"] = pd.Index([first, first])
    with pytest.raises(ValueError, match="memberships"):
        select_protocol_rows(applications, memberships, config)
