"""Larger campaigns preserve every customer reserved by previous experiments."""

import pandas as pd
import pytest

from src.data.expand_development import expand_splits


def memberships(frame):
    ids = frame.SK_ID_CURR
    return {
        "train": pd.Index(ids[:60]),
        "val": pd.Index(ids[60:80]),
        "test": pd.Index(ids[80:100]),
        "production": pd.Index(ids[100:120]),
    }


def test_expansion_preserves_old_reservations(applications, config):
    old = memberships(applications)
    result = expand_splits(applications, old, config, 0.4)
    assert sum(map(len, result.values())) == 240
    seen = set()
    for name, part in result.items():
        assert set(old[name]).issubset(part.index)
        assert not seen.intersection(part.index)
        seen.update(part.index)
        assert (
            part.loc[old[name], "TARGET"].tolist()
            == applications.set_index("SK_ID_CURR").loc[old[name], "TARGET"].tolist()
        )
    assert not set(old["test"]).intersection(result["train"].index)
    assert not set(old["production"]).intersection(result["val"].index)


@pytest.mark.parametrize("failure", ["overlap", "smaller", "unknown"])
def test_expansion_rejects_unsafe_memberships(applications, config, failure):
    old = memberships(applications)
    fraction = 0.4
    if failure == "overlap":
        old["val"] = old["train"]
    elif failure == "smaller":
        fraction = 0.1
    else:
        old["test"] = pd.Index([-1])
    with pytest.raises(ValueError):
        expand_splits(applications, old, config, fraction)
