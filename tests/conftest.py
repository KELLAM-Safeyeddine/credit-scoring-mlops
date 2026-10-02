"""Small synthetic labeled applications; no Kaggle data in tests."""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config


@pytest.fixture
def config() -> dict:
    cfg = load_config()
    cfg["data"]["sample_frac"] = 1.0
    return cfg


@pytest.fixture
def applications() -> pd.DataFrame:
    n = 600
    rng = np.random.default_rng(42)
    employed = -rng.integers(100, 3000, n)
    employed[::31] = 365243
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100000, 100000 + n),
            "TARGET": (np.arange(n) % 12 == 0).astype("int64"),
            "AMT_INCOME_TOTAL": rng.uniform(10000, 300000, n),
            "AMT_CREDIT": rng.uniform(100000, 1000000, n),
            "CODE_GENDER": np.where(np.arange(n) % 2, "M", "F"),
            "DAYS_BIRTH": -rng.integers(18 * 365, 75 * 365, n),
            "DAYS_EMPLOYED": employed,
            "EXT_SOURCE_1": rng.uniform(0, 1, n),
        }
    )
