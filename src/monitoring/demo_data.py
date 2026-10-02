"""Paired synthetic cohorts for an isolated, reproducible lifecycle demonstration."""

import numpy as np
import pandas as pd


def cohort(seed: int, size: int = 400) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Pair all gender/age groups on identical outcomes; keep proxies explicit."""
    rng = np.random.default_rng(seed)
    labels = np.arange(size) % 2
    strong = np.clip(0.8 - 0.6 * labels + rng.normal(0, 0.3, size), 0, 1)
    weak = np.clip(0.7 - 0.4 * labels + rng.normal(0, 0.3, size), 0, 1)
    features, truths, audits = [], [], []
    for gender in ("F", "M"):
        for age in ("<30", "30-45", "45-60", "60+"):
            features.append(
                pd.DataFrame(
                    {
                        "AMT_INCOME_TOTAL": np.full(size, 40000.0 if gender == "F" else 100000.0),
                        "AMT_CREDIT": np.full(size, 200000.0),
                        "AMT_ANNUITY": np.full(size, 12000.0),
                        "EXT_SOURCE_1": strong,
                        "EXT_SOURCE_2": weak,
                    }
                )
            )
            truths.append(pd.Series(labels, name="TARGET"))
            audits.append(
                pd.DataFrame({"CODE_GENDER": gender, "AGE_GROUP": age}, index=range(size))
            )
    result = (
        pd.concat(features, ignore_index=True),
        pd.concat(truths, ignore_index=True),
        pd.concat(audits, ignore_index=True),
    )
    for frame in result:
        frame.index = pd.Index(seed * 100000 + np.arange(len(frame)), name="SK_ID_CURR")
    return result


def training_cohort(version: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Simulate a weak initial sensor, then sampling bias, then its mitigation."""
    x, y, sensitive = cohort(42)
    if version == 1:
        x["EXT_SOURCE_1"] = np.tile(np.random.default_rng(7).uniform(0, 1, 400), 8)
        return x, y, sensitive
    if version not in (2, 3):
        raise ValueError("Demo version must be 1, 2 or 3.")
    favored = (sensitive.CODE_GENDER.eq("F") & y.eq(1)) | (sensitive.CODE_GENDER.eq("M") & y.eq(0))
    ids = np.repeat(np.arange(len(x)), np.where(favored, 8, 1))
    result = tuple(frame.iloc[ids].reset_index(drop=True) for frame in (x, y, sensitive))
    for frame in result:
        frame.index = pd.Index(42 * 100000 + np.arange(len(frame)), name="SK_ID_CURR")
    return result
