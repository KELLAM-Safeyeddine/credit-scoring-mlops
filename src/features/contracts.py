"""Model exclusions and the numeric feature schema."""

import numpy as np
import pandas as pd

AUDIT_ONLY = {"CODE_GENDER", "DAYS_BIRTH", "AGE_YEARS", "AGE_GROUP", "EMPLOYED_AGE_RATIO"}
EXCLUDED = AUDIT_ONLY | {"TARGET", "SK_ID_CURR"}


def numeric_features(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Use one column order and reject sensitive, target or missing features."""
    if columns is None:
        columns = [
            name
            for name in frame.select_dtypes(include=["number", "bool"]).columns
            if name not in EXCLUDED
        ]
    if EXCLUDED.intersection(columns):
        raise ValueError("Sensitive variables, identifiers and TARGET are forbidden features.")
    if missing := sorted(set(columns) - set(frame.columns)):
        raise ValueError(f"Missing model columns: {missing}")
    return frame.loc[:, columns].replace([np.inf, -np.inf], np.nan).astype("float32")
