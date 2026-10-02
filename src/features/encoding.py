"""Train-fitted categorical encoding with a reusable inference contract."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.features.contracts import EXCLUDED, numeric_features


class FeatureEncoder:
    """Fit category vocabulary on train and preserve raw numeric missingness."""

    def fit(self, train: pd.DataFrame) -> "FeatureEncoder":
        """Learn numeric columns and categories exclusively from training rows."""
        self.numeric_columns = list(numeric_features(train).columns)
        self.categorical_columns = [
            name
            for name in train.select_dtypes(include=["object", "string", "category"]).columns
            if name not in EXCLUDED
        ]
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        if self.categorical_columns:
            self.encoder.fit(self._categories(train))
            categories = list(self.encoder.get_feature_names_out(self.categorical_columns))
        else:
            categories = []
        self.columns = self.numeric_columns + categories
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("Encoded feature names collide.")
        return self

    def _categories(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fill categories with a constant token, without population statistics."""
        return frame[self.categorical_columns].astype("string").fillna("__MISSING__")

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the frozen schema; unseen categories map to all-zero indicators."""
        expected = set(self.numeric_columns + self.categorical_columns)
        actual = set(frame.columns) - EXCLUDED
        if actual != expected:
            raise ValueError(
                f"Unexpected feature inputs: missing={expected - actual}, extra={actual - expected}"
            )
        numeric = numeric_features(frame, self.numeric_columns)
        if self.categorical_columns:
            encoded = pd.DataFrame(
                self.encoder.transform(self._categories(frame)),
                index=frame.index,
                columns=self.columns[len(self.numeric_columns) :],
            )
            numeric = pd.concat([numeric, encoded], axis=1)
        return numeric.loc[:, self.columns]
