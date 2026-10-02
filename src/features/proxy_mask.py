"""Disable selected proxy inputs without changing the external feature schema."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ProxyMask(TransformerMixin, BaseEstimator):
    """Replace configured columns by zero at both fit and inference time."""

    def __init__(self, columns: tuple[str, ...] = ()) -> None:
        self.columns = columns

    def fit(self, x: pd.DataFrame, y=None):
        """Freeze the ordered input schema, without labels or audit attributes."""
        if not isinstance(x, pd.DataFrame) or not x.columns.is_unique:
            raise ValueError("Proxy masking requires uniquely named DataFrame columns.")
        if set(self.columns) - set(x.columns):
            raise ValueError("Unknown proxy columns.")
        self.feature_names_in_ = x.columns.to_numpy()
        self.n_features_in_ = len(x.columns)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Keep order and index; selected values cannot affect the fitted model."""
        check_is_fitted(self)
        if not isinstance(x, pd.DataFrame) or list(x.columns) != list(self.feature_names_in_):
            raise ValueError("Proxy mask input differs from the fitted schema.")
        result = x.copy()
        result.loc[:, list(self.columns)] = 0.0
        return result
