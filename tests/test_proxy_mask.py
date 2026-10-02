"""Disabled proxies cannot change scores or receive SHAP attribution."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features.proxy_mask import ProxyMask
from src.models.experiments import build_pipeline
from src.models.explain import CreditExplainer


def test_masked_proxy_does_not_affect_scoring_or_shap(config):
    x = pd.DataFrame({"signal": np.tile([0, 1], 40), "proxy": np.arange(80, dtype=float)})
    y = x.signal.to_numpy()
    config["train"]["lightgbm"].update(n_estimators=10, num_leaves=3, min_child_samples=2)
    model = build_pipeline("lightgbm", config)
    pipeline = Pipeline([("proxy_mask", ProxyMask(("proxy",))), *model.steps]).fit(x, y)
    altered = x.iloc[:4].copy()
    altered["proxy"] = [float("nan"), 1e10, -1e10, 999]
    np.testing.assert_allclose(pipeline.predict_proba(x.iloc[:4]), pipeline.predict_proba(altered))
    explainer = CreditExplainer({"pipeline": pipeline, "columns": list(x)})
    explanations = explainer.explain(altered)
    np.testing.assert_allclose(explanations.values[:, 1], 0)
    assert explanations.feature_names == list(x)
    assert pipeline[:-1].transform(altered).shape == altered.shape
    assert x.proxy.iloc[3] == 3  # Input is never mutated.


def test_mask_rejects_unknown_columns_and_schema_changes():
    x = pd.DataFrame({"signal": [1], "proxy": [2]})
    with pytest.raises(ValueError, match="Unknown"):
        ProxyMask(("missing",)).fit(x)
    mask = ProxyMask(("proxy",)).fit(x)
    with pytest.raises(ValueError, match="schema"):
        mask.transform(x[["proxy", "signal"]])
    with pytest.raises(ValueError, match="schema"):
        mask.transform(x.drop(columns="proxy"))
