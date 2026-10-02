"""Tree SHAP reconstructs actual predictions and respects the feature contract."""

import numpy as np
import pandas as pd
import pytest

from src.models.experiments import build_pipeline
from src.models.explain import CreditExplainer, feature_label


def test_shap_additivity_and_top_three_on_imputed_tree(config):
    config["train"]["lightgbm"].update(n_estimators=12, min_child_samples=2, num_leaves=3)
    x = pd.DataFrame(
        {
            "EXT_SOURCE_2": [0.1, 0.2, 0.8, 0.9] * 8,
            "AMT_CREDIT": [100.0, np.nan, 300.0, 400.0] * 8,
            "CREDIT_INCOME_RATIO": [1.0, 2.0, 3.0, 4.0] * 8,
        }
    )
    pipeline = build_pipeline("lightgbm", config).fit(x, [1, 1, 0, 0] * 8)
    explainer = CreditExplainer({"pipeline": pipeline, "columns": list(x.columns)})
    values = explainer.explain(x.iloc[:4])
    scores = 1 / (1 + np.exp(-(values.base_values + values.values.sum(axis=1))))
    np.testing.assert_allclose(scores, pipeline.predict_proba(x.iloc[:4])[:, 1], atol=1e-7)
    reasons = explainer.top_reasons(x.iloc[[1]])
    assert len(reasons) == 3
    magnitudes = [abs(reason["contribution_raw_score"]) for reason in reasons]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all(np.isfinite(reason["transformed_value"]) for reason in reasons)
    assert reasons[0]["label"]
    with pytest.raises(ValueError, match="ordered"):
        explainer.explain(x[list(reversed(x.columns))])
    with pytest.raises(ValueError, match="one observation"):
        explainer.top_reasons(x.head(2))


def test_sensitive_columns_cannot_be_explained_as_model_features():
    with pytest.raises(ValueError, match="Audit"):
        CreditExplainer({"pipeline": None, "columns": ["CODE_GENDER"]})


def test_labels_translate_categories_without_losing_source_meaning():
    assert (
        feature_label("NAME_EDUCATION_TYPE_Higher education")
        == "Niveau de formation : Higher education"
    )
    assert feature_label("CREDIT_INCOME_RATIO") == "Crédit demandé rapporté au revenu"
