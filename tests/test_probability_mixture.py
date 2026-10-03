"""Validate frozen probability averaging and additive ensemble SHAP."""

import pickle

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.models.explain import CreditExplainer
from src.models.probability_mixture import ProbabilityMixture


@pytest.fixture
def fitted_ensemble():
    rng = np.random.default_rng(11)
    x = pd.DataFrame(
        rng.normal(size=(120, 3)), columns=["AMT_CREDIT", "EXT_SOURCE_2", "AMT_INCOME_TOTAL"]
    )
    y = (x.AMT_CREDIT + x.EXT_SOURCE_2 * 0.8 > 0).astype(int)
    imputer = SimpleImputer().fit(x)
    transformed = imputer.transform(x)
    models = tuple(
        LGBMClassifier(
            n_estimators=15,
            num_leaves=5,
            min_child_samples=8,
            n_jobs=1,
            verbosity=-1,
            random_state=seed,
        ).fit(transformed, y)
        for seed in (1, 2)
    )
    mixture = ProbabilityMixture(models, (0.3, 0.7), transformed[:20])
    return (
        x,
        models,
        {"pipeline": Pipeline([("imputer", imputer), ("model", mixture)]), "columns": list(x)},
    )


def test_frozen_scores_roundtrip_and_no_refit(fitted_ensemble):
    x, models, bundle = fitted_ensemble
    pipeline = bundle["pipeline"]
    expected = sum(
        weight * model.predict_proba(pipeline[0].transform(x))[:, 1]
        for model, weight in zip(models, (0.3, 0.7), strict=True)
    )
    np.testing.assert_allclose(pipeline.predict_proba(x)[:, 1], expected)
    np.testing.assert_array_equal(
        pickle.loads(pickle.dumps(pipeline)).predict_proba(x), pipeline.predict_proba(x)
    )
    with pytest.raises(RuntimeError, match="retrained"):
        pipeline[-1].fit(x)


def test_probability_shap_reconstructs_ensemble(fitted_ensemble):
    x, _, bundle = fitted_ensemble
    explained = CreditExplainer(bundle).explain(x.iloc[:5])
    np.testing.assert_allclose(
        explained.base_values + explained.values.sum(axis=1),
        bundle["pipeline"].predict_proba(x.iloc[:5])[:, 1],
        atol=1e-6,
    )
    assert len(CreditExplainer(bundle).top_reasons(x.iloc[:1])) == 3
    with pytest.raises(ValueError, match="ordered"):
        CreditExplainer(bundle).explain(x.iloc[:2, ::-1])


def test_invalid_weights_and_background_fail_closed(fitted_ensemble):
    _, models, bundle = fitted_ensemble
    background = bundle["pipeline"][-1].background
    for weights in ((-0.1, 1.1), (0.0, 1.0), (np.nan, 1.0)):
        with pytest.raises(ValueError, match="weights"):
            ProbabilityMixture(models, weights, background)
    with pytest.raises(ValueError, match="background"):
        ProbabilityMixture(models, (0.5, 0.5), background[:, :2])
