"""Exact SHAP contributions in calibrated log-odds for a sigmoid mixture."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
from scipy.special import expit
from sklearn.pipeline import Pipeline

from src.config import write_json
from src.models.calibration import CalibratedMixture
from src.models.explain import CreditExplainer, feature_label


class CalibratedExplainer(CreditExplainer):
    """Transform probability SHAP linearly before applying the sigmoid link."""

    def __init__(self, bundle: dict):
        self.calibrated_pipeline = bundle["pipeline"]
        self.calibrator = self.calibrated_pipeline[-1]
        raw_pipeline = Pipeline(
            self.calibrated_pipeline.steps[:-1] + [("model", self.calibrator.mixture)]
        )
        super().__init__({**bundle, "pipeline": raw_pipeline})

    def explain(self, x):
        raw = super().explain(x)
        result = shap.Explanation(
            values=self.calibrator.slope * raw.values,
            base_values=self.calibrator.slope * raw.base_values + self.calibrator.intercept,
            data=raw.data,
            feature_names=self.columns,
        )
        np.testing.assert_allclose(
            expit(result.base_values + result.values.sum(axis=1)),
            self.calibrated_pipeline.predict_proba(x)[:, 1],
            atol=1e-6,
            rtol=1e-6,
        )
        return result

    def top_reasons(self, x, k=3):
        return [{**reason, "units": "calibrated_log_odds"} for reason in super().top_reasons(x, k)]


def make_explainer(bundle: dict) -> CreditExplainer:
    """Preserve the original explainer for uncalibrated historical models."""
    if isinstance(bundle["pipeline"][-1], CalibratedMixture):
        return CalibratedExplainer(bundle)
    return CreditExplainer(bundle)


def create_calibrated_shap(bundle: dict, x, output: Path, sample_size=100, seed=20261016):
    """Export required plots with calibrated log-odds units and reconstruction proof."""
    output.mkdir(parents=True, exist_ok=True)
    explainer = CalibratedExplainer(bundle)
    sample = x.sample(n=min(sample_size, len(x)), random_state=seed)
    explanation = explainer.explain(sample)
    explanation.feature_names = [feature_label(column) for column in bundle["columns"]]
    for name, plot in [("summary", shap.plots.beeswarm), ("bar", shap.plots.bar)]:
        plot(explanation, max_display=15, show=False)
        plt.gcf().set_size_inches(11, 7)
        plt.savefig(output / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    scores = bundle["pipeline"].predict_proba(x)[:, 1]
    accepted = np.where(scores < bundle["threshold"])[0]
    refused = np.where(scores >= bundle["threshold"])[0]
    if not len(accepted) or not len(refused):
        raise ValueError("Both decision outcomes are required for SHAP evidence.")
    choices = {
        "accepted": accepted[np.argmin(scores[accepted])],
        "refused": refused[np.argmax(scores[refused])],
        "borderline": np.argmin(abs(scores - bundle["threshold"])),
    }
    for label, position in choices.items():
        local = explainer.explain(x.iloc[[position]])[0]
        local.feature_names = explanation.feature_names
        shap.plots.waterfall(local, max_display=10, show=False)
        plt.title(f"{label} | calibrated log-odds; PD={scores[position]:.4f}")
        plt.savefig(output / f"waterfall_{label}.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    error = np.max(
        abs(
            expit(explanation.base_values + explanation.values.sum(axis=1))
            - bundle["pipeline"].predict_proba(sample)[:, 1]
        )
    )
    report = {
        "model_run_id": bundle["run_id"],
        "feature_count": len(bundle["columns"]),
        "sample_rows": len(sample),
        "shap_version": shap.__version__,
        "units": "calibrated_log_odds",
        "link": "probability = sigmoid(base_value + sum(SHAP))",
        "max_probability_reconstruction_error": float(error),
        "limitations": "Model associations, not causes; missing inputs are imputed.",
    }
    write_json(output / "shap_report.json", report)
    return report
