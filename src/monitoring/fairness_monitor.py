"""Monitor acceptance independently of delayed default outcomes."""

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate

from src.models.fairness import audit_fairness


def monitor_fairness(
    predictions: pd.Series,
    sensitive: pd.DataFrame,
    settings: dict,
    labels: pd.Series | None = None,
    scores: pd.Series | None = None,
    min_group_size: int = 1,
) -> dict:
    """Report acceptance DI now and default TPR gaps only after labels arrive."""
    if not predictions.index.equals(sensitive.index):
        raise ValueError("Prediction/audit indices differ.")
    if not predictions.isin([0, 1]).all():
        raise ValueError("Predictions must be binary refusals.")
    attributes = {}
    for attribute in settings["sensitive_features"]:
        frame = MetricFrame(
            metrics=selection_rate,
            y_true=np.zeros(len(predictions)),
            y_pred=1 - predictions.to_numpy(),
            sensitive_features=sensitive[attribute],
        )
        rates = frame.by_group
        required = settings["required_groups"][attribute]
        counts = sensitive[attribute].value_counts()
        eligible = (
            rates.reindex(required).where(counts.reindex(required) >= min_group_size).dropna()
        )
        complete = len(eligible) == len(required) and eligible.max() > 0
        attributes[attribute] = {
            "acceptance_rates": {str(key): float(value) for key, value in rates.items()},
            "disparate_impact": float(eligible.min() / eligible.max()) if complete else None,
            "sufficient_support": bool(complete),
            "minimum_group_size": min_group_size,
            "demographic_parity_difference": float(eligible.max() - eligible.min())
            if complete
            else None,
            "group_counts": {
                str(key): int(value) for key, value in sensitive[attribute].value_counts().items()
            },
        }
    result = {
        "positive_definition": "1 = predicted default/refusal; selection = 1 - prediction",
        "attributes": attributes,
        "labels_available": labels is not None,
    }
    if labels is not None:
        if not predictions.index.equals(labels.index):
            raise ValueError("Delayed label indices differ.")
        result["outcome_audit"] = audit_fairness(labels, predictions, sensitive, settings, scores)
    return result
