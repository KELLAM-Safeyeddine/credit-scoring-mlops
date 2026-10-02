"""Performance at the validation-selected refusal threshold."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def performance_metrics(
    y: pd.Series, probabilities: np.ndarray, threshold: float, costs: dict
) -> dict[str, float]:
    """Compute ranking, operating-point and total business-cost metrics."""
    probabilities = np.asarray(probabilities)
    if len(y) != len(probabilities) or not np.isfinite(probabilities).all():
        raise ValueError("Labels and finite probabilities must align.")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Performance requires both binary target classes.")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must lie in [0, 1].")
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(y, probabilities))
    fpr, tpr, _ = roc_curve(y, probabilities)
    return {
        "validation_auc": auc,
        "validation_pr_auc": float(average_precision_score(y, probabilities)),
        "validation_ks": float(np.max(tpr - fpr)),
        "validation_gini": 2 * auc - 1,
        "validation_recall": float(recall_score(y, predictions, zero_division=0)),
        "validation_precision": float(precision_score(y, predictions, zero_division=0)),
        "validation_business_cost": float(
            fn * costs["false_negative_cost"] + fp * costs["false_positive_cost"]
        ),
        "validation_acceptance_rate": float(np.mean(~predictions)),
        "validation_tn": float(tn),
        "validation_fp": float(fp),
        "validation_fn": float(fn),
        "validation_tp": float(tp),
        "threshold": float(threshold),
    }
