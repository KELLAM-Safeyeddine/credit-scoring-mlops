"""Fairlearn audits with explicit refusal and acceptance semantics."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import roc_auc_score


def acceptance_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Selection is acceptance: one minus the predicted default label."""
    return float(np.mean(1 - np.asarray(y_pred)))


def default_tpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall of actual defaulters; undefined when there are no defaults."""
    mask = np.asarray(y_true) == 1
    return float(np.mean(np.asarray(y_pred)[mask])) if mask.any() else float("nan")


def default_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Refusal of actual non-defaulters; undefined without non-defaulters."""
    mask = np.asarray(y_true) == 0
    return float(np.mean(np.asarray(y_pred)[mask])) if mask.any() else float("nan")


def intersection_groups(sensitive: pd.DataFrame) -> pd.Series:
    """Make explicit genre/age intersections for mitigation fitting only."""
    return sensitive["CODE_GENDER"].astype(str) + "|" + sensitive["AGE_GROUP"].astype(str)


def reweighing_weights(y: pd.Series, groups: pd.Series) -> np.ndarray:
    """Estimate P(group) P(label) / P(group,label) on training observations."""
    if not y.index.equals(groups.index) or y.isna().any() or groups.isna().any():
        raise ValueError("Training labels/groups must align and be non-null.")
    table = pd.DataFrame({"label": y, "group": groups})
    cells = table.groupby(["group", "label"]).size()
    group_counts = table.groupby("group").size()
    label_counts = table.groupby("label").size()
    weights = np.array(
        [
            group_counts[group] * label_counts[label] / (len(y) * cells[group, label])
            for group, label in zip(groups, y, strict=True)
        ],
        dtype=float,
    )
    return weights / weights.mean()


def _nullable(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def audit_fairness(
    y: pd.Series,
    predictions: np.ndarray,
    sensitive: pd.DataFrame,
    settings: Mapping[str, Any],
    scores: np.ndarray | None = None,
) -> dict[str, Any]:
    """Audit all groups and compare sufficiently supported groups only."""
    predictions = np.asarray(predictions)
    if not y.index.equals(sensitive.index) or len(predictions) != len(y):
        raise ValueError("Fairness inputs must have identical aligned indices/lengths.")
    if not np.isin(predictions, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Audit requires binary default labels and predictions.")
    report: dict[str, Any] = {
        "positive_definition": "1 = default predicted = refusal",
        "selection_definition": "acceptance = 1 - predicted_default",
        "equal_opportunity_definition": "max-min TPR of refusal among actual defaults",
        "support_policy": {
            "min_group_size": settings.get("min_group_size", 100),
            "min_class_count": settings.get("min_class_count", 20),
            "unknown_gender": "XNA reported but outside known-gender comparisons",
        },
        "attributes": {},
    }
    for attribute in settings["sensitive_features"]:
        groups = sensitive[attribute]
        if groups.isna().any():
            raise ValueError(f"Missing sensitive attribute: {attribute}")
        frame = MetricFrame(
            metrics={"acceptance_rate": acceptance_rate, "tpr": default_tpr, "fpr": default_fpr},
            y_true=y,
            y_pred=predictions,
            sensitive_features=groups,
        )
        rows = {}
        eligible = []
        for group, values in frame.by_group.iterrows():
            mask = groups == group
            positives = int(y[mask].sum())
            negatives = int(mask.sum()) - positives
            supported = (
                mask.sum() >= settings.get("min_group_size", 100)
                and min(positives, negatives) >= settings.get("min_class_count", 20)
                and not (attribute == "CODE_GENDER" and group == "XNA")
            )
            rows[str(group)] = {
                "rows": int(mask.sum()),
                "defaults": positives,
                "non_defaults": negatives,
                "default_rate": float(y[mask].mean()),
                "comparison_eligible": bool(supported),
                **{key: _nullable(value) for key, value in values.items()},
                "refusal_rate": 1 - float(values.acceptance_rate),
                "auc": float(roc_auc_score(y[mask], np.asarray(scores)[mask]))
                if scores is not None and min(positives, negatives) > 0
                else None,
            }
            if supported:
                eligible.append(str(group))
        required = settings.get("required_groups", {}).get(attribute, [])
        sufficient = len(eligible) >= 2 and set(required).issubset(eligible)
        summary: dict[str, Any] = {"eligible_groups": eligible, "sufficient_support": sufficient}
        if sufficient:
            accepted = [rows[group]["acceptance_rate"] for group in eligible]
            tpr = [rows[group]["tpr"] for group in eligible]
            fpr = [rows[group]["fpr"] for group in eligible]
            summary.update(
                {
                    "disparate_impact": min(accepted) / max(accepted)
                    if max(accepted) > 0
                    else None,
                    "demographic_parity_difference": max(accepted) - min(accepted),
                    "equal_opportunity_difference": max(tpr) - min(tpr),
                    "equalized_odds_difference": max(max(tpr) - min(tpr), max(fpr) - min(fpr)),
                }
            )
        else:
            summary.update(
                dict.fromkeys(
                    [
                        "disparate_impact",
                        "demographic_parity_difference",
                        "equal_opportunity_difference",
                        "equalized_odds_difference",
                    ]
                )
            )
        summary["passes_thresholds"] = bool(
            sufficient
            and summary["disparate_impact"] is not None
            and summary["disparate_impact"] >= settings["min_disparate_impact"]
            and summary["equal_opportunity_difference"] <= settings["max_equal_opp_diff"]
        )
        summary["comparison_coverage"] = sum(rows[g]["rows"] for g in eligible) / len(y)
        report["attributes"][attribute] = {"by_group": rows, **summary}
    report["passes_thresholds"] = all(
        attr["passes_thresholds"] for attr in report["attributes"].values()
    )
    return report
