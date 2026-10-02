"""Explicit retraining rules; insufficient evidence is never labelled fair."""


def evaluate_alerts(
    drift: dict, fairness: dict, settings: dict, max_score_psi: float | None = None
) -> dict:
    """Keep raw metrics visible and apply the configured AUC decision rule."""
    alerts = []
    unavailable = []
    if drift["drift_feature_share"] > settings["max_drift_feature_share"]:
        alerts.append("data_drift")
    if drift["score_psi"] > (settings["max_score_psi"] if max_score_psi is None else max_score_psi):
        alerts.append("prediction_drift")
    for column, psi in drift.get("critical_feature_psi", {}).items():
        if psi > settings["critical_feature_psi"]:
            alerts.append(f"data_drift.{column}")
    for column, delta in drift.get("missingness_delta", {}).items():
        if delta > settings["max_missingness_delta"]:
            alerts.append(f"data_missingness.{column}")
    for column, share in drift.get("constant_emergence", {}).items():
        if share > settings["max_constant_emergence"]:
            alerts.append(f"data_novel_constant.{column}")
    for attribute, psi in drift.get("population_psi", {}).items():
        if psi > settings["max_population_psi"]:
            alerts.append(f"population_drift.{attribute}")
    rule = settings["auc_alert_rule"]
    if rule not in {"confidence_upper_below_floor", "point_estimate_below_floor"}:
        raise ValueError("Unknown AUC alert rule.")
    value = drift.get("auc_upper") if rule == "confidence_upper_below_floor" else drift.get("auc")
    if value is None:
        unavailable.append("performance")
    elif value < settings["min_auc"]:
        alerts.append("performance")
    for attribute, report in fairness["attributes"].items():
        if report["disparate_impact"] is None:
            unavailable.append(f"fairness.{attribute}")
        elif report["disparate_impact"] < settings["min_disparate_impact"]:
            alerts.append(f"fairness.{attribute}")
    return {
        "trigger_retraining": bool(alerts),
        "alerts": alerts,
        "not_evaluable": unavailable,
        "auc_alert_rule": rule,
        "raw_auc_below_floor": drift.get("auc") is not None and drift["auc"] < settings["min_auc"],
    }
