"""Compare reweighing and Fairlearn mitigations on untouched validation rows."""

import importlib.metadata
import json
import pickle
import sys
import time
import warnings
from typing import Any

import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
from mlflow.models import infer_signature
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import load_config, parse_params, project_path, write_json
from src.models.evaluate import performance_metrics
from src.models.experiments import build_pipeline
from src.models.fairness import audit_fairness, intersection_groups, reweighing_weights
from src.models.policies import ReductionPolicy, ScoringPolicy, ThresholdPolicy
from src.models.train import select_threshold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def select_fair_threshold(
    y: pd.Series, scores: np.ndarray, sensitive: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[float, bool]:
    """Choose one global validation threshold under fairness and utility bounds."""
    settings = cfg["train"]["threshold"]
    fairness = cfg["fairness"]
    utility = fairness["mitigation"]
    grid = np.linspace(settings["min"], settings["max"], settings["steps"])
    refused = scores[:, None] >= grid
    labels = y.to_numpy()
    recall = refused[labels == 1].mean(axis=0)
    acceptance = 1 - refused.mean(axis=0)
    valid = (
        (recall >= utility["min_recall"])
        & (acceptance >= utility["min_acceptance_rate"])
        & (acceptance <= utility["max_acceptance_rate"])
    )
    support = audit_fairness(y, refused[:, 0].astype(int), sensitive, fairness)
    penalty = np.zeros(len(grid))
    for attribute, report in support["attributes"].items():
        if not report["sufficient_support"]:
            raise ValueError(f"Insufficient validation support for {attribute}")
        groups = sensitive[attribute].astype(str).to_numpy()
        selected, tprs = [], []
        for group in report["eligible_groups"]:
            mask = groups == group
            selected.append(1 - refused[mask].mean(axis=0))
            tprs.append(refused[mask & (labels == 1)].mean(axis=0))
        selected, tprs = np.array(selected), np.array(tprs)
        di = np.divide(
            selected.min(axis=0),
            selected.max(axis=0),
            out=np.zeros(len(grid)),
            where=selected.max(axis=0) > 0,
        )
        gap = np.ptp(tprs, axis=0)
        valid &= (di >= fairness["min_disparate_impact"]) & (gap <= fairness["max_equal_opp_diff"])
        penalty += np.maximum(fairness["min_disparate_impact"] - di, 0)
        penalty += np.maximum(gap - fairness["max_equal_opp_diff"], 0)
    fn = (~refused[labels == 1]).sum(axis=0)
    fp = refused[labels == 0].sum(axis=0)
    cost = fn * settings["false_negative_cost"] + fp * settings["false_positive_cost"]
    if valid.any():
        selected_index = np.where(valid)[0][np.argmin(cost[valid])]
        return float(grid[selected_index]), True
    # Preserve a useful decision policy when no feasible threshold exists.
    usable = (recall >= utility["min_recall"]) & (acceptance <= utility["max_acceptance_rate"])
    penalty[~usable] = np.inf
    if not np.isfinite(penalty).any():
        return select_threshold(y, scores, cfg), False
    return float(grid[int(np.argmin(penalty + cost / (1e6 * len(y))))]), False


def policy_metrics(y: pd.Series, scores: np.ndarray, decisions: np.ndarray, cfg: dict) -> dict:
    """Use actual policy decisions for confusion metrics, ranking scores for AUC."""
    metrics = performance_metrics(y, scores, 0.5, cfg["train"]["threshold"])
    decision_metrics = performance_metrics(
        y, decisions.astype(float), 0.5, cfg["train"]["threshold"]
    )
    for key in ("recall", "precision", "business_cost", "acceptance_rate", "tn", "fp", "fn", "tp"):
        metrics[f"validation_{key}"] = decision_metrics[f"validation_{key}"]
    metrics.pop("threshold")
    metrics["validation_decision_auc"] = decision_metrics["validation_auc"]
    return metrics


def proxy_analysis(
    xt: pd.DataFrame, xv: pd.DataFrame, st: pd.DataFrame, sv: pd.DataFrame, cfg: dict
) -> dict:
    """Predict known gender from allowed features; association is not causation."""
    mt, mv = st.CODE_GENDER.isin(["F", "M"]), sv.CODE_GENDER.isin(["F", "M"])
    yt = (st.loc[mt, "CODE_GENDER"] == "M").astype(int)
    yv = (sv.loc[mv, "CODE_GENDER"] == "M").astype(int)
    proxy = build_pipeline("logistic_regression", cfg).fit(xt.loc[mt], yt)
    correlations = xt.loc[mt].corrwith(yt)
    coefficients = pd.Series(proxy[-1].coef_[0], index=xt.columns)
    top = coefficients.abs().nlargest(cfg["fairness"]["proxy_max_features"]).index
    return {
        "validation_gender_auc": float(roc_auc_score(yv, proxy.predict_proba(xv.loc[mv])[:, 1])),
        "target": "M=1, F=0; XNA excluded because gender is unknown",
        "fit_split": "train",
        "evaluation_split": "validation",
        "causal_claim": False,
        "top_features": [
            {
                "feature": feature,
                "standardized_coefficient": float(coefficients[feature]),
                "train_gender_correlation": float(correlations[feature])
                if np.isfinite(correlations[feature])
                else None,
            }
            for feature in top
        ],
    }


def run_comparison(cfg: dict) -> dict:
    """Fit mitigation comparators, log real decisions and export eligible candidates."""
    processed = project_path(cfg, "processed")
    frames = {
        name: pd.read_csv(processed / f"{name}.csv", index_col="SK_ID_CURR")
        for name in ("X_train", "X_val", "y_train", "y_val", "sensitive_train", "sensitive_val")
    }
    xt, xv = frames["X_train"], frames["X_val"]
    yt, yv = frames["y_train"].TARGET, frames["y_val"].TARGET
    st, sv = frames["sensitive_train"], frames["sensitive_val"]
    for x, y, sensitive in ((xt, yt, st), (xv, yv, sv)):
        if not x.index.equals(y.index) or not x.index.equals(sensitive.index):
            raise ValueError("Mitigation features, labels and audit groups must align.")
    if len(xt.index.intersection(xv.index)):
        raise ValueError("Mitigation train/validation overlap.")
    base = pickle.loads((project_path(cfg, "models") / "baseline.pkl").read_bytes())
    if list(xt.columns) != base["columns"] or list(xv.columns) != base["columns"]:
        raise ValueError("Feature contract differs from the trained model.")
    output = project_path(cfg, "audit_reports")
    output.mkdir(parents=True, exist_ok=True)
    models = project_path(cfg, "audit_models")
    models.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    seed = cfg["data"]["random_state"]
    baseline_auc = base["metrics"]["validation_auc"]
    utility = cfg["fairness"]["mitigation"]
    policies: dict[str, Any] = {"baseline": ScoringPolicy(base["pipeline"], base["threshold"])}
    elapsed = {"baseline": 0.0}
    feasible = {"baseline": None}
    weights = reweighing_weights(yt, intersection_groups(st))
    weight_summary = {
        "min": float(weights.min()),
        "max": float(weights.max()),
        "mean": float(weights.mean()),
    }
    for power in utility["reweighing_powers"]:
        name = f"reweighing_{power:g}"
        print(f"Fitting {name}", flush=True)
        pipeline = build_pipeline("lightgbm", cfg)
        start = time.perf_counter()
        pipeline.fit(xt, yt, model__sample_weight=weights**power)
        elapsed[name] = time.perf_counter() - start
        threshold, feasible[name] = select_fair_threshold(
            yv, pipeline.predict_proba(xv)[:, 1], sv, cfg
        )
        policies[name] = ScoringPolicy(pipeline, threshold)

    print("Fitting ThresholdOptimizer (train-only calibration)", flush=True)
    fit_ids, calibration_ids = train_test_split(
        xt.index, test_size=utility["calibration_size"], stratify=yt, random_state=seed
    )
    start = time.perf_counter()
    threshold_base = build_pipeline("lightgbm", cfg).fit(xt.loc[fit_ids], yt.loc[fit_ids])
    groups = intersection_groups(st.loc[calibration_ids])
    counts = pd.crosstab(groups, yt.loc[calibration_ids]).reindex(columns=[0, 1], fill_value=0)
    known = tuple(counts.index[(counts.min(axis=1) >= 2) & ~counts.index.str.startswith("XNA|")])
    eligible = groups.isin(known)
    optimizer = ThresholdOptimizer(
        estimator=threshold_base,
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
        grid_size=1000,
    ).fit(
        xt.loc[calibration_ids[eligible]],
        yt.loc[calibration_ids[eligible]],
        sensitive_features=groups[eligible],
    )
    policies["threshold_optimizer"] = ThresholdPolicy(
        threshold_base, optimizer, known, base["threshold"], seed
    )
    elapsed["threshold_optimizer"] = time.perf_counter() - start
    feasible["threshold_optimizer"] = None

    print("Fitting ExponentiatedGradient (intersectional EqualizedOdds)", flush=True)
    start = time.perf_counter()
    reduction_cfg = {
        **cfg,
        "train": {**cfg["train"], "lightgbm": {**cfg["train"]["lightgbm"], "class_weight": None}},
    }
    reduction = ExponentiatedGradient(
        build_pipeline("lightgbm", reduction_cfg),
        EqualizedOdds(difference_bound=utility["reduction_difference_bound"]),
        eps=utility["reduction_eps"],
        max_iter=utility["reduction_max_iter"],
        sample_weight_name="model__sample_weight",
    ).fit(xt, yt, sensitive_features=intersection_groups(st))
    policies["exponentiated_gradient"] = ReductionPolicy(reduction, seed)
    elapsed["exponentiated_gradient"] = time.perf_counter() - start
    feasible["exponentiated_gradient"] = None

    rows, reports = [], {}
    requirements = [
        f"{package}=={importlib.metadata.version(package)}"
        for package in ("scikit-learn", "numpy", "pandas", "lightgbm", "mlflow")
    ]
    for name, policy in policies.items():
        scores = policy.scores(xv)
        decisions = policy.predict(xv, sv)
        fairness = audit_fairness(yv, decisions, sv, cfg["fairness"], scores)
        metrics = policy_metrics(yv, scores, decisions, cfg)
        util_ok = (
            metrics["validation_recall"] >= utility["min_recall"]
            and utility["min_acceptance_rate"]
            <= metrics["validation_acceptance_rate"]
            <= utility["max_acceptance_rate"]
        )
        loss = baseline_auc - metrics["validation_auc"]
        passes = (
            fairness["passes_thresholds"]
            and loss < cfg["fairness"]["max_auc_loss_mitigation"]
            and util_ok
        )
        row = {
            "method": name,
            **metrics,
            "training_seconds": elapsed[name],
            "auc_loss_vs_baseline": loss,
            "passes_phase5": bool(passes),
            "utility_passes": bool(util_ok),
            "global_threshold_feasible": feasible[name],
            "requires_sensitive_at_inference": isinstance(policy, ThresholdPolicy),
            "randomized_decision": not isinstance(policy, ScoringPolicy),
            "score_semantics": (
                "mixture probability of refusal, not default risk"
                if isinstance(policy, ReductionPolicy)
                else "underlying tree default ranking score, not calibrated PD"
            ),
            "threshold": policy.threshold if isinstance(policy, ScoringPolicy) else None,
        }
        for attr, values in fairness["attributes"].items():
            row[f"{attr}_disparate_impact"] = values["disparate_impact"]
            row[f"{attr}_equal_opportunity_difference"] = values["equal_opportunity_difference"]
        with mlflow.start_run(run_name=f"phase5-{name}") as run:
            mlflow.set_tags(
                {
                    "phase": "5",
                    "mitigation": name,
                    "parent_run_id": base["run_id"],
                    "gate_status": "not_evaluated",
                    "test_split": "untouched",
                    "positive_definition": "1=default/refusal; acceptance=1-prediction",
                    "training_data_sha256": base["metrics"]["training_data_sha256"],
                }
            )
            mlflow.log_params(
                {
                    "method": name,
                    "random_state": seed,
                    "utility.min_recall": utility["min_recall"],
                    "requires_sensitive_at_inference": row["requires_sensitive_at_inference"],
                }
            )
            mlflow.log_dict(cfg, "params.json")
            mlflow.log_metrics(metrics)
            mlflow.log_metric("training_seconds", elapsed[name])
            for attr, values in fairness["attributes"].items():
                for metric in (
                    "disparate_impact",
                    "demographic_parity_difference",
                    "equal_opportunity_difference",
                    "equalized_odds_difference",
                ):
                    if values[metric] is not None:
                        mlflow.log_metric(f"fairness.{attr}.{metric}", values[metric])
                for group, group_values in values["by_group"].items():
                    metric_group = group.replace("<", "under_").replace("+", "_plus")
                    for metric in ("acceptance_rate", "tpr", "fpr", "rows", "auc"):
                        if group_values[metric] is not None:
                            mlflow.log_metric(
                                f"group.{attr}.{metric_group}.{metric}", group_values[metric]
                            )
            mlflow.log_dict(fairness, "fairness_report.json")
            mlflow.log_dict(row, "mitigation_metrics.json")
            model_uri = None
            if isinstance(policy, ScoringPolicy):
                mlflow.log_metric("threshold", policy.threshold)
                info = mlflow.sklearn.log_model(
                    policy.pipeline,
                    name="model",
                    pyfunc_predict_fn="predict_proba",
                    input_example=xt.head(5),
                    pip_requirements=requirements,
                    signature=infer_signature(
                        xt.head(5), policy.pipeline.predict_proba(xt.head(5))
                    ),
                    metadata={"threshold": policy.threshold, "positive_label": 1},
                )
                model_uri = info.model_uri
                mlflow.log_artifact(str(processed / "feature_encoder.pkl"), "features")
                mlflow.log_dict(base["columns"], "feature_columns.json")
            bundle = {
                "policy": policy,
                "columns": base["columns"],
                "run_id": run.info.run_id,
                "model_uri": model_uri,
                "metrics": row,
                "fairness_report": fairness,
                "threshold": row["threshold"],
                "pipeline": getattr(policy, "pipeline", None),
            }
            (models / f"{name}.pkl").write_bytes(pickle.dumps(bundle))
            row["run_id"] = run.info.run_id
            row["model_uri"] = model_uri
            mlflow.log_artifact(str(models / f"{name}.pkl"), "policy")
        rows.append(row)
        reports[name] = fairness
        print(json.dumps(row), flush=True)
    # Group-free, deterministic tree models can receive the same SHAP/API contract.
    eligible_rows = [
        row for row in rows if row["method"].startswith("reweighing_") and row["passes_phase5"]
    ]
    selected = (
        max(eligible_rows, key=lambda row: row["validation_auc"])["method"]
        if eligible_rows
        else None
    )
    if selected:
        (models / "candidate.pkl").write_bytes((models / f"{selected}.pkl").read_bytes())
    else:
        (models / "candidate.pkl").unlink(missing_ok=True)
    proxy = proxy_analysis(xt, xv, st, sv, cfg)
    write_json(output / "proxy_report.json", proxy)
    comparison = {
        "phase": 5,
        "selected_group_free_candidate": selected,
        "results": rows,
        "weight_summary": weight_summary,
        "baseline_auc": baseline_auc,
        "test_split": "untouched",
        "promotion_status": "not_evaluated_not_promoted",
    }
    write_json(output / "mitigation_comparison.json", comparison)
    write_json(output / "fairness_report.json", reports)
    pd.DataFrame(rows).to_csv(output / "mitigation_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for row in rows:
        gap = max(
            row[f"{attr}_equal_opportunity_difference"]
            for attr in cfg["fairness"]["sensitive_features"]
        )
        ax.scatter(
            gap, row["validation_auc"], color="#267a92" if row["passes_phase5"] else "#ba553c"
        )
        ax.annotate(
            row["method"],
            (gap, row["validation_auc"]),
            xytext={
                "reweighing_1.5": (-5, 9),
                "reweighing_1.75": (-5, -15),
                "reweighing_2": (-5, -15),
                "reweighing_1": (5, -12),
            }.get(row["method"], (5, 4)),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(cfg["fairness"]["max_equal_opp_diff"], color="gray", linestyle="--")
    ax.axhline(
        baseline_auc - cfg["fairness"]["max_auc_loss_mitigation"], color="gray", linestyle="--"
    )
    ax.set(
        xlabel="Maximum default TPR gap (gender / age)",
        ylabel="Validation ROC-AUC",
        title="Performance versus measured fairness (red: at least one rule fails)",
    )
    ax.margins(x=0.20, y=0.08)
    fig.savefig(output / "performance_fairness.png", dpi=150)
    plt.close(fig)
    client = mlflow.MlflowClient()
    client.log_dict(rows[0]["run_id"], proxy, "proxy_report.json")
    client.log_artifact(rows[0]["run_id"], str(output / "performance_fairness.png"))
    return comparison


def main() -> None:
    """Run phase-five mitigation comparisons using creditS."""
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMClassifier.*",
        category=UserWarning,
    )
    print(json.dumps(run_comparison(load_config(parse_params())), indent=2))


if __name__ == "__main__":
    main()
