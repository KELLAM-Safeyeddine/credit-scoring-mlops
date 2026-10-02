"""Tree SHAP on the exact fitted preprocessing and ordered model columns."""

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import shap

from src.config import load_config, parse_params, project_path, write_json
from src.features.contracts import EXCLUDED

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FEATURE_LABELS = {
    "AMT_CREDIT": "Montant du crédit demandé",
    "AMT_INCOME_TOTAL": "Revenu annuel déclaré",
    "AMT_ANNUITY": "Montant de l'annuité",
    "AMT_GOODS_PRICE": "Prix du bien financé",
    "DAYS_EMPLOYED": "Ancienneté professionnelle (jours négatifs)",
    "EMPLOYED_ANOMALY": "Ancienneté professionnelle indisponible",
    "EXT_SOURCE_1": "Score externe 1",
    "EXT_SOURCE_2": "Score externe 2",
    "EXT_SOURCE_3": "Score externe 3",
    "EXT_SOURCE_MEAN": "Moyenne des scores externes disponibles",
    "CREDIT_INCOME_RATIO": "Crédit demandé rapporté au revenu",
    "ANNUITY_INCOME_RATIO": "Annuité rapportée au revenu",
    "CREDIT_TERM": "Crédit rapporté à l'annuité",
    "BUREAU_COUNT": "Nombre de crédits dans le bureau de crédit",
    "BUREAU_ACTIVE_RATE": "Part des crédits encore actifs",
    "BUREAU_CREDIT_SUM": "Montant total des crédits historiques",
    "BUREAU_DEBT_SUM": "Dette historique restante",
    "BUREAU_OVERDUE_MAX": "Plus grand impayé historique",
    "PREV_COUNT": "Nombre de demandes de crédit antérieures",
    "PREV_REFUSAL_RATE": "Part des demandes antérieures refusées",
    "PREV_APPLICATION_MEAN": "Montant moyen demandé auparavant",
    "PREV_CREDIT_MEAN": "Montant moyen accordé auparavant",
    "INSTALLMENTS_COUNT": "Nombre d'échéances historiques",
    "INSTALLMENTS_PAYMENT_COUNT": "Nombre de versements historiques",
    "INSTALLMENTS_DELAY_MEAN": "Retard moyen de paiement en jours",
    "INSTALLMENTS_LATE_RATE": "Part des échéances payées en retard",
    "INSTALLMENTS_PAYMENT_GAP_MEAN": "Écart moyen entre montant dû et payé",
    "NAME_EDUCATION_TYPE": "Niveau de formation",
    "NAME_INCOME_TYPE": "Type de revenu",
    "NAME_FAMILY_STATUS": "Situation familiale",
    "NAME_HOUSING_TYPE": "Situation de logement",
    "OCCUPATION_TYPE": "Profession déclarée",
    "ORGANIZATION_TYPE": "Type d'employeur",
    "CNT_CHILDREN": "Nombre d'enfants déclaré",
    "CNT_FAM_MEMBERS": "Nombre de personnes du foyer",
    "DAYS_ID_PUBLISH": "Ancienneté du document d'identité (jours négatifs)",
    "DAYS_REGISTRATION": "Ancienneté de l'enregistrement (jours négatifs)",
    "DAYS_LAST_PHONE_CHANGE": "Ancienneté du changement de téléphone (jours négatifs)",
    "FLAG_OWN_CAR": "Possession d'un véhicule",
    "FLAG_OWN_REALTY": "Possession d'un bien immobilier",
    "REGION_RATING_CLIENT": "Notation de la région déclarée",
    "REGION_RATING_CLIENT_W_CITY": "Notation de la région et de la ville",
    "REG_CITY_NOT_WORK_CITY": "Ville de résidence différente du lieu de travail",
    "REG_CITY_NOT_LIVE_CITY": "Ville d'enregistrement différente du lieu de vie",
    "REGION_POPULATION_RELATIVE": "Part relative de population dans la région",
}


def feature_label(feature: str) -> str:
    """Translate known variables/categories; retain an explicit fallback otherwise."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    for source in sorted(FEATURE_LABELS, key=len, reverse=True):
        if feature.startswith(source + "_"):
            category = feature[len(source) + 1 :]
            translated = {"Y": "oui", "N": "non", "__MISSING__": "information manquante"}
            return f"{FEATURE_LABELS[source]} : {translated.get(category, category)}"
    if feature.startswith("FLAG_DOCUMENT_"):
        return f"Présence du document justificatif {feature.removeprefix('FLAG_DOCUMENT_')}"
    return "Indicateur " + feature.replace("_", " ").lower()


class CreditExplainer:
    """Explain the tree's raw default score, not causal effects or fairness."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        self.pipeline = bundle["pipeline"]
        self.columns = bundle["columns"]
        if EXCLUDED.intersection(self.columns):
            raise ValueError("Audit attributes cannot enter explanations as model features.")
        self.explainer = shap.TreeExplainer(self.pipeline[-1], model_output="raw")

    def explain(self, x: pd.DataFrame) -> shap.Explanation:
        """Return class-one SHAP values for the exact transformed observations."""
        if list(x.columns) != self.columns:
            raise ValueError("SHAP input must match the ordered training columns.")
        transformed = self.pipeline[:-1].transform(x)
        result = self.explainer(transformed, check_additivity=True)
        values, base = np.asarray(result.values), np.asarray(result.base_values)
        if values.ndim == 3:
            values, base = values[:, :, 1], base[:, 1]
        if base.ndim == 0:
            base = np.full(len(x), float(base))
        return shap.Explanation(
            values=values, base_values=base, data=transformed, feature_names=self.columns
        )

    def top_reasons(self, x: pd.DataFrame, k: int = 3) -> list[dict[str, Any]]:
        """Describe the largest absolute raw-score contributions for one client."""
        if len(x) != 1 or not 1 <= k <= len(self.columns):
            raise ValueError("Provide one observation and a valid positive reason count.")
        explanation = self.explain(x)
        values = explanation.values[0]
        order = np.argsort(-np.abs(values), kind="stable")[:k]
        return [
            {
                "feature": self.columns[index],
                "label": feature_label(self.columns[index]),
                "direction": "augmente le score de défaut"
                if values[index] > 0
                else "diminue le score de défaut"
                if values[index] < 0
                else "effet nul",
                "contribution_raw_score": float(values[index]),
                "transformed_value": float(explanation.data[0, index]),
                "interpretation": "association du modèle, pas une cause",
            }
            for index in order
        ]


def top_reasons(x: pd.DataFrame, k: int = 3, *, bundle: dict | None = None) -> list[dict]:
    """Convenience helper; services should reuse a CreditExplainer instance."""
    if bundle is None:
        cfg = load_config()
        candidate = project_path(cfg, "audit_models") / "candidate.pkl"
        path = candidate if candidate.exists() else project_path(cfg, "models") / "baseline.pkl"
        bundle = pickle.loads(path.read_bytes())
    return CreditExplainer(bundle).top_reasons(x, k)


def create_shap_artifacts(
    bundle: dict, x: pd.DataFrame, output: Path, sample_size: int, seed: int
) -> dict:
    """Export summary, importance and three anonymized operating-point examples."""
    output.mkdir(parents=True, exist_ok=True)
    explainer = CreditExplainer(bundle)
    sample = x.sample(n=min(sample_size, len(x)), random_state=seed)
    explanation = explainer.explain(sample)
    # Plots use readable labels; the machine-readable contract retains original names.
    explanation.feature_names = [feature_label(column) for column in bundle["columns"]]
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.gcf().set_size_inches(11, 7)
    plt.tight_layout()
    plt.savefig(output / "summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    shap.plots.bar(explanation, max_display=15, show=False)
    plt.gcf().set_size_inches(11, 7)
    plt.tight_layout()
    plt.savefig(output / "bar.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    scores = bundle["pipeline"].predict_proba(x)[:, 1]
    threshold = bundle["threshold"]
    decisions = scores >= threshold
    if not decisions.any() or decisions.all():
        raise ValueError("Cannot illustrate both accepted and refused observations.")
    near = int(np.argmin(np.abs(scores - threshold)))
    accepted = np.where(~decisions)[0]
    refused = np.where(decisions)[0]
    choices = {
        "accepted": int(accepted[np.argmin(scores[accepted])]),
        "refused": int(refused[np.argmax(scores[refused])]),
        "borderline": near,
    }
    examples = {}
    for label, position in choices.items():
        one = x.iloc[[position]]
        local = explainer.explain(one)[0]
        local.feature_names = [feature_label(column) for column in bundle["columns"]]
        shap.plots.waterfall(local, max_display=10, show=False)
        plt.gcf().set_size_inches(11, 7)
        plt.title(f"{label} | score={scores[position]:.3f}, refusal threshold={threshold:.2f}")
        plt.savefig(output / f"waterfall_{label}.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        examples[label] = {
            "score": float(scores[position]),
            "predicted_default": int(decisions[position]),
            "reasons": explainer.top_reasons(one),
            "raw_score_reconstructed": float(local.base_values + local.values.sum()),
        }
    reconstructed = explanation.base_values + explanation.values.sum(axis=1)
    predicted = bundle["pipeline"].predict_proba(sample)[:, 1]
    if bundle["pipeline"][-1].__class__.__name__ in {"LGBMClassifier", "XGBClassifier"}:
        recovered = 1 / (1 + np.exp(-reconstructed))
        units = "raw log-odds of predicted default; positive increases default score"
    else:
        recovered, units = reconstructed, "class-one tree output"
    np.testing.assert_allclose(recovered, predicted, atol=1e-6, rtol=1e-6)
    report = {
        "shap_version": shap.__version__,
        "model_run_id": bundle["run_id"],
        "sample_rows": len(sample),
        "feature_count": len(bundle["columns"]),
        "units": units,
        "max_probability_reconstruction_error": float(np.max(abs(recovered - predicted))),
        "examples": examples,
        "feature_labels": {column: feature_label(column) for column in bundle["columns"]},
        "limitations": (
            "No causal claim; explanations do not demonstrate fairness; "
            "imputed values are explained."
        ),
    }
    write_json(output / "shap_report.json", report)
    return report


def main() -> None:
    """Explain baseline and the selected group-free mitigation, log to MLflow."""
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(parse_params())
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    x = pd.read_csv(project_path(cfg, "processed") / "X_val.csv", index_col="SK_ID_CURR")
    models = project_path(cfg, "audit_models")
    names = ["baseline"] + (["candidate"] if (models / "candidate.pkl").exists() else [])
    client = mlflow.MlflowClient()
    for name in names:
        bundle = pickle.loads((models / f"{name}.pkl").read_bytes())
        output = project_path(cfg, "shap_reports") / name
        report = create_shap_artifacts(
            bundle, x, output, cfg["fairness"]["shap_sample_size"], cfg["data"]["random_state"]
        )
        for path in sorted(output.iterdir()):
            client.log_artifact(bundle["run_id"], str(path), "shap")
        client.set_tag(bundle["run_id"], "shap_artifacts_present", "true")
        print(
            json.dumps(
                {
                    "model": name,
                    "run_id": bundle["run_id"],
                    "reconstruction_error": report["max_probability_reconstruction_error"],
                }
            )
        )


if __name__ == "__main__":
    main()
