# Phase 4 — Expérimentation MLflow

J'ai comparé les familles de modèles sur les mêmes splits et enregistré les expériences dans MLflow.

Ce compte rendu décrit les vérifications de cette phase au moment de son exécution. Le [bilan définitif](project-completion.md) présente le champion final et les dernières validations.

Validation exécutée le 2 octobre 2026, Windows, Python 3.11.9 dans `creditS`.
MLflow 3.10.1, scikit-learn 1.7.2, LightGBM 4.6.0, XGBoost 3.0.5.
La signature de `mlflow.sklearn.log_model` et les options du serveur ont été
vérifiées sur les versions installées. Le modèle est loggé avec `name="model"`
et une sortie pyfunc `predict_proba` de deux colonnes, dans l'ordre `[0, 1]`.

## Résultats mesurés

Même train (14 760 clients), validation (4 920), 261 features, sample_frac=0.1.
Les transformations sont ajustées sur le train, puis appliquées à la validation.
L'entraînement n'ouvre ni X_test/y_test ni les labels de production.

| Modèle | ROC-AUC validation | PR-AUC (average precision) | KS | Seuil de refus |
|---|---:|---:|---:|---:|
| Régression logistique | 0.7210 | 0.1984 | 0.3647 | 0.09 |
| Random Forest | 0.7240 | 0.2339 | 0.3294 | 0.09 |
| XGBoost | 0.7372 | 0.2284 | 0.3646 | 0.07 |
| LightGBM | 0.7487 | 0.2427 | 0.3770 | 0.49 |

Le seuil minimise `10 × FN + 1 × FP` sur validation, dans la grille configurée.
Positif = défaut prédit = refus ; acceptation = 1 - prédiction.
Le candidat est choisi par AUC validation, indépendamment du seuil.
Le coût obtenu pour LightGBM est 2 645, contre 2 697 pour XGBoost.
Les durées réelles, recall, precision, Gini, matrice de confusion et taux
d'acceptation sont enregistrés dans chaque run et dans la comparaison DVC.

LightGBM initial (500 arbres, 31 feuilles) : AUC 0.7263.
Six essais prédéfinis, validation uniquement, min_child_samples=100 et reg_lambda=5 :

| Feuilles | Arbres | Pondération | AUC validation |
|---:|---:|---|---:|
| 15 | 150 | aucune | 0.7450 |
| 15 | 300 | aucune | 0.7414 |
| 7 | 300 | aucune | 0.7479 |
| 15 | 150 | balanced | 0.7487 |
| 15 | 300 | balanced | 0.7443 |
| 7 | 300 | balanced | 0.7456 |

Les essais sont conservés dans MLflow ; les paramètres retenus sont dans
params.yaml. `python -m scripts.tune_lightgbm` reproduit cette recherche bornée.
La sélection de paramètres, de famille et de seuil sur la même validation peut
optimiser excessivement ces résultats : le test indépendant sera nécessaire
pour l'évaluation finale. Avec les pondérations de classes, le score n'est pas
une probabilité de défaut calibrée ; aucune calibration n'est revendiquée.

## Persistance et provenance

Serveur local : http://localhost:5000, SQLite `mlruns/tracking.db`, artefacts
dans `mlartifacts/`, processus sans fenêtre, écoute 127.0.0.1, un worker.
`make mlflow` le démarre et `make train` le démarre avant DVC. Les dossiers du
serveur restent ignorés par Git ; les modèles et la comparaison sont suivis DVC.
L'UI présente aussi les essais et les exécutions de diagnostic antérieures ;
les quatre run_id de `reports/model_comparison.json` identifient la comparaison finale.

Chaque run final contient un modèle, sa signature, les paramètres, le seuil,
les colonnes ordonnées, l'encodeur train et son vocabulaire, les métriques,
le hash DVC brut et les SHA-256 des entrées train/validation.
Le tag `dvc_lock_sha256` correspond exactement à l'artefact `provenance/dvc.lock`
capturé avant entraînement. Ce snapshot ne peut pas être le lock final qui
contient les hashes des sorties de ce même stage. Il est transféré en octets,
pour préserver les fins de ligne sous Windows.
Hash combiné des entrées :
`f25c1a50ec60b966500509aae15df822853865ffc2b9e501612846204965458d`.

`models/baseline.pkl` reste le chemin de compatibilité pour la simulation ;
il contient maintenant le meilleur candidat LightGBM. Aucun alias champion
n'est créé : équité, SHAP et quality gate sont les phases suivantes.
Les huit lots restent normaux jusqu'à la phase 10.
Les noms catégoriels contenant des caractères JSON imposent des matrices numpy
dans LightGBM ; les noms originaux ordonnés restent conservés dans les artefacts.
L'avertissement sklearn sur les noms LightGBM n'altère pas ce contrat.

## Vérifications

```powershell
.\creditS\Scripts\Activate.ps1
make train
make test
python -m scripts.verify_experiments
python -m scripts.verify_data_pipeline
python -m ruff check .
python -m ruff format --check .
python -m dvc status
python -m dvc status --cloud
```

37 tests couvrent notamment les métriques connues, le coût métier, les quatre
pipelines et une sauvegarde/relecture réelle MLflow dans SQLite temporaire.
Le vérificateur des expériences contrôle les runs FINISHED, les hashes du lock,
les encodeurs présents et l'identité des probabilités entre pickle local,
flavor sklearn MLflow et pyfunc MLflow, pour les quatre modèles réels.
Le notebook 02_baseline a trois cellules exécutées, tableau et figure embarquée.
Les temps d'entraînement, run_id et URI de modèles varient entre entraînements ;
une reproduction ne promet donc pas des métadonnées identiques en octets.
Un `dvc repro` sans changement restaure ou réutilise les sorties sans créer de
nouveaux runs. Pour refaire les expériences, démarrer MLflow puis exécuter
`python -m dvc repro --force`.
