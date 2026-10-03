# Phase 5 — SHAP, équité et mitigations

J'ai analysé les explications SHAP, les proxys et les écarts d'équité, puis comparé les méthodes de mitigation.

Ce compte rendu décrit les vérifications de cette phase au moment de son exécution. Le [bilan définitif](project-completion.md) présente le champion final et les dernières validations.

Environnement utilisé : `creditS`, Python 3.11.9, SHAP 0.49.1, Fairlearn 0.13.0,
MLflow 3.10.1. Signatures inspectées sur les installations réelles, puis
ThresholdOptimizer et ExponentiatedGradient exécutés sur les données du projet.
Les schémas et features des phases précédentes restent identiques.

## Exécution

Deux stages DVC supplémentaires : `fairness` et `explain`.
`make fairness` démarre MLflow puis reproduit leurs dépendances.
Les sorties sont `artifacts/fairness_models/`, `reports/fairness/` et
`reports/shap/`, suivies DVC. Le dossier artifacts reste ignoré par Git et Docker.
Les sorties d'entraînement de phase 4 restent séparées : aucun remplacement
de baseline, alias champion ou promotion n'a lieu pendant la phase 5.

Les poids `P(g)P(y)/P(g,y)` sont estimés sur le train, sur les intersections
CODE_GENDER/AGE_GROUP. Puissances comparées : 0.5, 1, 1.5, 1.75, 2.
Un seuil global est choisi sur validation en minimisant le coût métier parmi
les points respectant équité et utilité. À défaut, la solution reste marquée
non conforme ; les règles ne sont jamais relâchées automatiquement.
Cette sélection sur la validation exige encore une évaluation sur le test.

ThresholdOptimizer emploie une calibration séparée de 20% du train,
EqualizedOdds et balanced_accuracy_score. Son AUC est celle du score de base ;
ses métriques de confusion et d'équité utilisent les décisions randomisées.
ExponentiatedGradient utilise EqualizedOdds, eps=0.01, difference_bound=0.02,
max_iter=12 et la mixture de prédictions binaires pondérées pour son score.
Son faible rappel sur cette exécution entraîne un échec d'utilité ; aucun
succès d'équité ne masque une perte de performance importante.

## Résultats mesurés

| Méthode | AUC | DI genre | Écart TPR genre | DI âge | Écart TPR âge | Validation phase 5 |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 0.7487 | 0.8901 | 0.0722 | 0.5996 | 0.1974 | échec |
| Reweighing 0.5 | 0.7427 | 0.9370 | 0.0812 | 0.7178 | 0.1700 | échec |
| Reweighing 1 | 0.7374 | 0.9676 | 0.0340 | 0.7602 | 0.1690 | échec |
| Reweighing 1.5 | 0.7311 | 0.9871 | 0.0150 | 0.8142 | 0.0921 | réussite |
| Reweighing 1.75 | 0.7296 | 0.9745 | 0.0110 | 0.7976 | 0.0961 | échec DI âge |
| Reweighing 2 | 0.7246 | 0.9699 | 0.0086 | 0.8361 | 0.0816 | échec perte AUC |
| ThresholdOptimizer | 0.7417 | 0.9535 | 0.0169 | 0.9460 | 0.1166 | échec écart âge |
| ExponentiatedGradient | 0.5772 | 0.9987 | 0.0160 | 0.9678 | 0.1053 | échec utilité/AUC/âge |

Le candidat déterministe sans attribut sensible à l'inférence est Reweighing 1.5,
seuil 0.55, perte AUC 0.01752. Le minimum absolu AUC 0.74 du quality gate n'est
pas satisfait : cette preuve répond à la validation de phase 5 uniquement.
La Model Card expose ce résultat et les limites des seuils par groupe.

Les métriques détaillées, par groupe et par méthode, sont loggées dans MLflow.
Les libellés `<30` et `60+` deviennent `under_30` et `60_plus` dans les noms
de métriques MLflow ; les JSON conservent les noms d'origine.
Tous les groupes sont affichés, et les valeurs non estimables sont null,
pas zéro. XNA reste hors comparaison, avec une couverture explicite 4919/4920.
F/M et les quatre âges doivent avoir au moins 100 clients et 20 exemples de
chaque classe. Un groupe requis manquant fait échouer l'audit.

## Explications et proxys

SHAP explique le modèle après son imputation ajustée sur le train. Contrat
strict des colonnes et exclusion des attributs sensibles. Summary, bar et
waterfalls accepté/refusé/limite pour baseline et candidat : dix PNG au total,
plus deux rapports JSON. Erreur de reconstruction de la probabilité < 1e-6.
`CreditExplainer.top_reasons(x, k=3)` et `top_reasons(x, k=3)` renvoient des
libellés français, la direction et la contribution brute, sans causalité.
Les exemples sont anonymisés : aucun SK_ID_CURR n'est loggé dans les raisons.

L'étude des proxys utilise une régression logistique ajustée sur le train pour
prédire M/F. AUC validation 0.8734 ; coefficients standardisés et corrélations
train des 20 features les plus associées sont archivés dans proxy_report.json.
Le notebook 03_fairness_analysis contient l'analyse et les figures exécutées.

## Vérification

```powershell
.\creditS\Scripts\Activate.ps1
make fairness
make test
python -m scripts.verify_fairness
python -m scripts.verify_data_pipeline
python -m ruff check .
python -m ruff format --check .
python -m dvc status
python -m dvc status --cloud
```

48 tests passent, dont les métriques d'acceptation/refus, les supports insuffisants,
les poids estimés sur le train, l'absence de solution dégénérée, l'additivité
SHAP sur des valeurs imputées et la cohérence du contrat de features.
Le vérificateur réexécute les politiques persistées sur validation, retrouve
leurs audits, contrôle les runs FINISHED, les rechargements MLflow et les
six artefacts SHAP par modèle expliqué. Il exige une mitigation group-free
avec perte AUC strictement inférieure à 0.02 et utilité minimale respectée.
