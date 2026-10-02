# Preuves — phase 11

Validation exécutée sur Windows dans `creditS` Python 3.11.9, avec Docker Linux,
MLflow 3.10.1, Airflow 2.11.0, Evidently 0.7.20, Fairlearn 0.13.0 et SHAP 0.49.1.

- `make test` : **95 tests réussis**, 25 avertissements de bibliothèques connus.
  Deux tests ajoutés couvrent les cohortes disjointes, l'absence des attributs
  sensibles dans les entrées et les performances/équités réellement calculées
  des trois candidats. Les tests data, features, API, fairness, gate et monitoring
  restent verts, sans téléchargement Kaggle.
- Après déplacement des paramètres du scénario dans `params.yaml`, les cinq
  tests affectés (`test_demo`, `test_setup`) passent à nouveau.
- Ruff et vérification du formatage réussis ; `git diff --check` réussi.
- `make up` réussi : les cinq services du projet restent healthy. Airflow accède
  au nouvel espace privé `runtime/demo/` ; aucune installation locale d'Airflow.
- `make demo` a réussi deux fois : `demo__b0c4f9bebe3f`, puis
  **`demo__96229a11ef3b`** avec les paramètres centralisés. Chaque invocation
  utilise un nouveau modèle Registry et une nouvelle API Docker isolée.
- Données synthétiques suivies par DVC dans un dépôt privé sans SCM :
  initialisation, `dvc add fixture`, remote local distinct et **13 fichiers poussés**.
- DAG parent `credit_demo` réussi : provenance, entraînement v1, déploiement v1,
  monitoring Evidently, `trigger_retraining=success`, `no_alert=skipped`.
- Alertes mesurées : `data_drift`, `prediction_drift`,
  `data_drift.AMT_INCOME_TOTAL`. Performance non évaluée à ce stade sans labels.
- DAG enfant **`retraining__demo__96229a11ef3b`** réussi : v2 rejeté pour
  `CODE_GENDER.disparate_impact` et `CODE_GENDER.equal_opportunity_difference`,
  v3 repondéré accepté, puis `reload_api=success`.
- AUC et DI genre : v1 **0,7548 / 1,0000**, v2 **0,9353 / 0,6509**,
  v3 **0,9583 / 1,0000**. Écart d'égalité des chances genre de v2 : **0,1950**.
  Les gates emploient les seuils originaux, les supports réels et les six
  artefacts SHAP téléchargés et vérifiés dans MLflow.
- Modèle `credit-demo-96229a11ef3b` : versions 1/2/3 enregistrées,
  statuts accepted/rejected/accepted, aliases `champion=3`, `previous=1`.
- API Docker : `/predict` et `/explain` répondent avec la version 1 puis 3,
  scores cohérents et trois raisons SHAP. Les événements sont pseudonymisés.
  Le conteneur temporaire est supprimé après vérification.
- Les alias du modèle **Home Credit `credit-scoring` restent vides**,
  vérifiés avant/après. Aucune de ses cinq versions rejetées n'est promue.
- Commandes de build du workflow exécutées après les tests :
  `docker build -f api/Dockerfile -t credit-scoring-api:ci .` et
  `docker build -f airflow/Dockerfile -t credit-scoring-airflow:ci .` réussies.
- Workflow GitHub Actions défini pour push/PR/manual : venv `.ci-venv`, mêmes
  dépendances racine, Ruff → pytest → deux builds Docker, permissions de lecture,
  actions épinglées par commit. Aucun remote Git n'est configuré ; **aucun run
  GitHub Actions distant n'a été exécuté**. Les preuves locales ne remplacent
  pas une exécution du runner Ubuntu.

Rapport complet local : `runtime/demo/96229a11ef3b/validation.json`.
Limites du scénario, métriques et fonctionnement : `docs/demo.md`.
La démonstration synthétique mesure et contrôle le biais selon les métriques
définies ; elle ne fournit aucune qualification de crédit réelle.

```powershell
.\creditS\Scripts\Activate.ps1
make up
make test
make demo
make lint
python -m ruff format --check .
```

La documentation finale et la Model Card seront complétées en phase 12 après
validation de cette phase.
