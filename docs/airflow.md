# Orchestration Airflow (phase 7)

Airflow 2.11.0 tourne exclusivement dans Docker, avec LocalExecutor et PostgreSQL.
Les dépendances ML tournent dans le `creditS` propre à l'image, séparées de celles
d'Airflow. Le venv Windows n'est jamais copié. XGBoost utilise sa distribution CPU.

`airflow/requirements-worker-lock.txt` épingle aussi les dépendances transitives
Linux. Un cache optionnel accélère les connexions lentes : après activation de
`creditS`, exécuter `python scripts/cache_worker_wheels.py` avant `make up`.
Les wheels restent dans `.docker-wheels/`, ignorées par Git ; BuildKit les monte
uniquement pendant l'installation. Un clone sans cache télécharge les mêmes
versions épinglées lors du build.

```powershell
.\creditS\Scripts\Activate.ps1
make up
python scripts/verify_airflow.py
python scripts/verify_airflow.py --trigger training --wait
python scripts/verify_airflow.py --trigger monitoring --wait
```

Interfaces : [Airflow](http://localhost:8081) et [MLflow Docker](http://localhost:5001).
L'utilisateur Airflow est `credit_admin` ; son mot de passe généré se trouve dans
`.env`, ignoré par Git. Les ports sont configurables dans ce fichier.
`make down` arrête uniquement les services de ce dépôt, en conservant les volumes.

Le MLflow Docker reçoit, au premier démarrage seulement, une copie cohérente SQLite
du serveur local et de ses artefacts. Les deux serveurs évoluent ensuite séparément.
Le MLflow local reste sur 5000. La base SQLite Windows n'est pas partagée en écriture
avec Linux. Le Registry Docker ne contient initialement aucun champion accepté.

## Entraînement

`ingest → validate_data → build_features → train_models → evaluate_performance →
evaluate_fairness → compute_shap → quality_gate → register_and_promote | reject_model
→ notification`.

Les tâches appellent les services métier dans `src/`, par sous-processus sans shell.
Seuls chemins, run IDs et petites métriques passent en XCom. Les sorties sont isolées
dans `runtime/airflow/runs/`, avec les paramètres, hashes des données, rapports et
résultats du gate. Les données sources, modèles et fichiers DVC locaux sont montés
en lecture seule ; les nouvelles données calculées sont identifiées par leurs hashes
dans MLflow. Le lock DVC archivé correspond au snapshot des données sources locales,
pas à un nouveau `dvc repro` dans le conteneur.

Un pool de capacité 1 sérialise les calculs et promotions des deux DAGs. Chaque DAG
autorise une exécution active, un retry et des délais maximum. Une tâche coûteuse
terminée possède un checkpoint ; une tâche échouée est relancée. Le gate est recalculé
avant l'enregistrement. La notification est un résumé local JSON et un log Airflow.

Dans l'UI, déclencher `training_pipeline` avec `{"min_auc": 1.0}` pour tester le rejet.
Cet override peut seulement renforcer le gate. Le rejet métier termine le DAG avec
succès, taggue la version `rejected` et laisse les alias inchangés. Une panne technique
échoue la tâche. La promotion acceptée utilise la même logique testée en phase 6.

## Surveillance

`load_production_batch → drift_report → fairness_monitor → evaluate_alert_rules →
alert_branch → trigger_training | no_alert`.

Les rapports HTML et JSON utilisent réellement Evidently 0.7.20 (API `Report` et
`Dataset`). Les colonnes sont celles du modèle ; le PSI des features variables et
du score est comparé à la référence de validation figée. Les constantes de référence
sont comptées mais exclues du test PSI. Fairlearn mesure l'acceptation par genre et
âge. AUC et audit des taux de défaut ne sont calculés qu'après disponibilité des
labels selon le manifeste et `observed_batch`.

Exemple de configuration : `{"batch_id": "01", "observed_batch": 3}`. Sans clock
explicite, le lot 01 est observé au temps 1 et ses labels ne sont pas lus. Pour tester
le déclenchement sur une différence de score, utiliser `{"batch_id": "01",
"max_score_psi": 0.0}`. Le `TriggerDagRunOperator` crée une exécution d'entraînement
identifiée par le run de monitoring ; son retry ne crée pas une seconde exécution.

Actuellement, les lots sont les lots normaux de la phase 3 et leurs prédictions
proviennent d'un candidat baseline rejeté, explicitement identifié comme tel.
Cette surveillance est diagnostique, sans service champion déployé. Les scénarios
de drift 04–08, les rapports de performance Evidently complets et la validation de
l'absence d'alertes sur les lots normaux sont prévus en phase 10.
