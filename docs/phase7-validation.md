# Validation exécutée — phase 7

Date : 2 octobre 2026. Hôte Windows / PowerShell, Python local 3.11.9 dans
`creditS`, Docker Linux et Docker Compose. Airflow n'est pas installé dans le venv
local. Le worker Docker affiche `sys.prefix = /opt/creditS`.

## Environnement et services

Airflow **2.11.0** ; MLflow **3.10.1** ; Evidently **0.7.20** ; Fairlearn **0.13.0** ;
SHAP **0.49.1** ; LightGBM **4.6.0** ; XGBoost CPU **3.0.5** ; PostgreSQL **14.17**.
Le verrou Linux couvre 135 distributions ; `pip check` dans l'image : aucun conflit.
`libgomp1` est installé dans l'image pour les arbres LightGBM / XGBoost.

`make up` a été exécuté, puis réexécuté après les tests d'intégration. Il attend les
healthchecks : postgres, mlflow, airflow-webserver et airflow-scheduler sont sains.
L'initialisation répétée de la base et de l'utilisateur réussit sans effacer les runs.
Les conteneurs de l'autre projet utilisant le port 8080 ont été préservés.

Interfaces : [Airflow](http://localhost:8081) et [MLflow Docker](http://localhost:5001).
Le serveur local reste sur 5000. Le Docker Registry est une copie initiale indépendante,
avec ses propres nouveaux runs. Les secrets sont dans `.env`, ignoré par Git et Docker.

## Exécutions réelles par le scheduler

| Exécution | Résultat vérifié |
|---|---|
| `training_pipeline / phase7__20261002T210008681919` | Succès ; toutes les étapes métier exécutées ; `reject_model` réussi, `register_and_promote` ignoré, notification réussie |
| `monitoring_pipeline / phase7__20261002T211612471469` | Succès ; rapport Evidently, audit Fairlearn, règles d'alerte et `TriggerDagRunOperator` exécutés ; `no_alert` ignoré |
| `training_pipeline / monitoring__phase7__20261002T211612471469` | Déclenché automatiquement ; succès complet au premier passage de chaque tâche ; candidat rejeté, notification réussie |

Le premier entraînement utilise `min_auc = 1.0` pour vérifier la branche stricte.
Le second utilise le gate normal `min_auc = 0.74`, sans affaiblissement des règles.
Les versions **3** et **4** du Registry Docker sont rejetées pour `minimum_auc` :
AUC du candidat reweighing 1.5 = **0.7311351831**. Le baseline retrouve **0.7486515882**.
Les artefacts SHAP sont vérifiés avant le gate. Les quatre versions du Registry Docker
sont rejetées ; la liste d'alias reste vide. Aucune promotion réelle acceptée n'est
revendiquée : les cas d'acceptation restent ceux du Registry isolé de la phase 6.

L'intégration a révélé deux erreurs corrigées : partitions temporaires créées sous
les sources montées en lecture seule, puis bibliothèque système OpenMP manquante.
Le premier run a été repris après correction depuis les tâches échouées.
Le second run automatique confirme le fonctionnement complet sans reprise manuelle.
Les temporaires sont confinés au répertoire autorisé et nettoyés ; les nouvelles
exécutions Linux utilisent un scratch dans le conteneur pour éviter les écritures
de partitions à travers un bind mount Windows.

## Données, surveillance et limites

Les sources DVC, ZIP et modèles locaux sont en lecture seule dans Airflow. Le contrôle
`dvc status data/raw.dvc` confirme les données sources à jour. Les fichiers X/y de
train et validation recalculés sont identiques octet par octet aux fichiers locaux.
La liste ordonnée des 261 colonnes est identique ; son JSON a des fins de ligne
différentes selon l'OS, donc un hash de fichier propre à chaque format. Les hashes
réels des entrées et le snapshot binaire du lock DVC sont loggés dans MLflow.

Lot 01 : 252 features variables testées, 9 constantes de référence signalées ;
part de features en drift **0.0** ; PSI du score **0.0161324875**. Le seuil PSI
volontairement fixé à **0.0** déclenche `prediction_drift`. L'audit constate aussi
`fairness.AGE_GROUP` sur le baseline rejeté. Le résultat porte explicitement le
rôle « frozen baseline candidate; not deployed champion ».

Au temps d'observation 1, les labels du lot 01 attendus au temps 3 ne sont pas lus :
AUC absente et audit des issues différé. Un test vérifie aussi leur lecture après
libération. Les scénarios 04–08, les rapports de performance Evidently complets et
l'absence d'alertes sur les lots normaux d'un champion sont réservés à la phase 10.

Les preuves détaillées non versionnées se trouvent dans `runtime/airflow/validation/`
(`training.json`, `monitoring.json`), les logs Airflow et les dossiers des runs :
`cc3a1de86549a201165422e9`, `a2bed1addd158445b3e722d2`,
`bc8d97dcd1638bce039efdd5`. Les rapports, paramètres, décisions et logs worker y sont
conservés ; les artefacts de modèle, d'équité et SHAP sont dans MLflow.

## Vérifications reproductibles

```powershell
.\creditS\Scripts\Activate.ps1
make up
make test
python -m ruff check .
python -m ruff format --check .
python scripts/verify_airflow.py
```

Résultats : **63 tests réussis**, ruff sans erreur, format vérifié, deux DAGs chargés
sans erreur d'import. Les avertissements concernent les dépréciations des bibliothèques
et les noms de features LightGBM connus depuis la phase 4.

Pour rejouer les intégrations (nouveaux entraînements et nouvelles versions rejetées) :

```powershell
.\creditS\Scripts\Activate.ps1
python scripts/verify_airflow.py --trigger training --wait
python scripts/verify_airflow.py --trigger monitoring --wait
```

Un test supplémentaire protège le Registry : si le gate change après le branchement,
l'enregistrement et toute modification des alias sont bloqués avant la mutation.
