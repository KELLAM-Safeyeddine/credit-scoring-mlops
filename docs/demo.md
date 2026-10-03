# Démonstration de la boucle fermée

`make demo` déroule une boucle réelle DVC → MLflow → Airflow → Docker/FastAPI
→ Evidently, sur des **données synthétiques**. Elle prouve les transitions du
système ; elle ne qualifie pas un modèle Home Credit. Le candidat Home Credit
avait initialement été rejeté ; la qualification finale v3 a depuis réussi.
L'API principale sert désormais le champion Home Credit version 6.
[Résultats définitifs](project-completion.md).

## Exécution

```powershell
.\creditS\Scripts\Activate.ps1
make up
make demo
make test
```

Sur Linux, activer `source creditS/bin/activate` avant ces commandes.
La stack Docker doit être démarrée. Chaque invocation crée un identifiant unique,
un modèle MLflow `credit-demo-<id>` et une API Docker sur un port local dynamique.
Le réseau, MLflow et Airflow sont ceux du projet ; le modèle `credit-scoring`,
ses versions et ses alias ne sont pas modifiés. Le conteneur API de démonstration
est supprimé à la fin ; données, rapports, logs et modèles sont conservés pour
inspection. Les credentials temporaires restent dans le dossier ignoré
`runtime/demo/<id>/`, jamais dans les XCom ni dans Git.

## Chaîne exécutée

1. Le Python local `creditS` génère les petites fixtures, initialise un dépôt DVC
   privé sans SCM, suit `fixture/` et pousse vers son remote local `dvc-storage/`.
   Les tables de train, validation et production ont des identifiants distincts.
2. Le DAG `credit_demo` vérifie la provenance et les contrats puis entraîne v1,
   calcule les métriques Fairlearn et les six artefacts SHAP. Le gate réel,
   inchangé, vérifie les artefacts téléchargés et accepte v1.
3. L'API Docker charge `champion=1` via MLflow HTTP ; `/predict` et `/explain`
   sont vérifiés avec des scores et versions identiques et trois raisons SHAP.
4. Un lot de production séparé subit un changement de revenu et de `EXT_SOURCE_2`.
   Il est scoré par v1 ; Evidently génère le rapport et déclenche les règles
   de dérive de données et de prédictions. Les labels ne sont pas utilisés
   pour ce déclenchement.
5. `TriggerDagRunOperator` déclenche **automatiquement** le DAG distinct
   `credit_demo_retraining`. V2 utilise une meilleure feature mais un échantillon
   biaisé par genre/classe. Son AUC s'améliore, tandis que les métriques d'équité
   échouent. La version est enregistrée `rejected` ; v1 reste champion.
6. V3 utilise le même échantillon biaisé, avec des poids de reweighing par
   intersection genre/âge et label, calculés exclusivement sur le train.
   Fairlearn mesure l'équité sur la validation inchangée. Le gate accepte v3,
   met `previous=1`, `champion=3`, puis le DAG appelle le rechargement protégé
   de l'API et vérifie `/predict` et `/explain` sur la version 3.

La logique métier est dans `src/demo/lifecycle.py`, les fixtures dans
`src/monitoring/demo_data.py`. Les DAGs restent minces et utilisent le worker
Python séparé d'Airflow. Aucun autre moteur d'orchestration, tracking ou monitoring
n'est ajouté. Le trigger attend dans `default_pool`, permettant aux entraînements
du DAG enfant d'utiliser le pool `credit_compute` sans blocage mutuel.

## Configuration et résultats

`params.yaml` contient le seuil global de démonstration 0,37, les multiplicateurs
de drift et les paramètres LightGBM. Le seuil est fixé avant le scoring de
production. Les exigences du gate restent AUC ≥ 0,74, DI ≥ 0,8, écart d'égalité
des chances ≤ 0,10, supports, utilité et SHAP vérifié. Les métriques sont calculées
à partir des modèles et labels réels des fixtures, sans valeurs de gate fabriquées.

| Version | AUC validation | DI genre | Écart égalité des chances genre | DI âge | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| v1 | 0,7548 | 1,0000 | 0,0000 | 1,0000 | Accepté |
| v2 | 0,9353 | 0,6509 | 0,1950 | 1,0000 | Rejeté pour équité |
| v3 reweighing | 0,9583 | 1,0000 | 0,0000 | 1,0000 | Accepté |

Les valeurs égales entre groupes viennent des cohortes synthétiques appariées,
pas d'une garantie d'absence de biais sur des emprunteurs réels. La prévalence
synthétique est de 50 %, distincte des ~8 % Home Credit. Le revenu est un proxy
contrôlé du genre pour exposer ce risque ; genre, âge et features dérivées
d'âge restent absents des entrées du modèle. Le scénario modifie également
la qualité d'une feature entre v1 et v2 : il démontre le cycle de vie plutôt
qu'une adaptation estimée au drift réel du marché du crédit.

Les artefacts privés se trouvent dans `runtime/demo/<id>/` : descriptor DVC,
fixtures, `v*/gate.json`, SHAP, `monitoring/drift.html`, `alerts.json`,
`api_v1.json`, `api_v3.json`, événements pseudonymisés et `validation.json`.
Les modèles de démonstration restent consultables dans MLflow sur 5001,
les deux DAGs dans Airflow sur 8081.

## CI

`.github/workflows/ci.yml` s'exécute sur push, pull request ou déclenchement manuel.
Le runner Ubuntu crée `.ci-venv` avec Python 3.11.9 et installe le même
`requirements.txt` que le développement. Les tests fabriquent leurs fixtures :
aucun Kaggle, remote DVC privé ou secret n'est nécessaire. Ruff et pytest
précèdent les builds Docker API et Airflow ; aucun déploiement n'est effectué.
Les actions [checkout](https://github.com/actions/checkout) et
[setup-python](https://github.com/actions/setup-python) sont épinglées aux commits
résolus de leurs versions v6. Les tests locaux imposent `creditS` ; seule
l'exécution GitHub Actions attend `.ci-venv`. Airflow reste absent de ce venv.

Aucun remote Git n'est encore configuré : le workflow est prêt et ses commandes
Ruff/pytest/build ont été exécutées localement. Aucun résultat GitHub Actions
distant n'est revendiqué. La démonstration Airflow complète est une validation
locale d'intégration ; elle n'est pas exécutée dans la CI légère.
