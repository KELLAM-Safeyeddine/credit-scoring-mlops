# Surveillance des lots

Le monitoring utilise Evidently **0.7.20**, Fairlearn **0.13.0** et les mêmes
fonctions métier en local et dans le DAG Airflow. Le modèle de diagnostic est
`artifacts/fairness_models/candidate.pkl`, figé, avec son seuil 0,55 et son run
MLflow. Il n'est pas un champion déployé : son AUC de validation 0,7311 reste
inférieure au minimum 0,74 du quality gate. Cette limitation reste explicite dans
les manifestes et les rapports ; aucune promotion n'est forcée.

## Données et horloge

Les 6 151 observations de la réserve de production sont disjointes du train,
de la validation et du test. Elles sont réparties sans remplacement entre huit
lots ; `SK_ID_CURR` et les index sont conservés. Les 261 colonnes ordonnées
respectent le contrat du modèle. Les groupes sensibles restent dans des fichiers
d'audit séparés, jamais dans les entrées du modèle.

Les prédictions sont calculées par le pipeline figé. Les labels sont conservés
dans `ground_truth/`, avec une disponibilité simulée à `numéro du lot + 2`.
Avant cette date, le monitoring ne lit pas le fichier de vérité terrain et ne
calcule ni performance, ni métriques d'équité nécessitant les labels. L'horloge
est un mécanisme de démonstration, pas un contrôle d'accès aux fichiers physiques.

Les lots 01–03 restent normaux. Les lots 04–05 multiplient le revenu par 0,8
et le crédit par 1,2, puis recalculent les ratios associés. Le lot 06 comporte
60 % de personnes de moins de 30 ans, sélectionnées dans la réserve réelle.
Cette redistribution réduit leur présence dans les autres lots tardifs ; le PSI
de population de ces lots est donc également publié.

Le lot 07 dégrade les `EXT_SOURCE` des femmes pour simuler une dégradation ciblée.
Le groupe sert uniquement à construire ce scénario contrôlé d'audit. Le lot 08
conserve features et scores, mais réaffecte les défauts aux scores les plus bas
en conservant la prévalence du lot : c'est une dérive conceptuelle synthétique
extrême, pas une représentation estimée du comportement réel des emprunteurs.

## Rapports et alertes

Chaque observation d'un lot produit `drift.html`, `population.html`, les résultats
JSON Evidently, `drift.json`, `fairness.json` et `alerts.json`. Quand les labels
sont disponibles, `performance.html` et les résultats de classification sont
ajoutés. Les chemins locaux incluent le lot et l'horloge, par exemple
`reports/monitoring/08/clock_10/`. Les exécutions Airflow ont leur propre espace
sous `runtime/airflow/runs/`.

Le PSI des features variables utilise un seuil 0,20. La part globale déclenche
au-delà de 30 %, le PSI du score au-delà de 0,20. Les features financières
critiques et `EXT_SOURCE_MEAN` ont un seuil spécifique de 0,10 : cela détecte
une dérive portant sur quelques variables même si la part globale reste faible.
Les 252 features variables sont testées par PSI ; les neuf features constantes
de référence ont un contrôle d'apparition de valeurs nouvelles (seuil 5 %).
Un changement de proportion de valeurs manquantes supérieur à 20 points
déclenche aussi une alerte, y compris lorsqu'une feature devient entièrement vide.

L'âge et le genre ont un rapport de population séparé (PSI > 0,20). Fairlearn
mesure les taux d'acceptation et le disparate impact par attribut :
**1 = défaut prédit = refus ; acceptation = 1 − prédiction**. Après réception
des labels, TPR, FPR, égalité des chances et equalized odds sont également
calculés. Le DI inférieur à 0,8 déclenche une alerte quand les groupes requis
ont un support suffisant. Les groupes insuffisants sont signalés comme non
évaluables, jamais comme équitables. Le monitoring utilise 20 observations
par groupe et 5 par classe pour les métriques conditionnelles ; le gate conserve
ses exigences distinctes de 100 observations et 20 par classe.

La performance utilise un bootstrap stratifié de 200 répétitions et un intervalle
à 95 %. Par défaut, l'alerte exige que la borne supérieure de l'AUC soit sous
0,70. L'AUC brute sous le seuil reste visible via `raw_auc_below_floor`.
Cette règle réduit les alertes sur de petits lots, mais peut retarder la détection.
`auc_alert_rule: point_estimate_below_floor` active la règle stricte sur l'AUC
brute ; le lot normal 02 déclencherait alors une alerte. Aucun de ces réglages
ne modifie le minimum 0,74 du quality gate.

## Résultats exécutés

Valeurs sur les lots aux horloges où leurs labels sont disponibles :

| Lot | AUC | Borne supérieure | PSI score | DI genre | DI âge | Alertes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 01 | 0,7503 | 0,8197 | 0,0289 | 0,9821 | 0,8263 | Aucune |
| 02 | 0,6622 | 0,7340 | 0,0511 | 0,9498 | 0,8011 | Aucune ; avertissement AUC brute |
| 03 | 0,7503 | 0,8156 | 0,0217 | 0,9211 | 0,8544 | Aucune |
| 04 | 0,7101 | 0,7804 | 0,0317 | 0,9391 | 0,9635 | Cinq variables financières |
| 05 | 0,7452 | 0,8091 | 0,0282 | 0,9046 | 0,8941 | Cinq variables financières |
| 06 | 0,7145 | 0,7690 | 0,0885 | 0,9775 | 0,8264 | Population âge (PSI 1,0003), EXT_SOURCE_MEAN |
| 07 | 0,5673 | 0,6475 | 1,5449 | 0,0564 | 0,5820 | Scores, EXT_SOURCE_MEAN, performance, équité |
| 08 | 0,0000 | 0,0000 | 0,0231 | 0,9779 | 0,8595 | Performance après labels |

Au lot 08 observé à l'horloge 8, aucune alerte de performance n'est produite :
les labels sont encore indisponibles. À l'horloge 10, l'alerte de performance
déclenche le réentraînement Airflow ; tout candidat repasse le quality gate.

## Journaux API

`src.monitoring.api_logs.read_api_events` sépare les événements par `model_run_id`,
valide le contrat ordonné et aligne features et scores par `request_id`.
`report_api_events` alimente les rapports Evidently de données et de scores,
y compris avec des entrées partielles et des features manquantes. Les index
sont vérifiés avant de passer à des index d'observation pour les graphiques.

Le volume `api_logs` est monté en lecture seule dans Airflow. L'adaptateur est
testé avec de vrais rapports Evidently, mais le DAG courant consomme les fichiers
de lots simulés. L'audit d'équité des journaux exige une jointure séparée,
autorisée, par identifiant de requête ; la performance exige des labels libérés.
Les groupes sensibles ne sont ni demandés par l'API ni inférés depuis les features.
L'API principale ne produit pas encore de scoring qualifié faute de champion.

## Commandes Windows

```powershell
.\creditS\Scripts\Activate.ps1
make monitor
python -m src.monitoring.run_batches --batch-id 08 --observed-batch 8
python scripts/verify_airflow.py --trigger monitoring --batch-id 01 --normal-thresholds --expect-no-alert --wait
python scripts/verify_airflow.py --trigger monitoring --batch-id 08 --observed-batch 10 --normal-thresholds --wait
make test
```

`make monitor` reproduit uniquement le stage DVC de simulation à partir du
candidat figé, puis vérifie les huit scénarios localement. Il ne déclenche pas
Airflow. Le DAG réalise le déclenchement automatique ; `--normal-thresholds`
utilise les règles de `params.yaml`, sans l'override strict de la démonstration
Airflow de phase 7. Les données sont conservées dans le remote DVC local.
