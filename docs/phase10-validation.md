# Preuves — phase 10

Exécutions sur Windows, dans `creditS` Python 3.11.9, avec Evidently 0.7.20
et Fairlearn 0.13.0. Airflow 2.11.0 reste exclusivement dans Docker.

- `make monitor` exécuté jusqu'au bout : les huit rapports HTML/JSON Evidently
  sont produits ; l'assertion des scénarios valide zéro alerte sur 01–03 et
  des alertes sur 04–08. Résultats détaillés dans `docs/monitoring.md` et
  `reports/monitoring/validation.json` (sortie locale non versionnée).
- Réserve de production : 6 151 identifiants uniques, sans recouvrement entre
  lots ni avec train/validation/test ; contrat ordonné de 261 features conservé.
- `dvc repro --single-item simulate_production`, `dvc status simulate_production`
  et `dvc push` réussis. Le stage utilise le candidat mitigé figé ; les autres
  stages d'entraînement ne sont pas réexécutés pour cette vérification locale.
- Lot 08 à l'horloge 8 : labels indisponibles, AUC non calculée, performance
  signalée non évaluable et aucune alerte. À l'horloge 10 : AUC 0 et alerte
  de performance, sans inventer une dérive des features ou des scores.
- Lot normal 02 : AUC brute 0,6622 sous 0,70, avertissement conservé. Borne
  supérieure bootstrap 0,7340 ; pas d'alerte avec la règle d'incertitude configurée.
  Une règle stricte sur l'AUC brute déclenche bien dans le test dédié.
- Lot 07 : DI genre 0,0564, DI âge 0,5820, PSI score 1,5449 et AUC 0,5673 ;
  alertes de prédiction, feature critique, performance et équité.
- Intégration des événements API par identifiant de requête : séparation des
  runs, contrat ordonné, absence d'identifiant client brut et rapport Evidently
  réellement généré avec des features manquantes. Aucune donnée sensible inférée.
- Correction vérifiée d'un échec des graphiques Evidently avec des index texte
  et NumPy 2 : alignment vérifié puis index d'observation pour les graphiques.
- `make test` : **93 tests réussis**, 25 avertissements de bibliothèques connus.
  Les neuf nouveaux tests couvrent simulation, reports Evidently réels,
  incertitude de performance, support d'équité et ingestion des logs API.
- Ruff, contrôle du formatage (71 fichiers) et `git diff --check` réussis.
- `make up` réussi après ajout des montages des artefacts et du volume de logs
  en lecture seule dans Airflow ; les cinq services du projet sont healthy.
- Airflow lot 01, run `phase10__20261002T224159708667` : DAG réussi,
  `no_alert=success`, `trigger_training=skipped`, règles normales de `params.yaml`.
- Airflow lot 08 à l'horloge 10, run `phase10__20261002T224520685185` :
  DAG réussi, `trigger_training=success`, `no_alert=skipped`, sans seuil forcé.
  Le DAG déclenché `monitoring__phase10__20261002T224520685185` réussit toutes
  les étapes jusqu'à `reject_model=success` et `register_and_promote=skipped`.
- Candidat MLflow run `ead82f7fcf8f46ea9cbd7dac79646c45`, version **5** du
  Registry Docker : rejeté pour AUC 0,731135 < 0,74. Les règles d'équité,
  les supports des groupes et les six artefacts SHAP passent. Le Registry
  Docker contient cinq versions rejetées et aucun alias `champion`.

Les preuves Airflow par run sont archivées dans
`runtime/airflow/validation/monitoring_<hash>.json`. Les sorties de tâches et
rapports sont conservés dans les espaces privés de chaque exécution.

```powershell
.\creditS\Scripts\Activate.ps1
make monitor
python -m src.monitoring.run_batches --batch-id 08 --observed-batch 8
python scripts/verify_airflow.py --trigger monitoring --batch-id 01 --normal-thresholds --expect-no-alert --wait
python scripts/verify_airflow.py --trigger monitoring --batch-id 08 --observed-batch 10 --normal-thresholds --wait
make test
```

Ces validations mesurent le biais et la dérive selon les métriques documentées.
Elles ne qualifient pas un champion Home Credit et ne constituent pas encore
le scénario complet v1/v2/v3 de la phase 11.
