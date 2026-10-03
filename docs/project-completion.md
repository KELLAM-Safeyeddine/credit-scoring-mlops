# Projet terminé : champion Home Credit et preuves finales

J'ai terminé le projet en qualifiant le candidat v3 sur son test indépendant, puis en déployant le champion dans l'API Docker.

La chaîne DVC → MLflow → Airflow → Docker → FastAPI → Evidently est livrée.
Après gel du plan, le candidat v3 figé a été évalué **une seule fois**
sur son test indépendant, accepté par le quality gate et servi dans l'API Docker.
Les douze phases et la qualification finale du projet pédagogique sont terminées.
La démonstration Airflow v1 accepté → drift → v2 discriminatoire rejeté → v3 mitigé
accepté demeure un scénario synthétique distinct, déjà exécuté et archivé.

## Résultat du test indépendant

19 681 clients, 264 features, neuf oracles LightGBM à 500 arbres et seuil
**0,397**, fixés avant test. Aucun fit, calibration, sélection ou recherche de
seuil supplémentaire n'est effectué. Le test v3 est maintenant fermé à la
sélection. Le registre v2 et ses rejets demeurent inchangés.

| Règle sur test | Résultat | Limite | Gate |
|---|---:|---:|---|
| AUC | **0,752030** | ≥ 0,74 | Passe |
| DI acceptation âge | **0,886048** | ≥ 0,8 | Passe |
| DI acceptation genre | **0,963729** | ≥ 0,8 | Passe |
| Écart EO âge | **0,061134** | ≤ 0,10 | Passe |
| Écart EO genre | **0,020345** | ≤ 0,10 | Passe |
| Recall défaut | 0,789805 | ≥ 0,50 | Passe |
| Taux d'acceptation | 0,542503 | Entre 0,10 et 0,95 | Passe |
| Supports des groupes comparés | Suffisants | ≥ 100 / ≥ 20 par classe | Passe |
| SHAP et inférence sans groupes | Vérifiés | Six artefacts, déterministe | Passe |

PR-AUC : 0,225404 ; Gini : 0,504060 ; KS : 0,376728 ; precision : 0,139382.
Confusion : TN 10 343, FP 7 749, FN 334, TP 1 255. Le seuil favorise le rappel
des défauts selon les coûts métier configurés, avec de nombreux faux refus.
Le positif signifie **1 = défaut prédit = refus**, et l'acceptation vaut
`1 - prédiction`. Le biais est mesuré et contrôlé selon les métriques du gate.

Les détails des taux de sélection, TPR, FPR et supports par groupe figurent dans
le [rapport indépendant complet](assets/v3-ensemble-qualification/holdout_report.json).
Les comparaisons de genre portent sur F/M ; l'unique observation XNA est
rapportée mais exclue de ces comparaisons, son support étant insuffisant.

## Promotion et déploiement vérifiés

Le modèle est accepté dans MLflow local (5000), version **6**, alias `champion`.
Le run qualifié est `be0bd59daa9c4b1ba83a23798b016a86`, avec SHA256 du candidat
`fa44ff4fe614630456f5980301b852535e488a4dfb1bd079ecab0df7c2244ecd`.
Le [plan figé](assets/v3-ensemble-qualification/protocol.json) et ses empreintes
passent encore le contrôle après évaluation. Le registre de réservation parent
`home-credit-independent-v3.json` contient une unique évaluation terminée.

Le modèle et les preuves sont ensuite transférés **via les API MLflow** au
serveur Docker (5001), sans copier une base SQLite vivante ni réévaluer le test.
Le run importé est `ebedf3ecbea948e482e0547193c0db3e`, version **6**, champion.
Le pipeline, seuil, métriques, encoder et SHAP sont conservés. Les rapports
importés indiquent leur run source ; les preuves originales restent archivées.
Le gate complet est appliqué à nouveau aux preuves importées avant promotion.

L'API principale 8000 recharge ce champion et les contrôles HTTP réels passent :

- `/health`, `/model-info`, `/predict`, `/explain` : **200**.
- Prédiction et explication identiques ; trois raisons SHAP, seuil 0,397.
- Payload invalide : **422**.
- Probe synthétique : score 0,2371972303729198, identique au modèle source local.
- Deux événements vérifiés : 264 features, pseudonymes HMAC, aucun identifiant brut.
- Cinq services Docker sains ; MLflow local/Docker et Airflow accessibles.

Les deux événements de probe sont aussi traités par l'adaptateur API → Evidently,
avec les **264 colonnes du champion** et une référence de 400 lignes de validation
v3. Un rapport HTML et JSON est créé, séparé des versions historiques. Le PSI
élevé de ces deux probes identiques ne démontre pas une dérive de production.
Performance et équité en trafic API exigent respectivement les résultats retardés
et une jointure d'audit séparée par request_id ; ces données ne sont pas inventées.
Les lots simulés du DAG de monitoring historique gardent leur rôle diagnostique
distinct, et la boucle de réentraînement est prouvée par la démonstration Airflow.

Un premier import s'est interrompu à cause de l'encodage Windows cp1252 lors
de l'affichage d'un emoji MLflow. Son run `c1987fe40e8e4a29930466e1f6dd44cb`
est conservé comme `FAILED`. Le CLI utilise désormais UTF-8 ; la reprise copie
le même modèle qualifié, sans entraînement et sans nouvel accès au test.

## Validation et limites

**160 tests passent en 32,01 secondes**, 41 avertissements de dépendances.
Ruff et le formatage des 100 fichiers Python sont conformes. Les nouveaux tests
interdisent le transfert d'un candidat non qualifié avant toute mutation cible.
Les checkpoints, rapports, registre de test et preuves finales sont suivis par
DVC dans `runtime/v3-ensemble-qualification`, puis poussés au remote local.
Les modèles binaires et `creditS` restent exclus de Git.
J'ai défini la CI GitHub et publié le projet sur GitHub après cette clôture locale.
Son statut distant est consultable dans l'onglet Actions du dépôt. Les builds et la démonstration antérieurs sont documentés dans
le [bilan de livraison](final-validation.md), sans les présenter comme réexécutés ici.

Le score est non calibré, la validation a servi à plusieurs études, et les
entrées API ne couvrent qu'un sous-ensemble des variables d'entraînement.
Les entrées absentes sont signalées puis imputées avec le transformateur train.
Le dataset historique ne représente pas nécessairement une population actuelle.
Le gate du projet ne constitue pas une qualification réglementaire ni une
autorisation de décision bancaire automatisée. Voir la [Model Card](MODEL_CARD.md).

## Preuves et commandes de contrôle

[Test](assets/v3-ensemble-qualification/holdout_report.json),
[registre unique](assets/v3-ensemble-qualification/holdout_ledger.json),
[promotion locale](assets/v3-ensemble-qualification/final_local_promotion.json),
[transfert Docker](assets/v3-ensemble-qualification/registry_transfer.json),
[API](assets/v3-ensemble-qualification/api_verification.json),
[traçabilité finale](assets/v3-ensemble-qualification/completion_checks.json),
[services](assets/v3-ensemble-qualification/final_services.json),
[monitoring API](assets/v3-ensemble-qualification/api_monitoring_verification.json),
[tests](assets/v3-ensemble-qualification/final_software_checks.json).

```powershell
.\creditS\Scripts\Activate.ps1
python -m pytest -q
python -m ruff check .
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/model-info
docker compose ps
```

UI MLflow Docker : http://localhost:5001 ; UI MLflow locale : http://localhost:5000 ;
Airflow : http://localhost:8081 ; documentation API : http://localhost:8000/docs.
Il n'est pas nécessaire de relancer l'évaluation du test pour vérifier ces résultats.
