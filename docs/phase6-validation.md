# Phase 6 — Quality gate et Registry

Exécutée dans `creditS`, Python 3.11.9, MLflow 3.10.1. Signatures et documentation
installée du client vérifiées, puis appels REST réels et Registry SQLite de test
exécutés. Les alias sont utilisés ; les stages historiques MLflow ne le sont pas.

`quality_gate(candidate, prod, cfg)` retourne un booléen et un détail par règle :
AUC >= 0.74, baisse AUC <= 0.005 contre production, DI acceptation >= 0.8 et
écart TPR du défaut <= 0.10 pour genre et âge, groupes requis suffisamment
représentés, et artefacts SHAP réellement disponibles. Les garde-fous de phase 5
restent applicables : recall >= 0.50, acceptation entre 0.10 et 0.95, inférence
déterministe sans attribut sensible. Une métrique manquante ou non finie échoue.
La comparaison à production exige les mêmes hashes X_val/y_val ; si les données
d'évaluation changent, le champion doit être réévalué sur le même jeu avant
de comparer les AUC. Une absence de champion dispense seulement de cette règle.

Les preuves sont lues depuis les runs FINISHED : métriques cohérentes avec le
bundle local, rapport Fairlearn identique, cinq PNG SHAP non vides et rapport
SHAP correspondant au run et au nombre de features, erreur d'additivité < 1e-6.
Un tag `shap_artifacts_present=true` seul n'est pas une preuve suffisante.

## Résultats réels

| Version Registry credit-scoring | Candidat | Gate | Règles échouées |
|---|---|---|---|
| 1 | LightGBM baseline, AUC 0.7487 | rejected | DI âge 0.5996 ; écart TPR âge 0.1974 |
| 2 | Reweighing 1.5, AUC 0.7311 | rejected | minimum AUC 0.74 |

Les règles ne sont pas assouplies. Aucun alias champion ou previous n'est créé
dans le Registry réel. Ces versions enregistrées sont des candidats rejetés,
pas des modèles déployés. L'absence de champion doit être gérée explicitement
par le futur service ; aucun contournement du gate ne fournit une baseline.
Le scénario final accepté exige encore un candidat performant ET conforme.

Les runs et versions portent gate_status et gate_failed_rules. Les artefacts
`promotion/evidence.json`, `promotion/params.json` et `promotion/quality_gate.json`
conservent les preuves et décisions ; `reports/promotion/` contient les copies
locales ignorées par Git. SQLite et les artefacts sont persistants dans les
dossiers du serveur MLflow. Les décisions Registry ne sont pas un stage DVC
mis en cache : la promotion est une action explicite, puis un appel du DAG
pendant la prochaine phase. Les données/modèles DVC restent inchangés.

Un retry réutilise la version du même run_id. En cas de succès, l'ancien champion
devient previous et le nouveau devient champion. Un rejet laisse les alias
inchangés. Le code détecte un changement de champion pendant l'évaluation et
demande une nouvelle évaluation. Les deux mises à jour d'alias MLflow ne sont
pas transactionnelles ; les promotions doivent être sérialisées par le DAG
(max_active_runs=1 pendant la phase 7).

## Preuves d'exécution

58 tests passent. Les nouveaux cas couvrent acceptation et limites exactes,
AUC insuffisante/non finie, équité genre/âge, preuves SHAP absentes, supports
insuffisants, inférence avec attribut sensible, utilité dégénérée et comparaison
de production non comparable.
Le test d'intégration emploie son propre Registry SQLite temporaire et des
preuves jouets : deux promotions successives, previous conservé, rejet AUC,
retry idempotent, puis rejet pour équité d'un candidat dont l'AUC augmente.
Le champion reste la version 2 de ce Registry isolé ; les tests n'écrivent
pas de champion dans le serveur du projet.

```powershell
.\creditS\Scripts\Activate.ps1
make test
python -m scripts.verify_registry
make promote
python -m ruff check .
python -m ruff format --check .
python -m dvc status
```

`make promote` réévalue le candidat mitigé avec les exigences actuelles et
confirme son rejet. La commande retourne normalement un rapport de décision ;
un rejet métier n'est pas une panne du pipeline. Les erreurs d'infrastructure
ou de cohérence des preuves interrompent le traitement sans promotion.
