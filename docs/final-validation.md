# Livraison et validation finale — 3 octobre 2026

## État de livraison

La chaîne **DVC → MLflow → Airflow → Docker → FastAPI → Evidently** est
implémentée, avec tests, quality gate, audits Fairlearn, explications SHAP,
monitoring, démonstration et documentation. **La qualification d'un champion
Home Credit reste non obtenue.** La démonstration fonctionne sur un modèle
synthétique distinct ; ses résultats ne valident pas un modèle de crédit réel.

Cette clôture de développement n'assouplit aucune règle, ne sélectionne aucun
candidat à partir d'un test déjà consulté et ne promeut aucun candidat rejeté.
Genre, âge et ratio emploi/âge restent exclus des features du modèle.

## Vérifications exécutées

| Vérification | Résultat |
|---|---|
| Venv local | Python 3.11.9 dans `creditS` |
| `make test` | 133 tests réussis, 27 avertissements de dépendances, 124,56 s |
| `make lint` | Ruff sans erreur |
| Ruff format | 93 fichiers Python conformes |
| `make train` | Pipeline DVC complet rejoué, features et SHAP régénérés |
| DVC racine | À jour après exécution ; 27 nouveaux fichiers poussés au remote |
| `make promote` | Candidat local version 4 rejeté pour AUC insuffisante |
| `make monitor` | Huit lots rejoués ; aucune alerte 01–03, alertes attendues 04–08 |
| `make up` | Images API/Airflow construites ; cinq services sains |
| MLflow local / Docker | `/health` HTTP 200 sur 5000 et 5001 |
| Airflow | `/health` HTTP 200 sur 8081 ; DAGs de démonstration réussis |
| API Home Credit | `/live` HTTP 200 ; `/health` HTTP 503, aucun champion |
| Test indépendant v2 | Manifeste/configuration SHA256 valides ; registre d'évaluation absent |
| Registry Home Credit | Alias vides sur les deux serveurs MLflow |

Le healthcheck Docker vérifie la vie du processus via `/live`. La disponibilité
du scoring dépend du champion accepté et reste signalée séparément par `/health`.
Le HTTP 503 protège l'inférence lorsque le modèle n'est pas qualifié.
Les avertissements des tests ne sont pas présentés comme des erreurs corrigées.

Les descripteurs DVC archivés dans `docs/assets/` sont désormais exclus de la
recherche DVC racine via `.dvcignore`. Ces copies documentaires correspondent
aux dépôts isolés sous `runtime/` et ne sont pas des sorties actives du pipeline
racine. Cela évite les faux signalements de cache manquant pour la documentation.

## Démonstration rejouée

Identifiant **`03ef5c1d870c`**, modèle Registry **`credit-demo-03ef5c1d870c`**.
Le DAG `credit_demo` et le DAG enfant `credit_demo_retraining` sont `success`.
Evidently détecte le drift, puis Airflow déclenche automatiquement le DAG enfant.

| Version synthétique | Gate | Conséquence |
|---|---|---|
| v1 | Accepté | Déploiement et prédiction/SHAP vérifiés |
| v2 | Rejeté : DI et EO genre | v1 conserve l'alias champion |
| v3 mitigé Fairlearn | Accepté | champion=3, previous=1, API rechargée |

La preuve vérifie que les alias Home Credit restent inchangés. Le conteneur API
temporaire est supprimé par le script ; les artefacts, journaux pseudonymisés,
rapports HTML, modèles et états Airflow sont conservés. Les secrets de contexte
restent dans `runtime/`, ignoré par Git.
[Preuve de démonstration](assets/final-validation/demo.json).

## Surveillance Home Credit rejouée

Le monitoring local utilise le candidat mitigé figé de diagnostic, explicitement
distinct d'un champion déployé. Les huit rapports Evidently et audits Fairlearn
sont recalculés après régénération des lots. Chaque lot est observé après le délai
configuré de deux lots pour la vérité terrain.

| Lots | Résultat |
|---|---|
| 01–03 | Aucune alerte |
| 04–05 | Dérive financière : revenu, crédit et ratios |
| 06 | Dérive de population : AGE_GROUP ; EXT_SOURCE_MEAN également signalé |
| 07 | Dérives de prédictions, données, performance et équité genre/âge |
| 08 | Performance, AUC 0 après disponibilité des labels retardés |

Le lot normal 02 a AUC 0,66217, mais borne supérieure bootstrap 0,73404 : il
ne déclenche pas la règle de performance, qui utilise cette borne et le seuil
0,70. Ce comportement est configuré et documenté, pas corrigé après observation.
Le script local archive les alertes sans dispatcher de DAG ; le déclenchement
automatique est vérifié séparément dans la démonstration Airflow ci-dessus.
[Résultats complets des huit lots](assets/final-validation/monitoring.json).

Pendant ce calcul, la mémoire disponible est descendue à environ 400 Mo.
Les deux services Airflow ont été arrêtés temporairement, puis redémarrés à la
fin des rapports ; aucun service n'est laissé arrêté par cette adaptation.
La surveillance complète des 261 features produit des HTML volumineux et se
lance séparément de la démonstration sur cette machine.

## Qualification Home Credit : limite restante

Le pipeline racine rejoué utilise l'échantillon de développement à 10 %,
14 760 lignes de train, 4 920 de validation et 261 features. Il retrouve les
mêmes métriques que l'expérience historique :

| Modèle du pipeline racine | AUC validation |
|---|---:|
| Régression logistique | 0,72103 |
| Random Forest | 0,72398 |
| XGBoost | 0,73724 |
| LightGBM | 0,74865 |

La baseline LightGBM échoue à l'équité par âge. Le candidat mitigé
`reweighing_1.5` atteint AUC 0,73114, DI âge 0,81425 et EO âge 0,09211.
Les six artefacts SHAP sont présents et vérifiés, avec erreur de reconstruction
4,44×10⁻¹⁶. Le gate le rejette pour AUC sous 0,74 : version **4** du Registry
local, run **`6e42407b49084c6dbcf619300da6be3a`**, sans alias champion.
Cette réexécution vérifie la reproductibilité du pipeline et ne constitue pas
une nouvelle recherche sur les splits v2 ni une consultation de son test.
L'entraînement sur la totalité des données avec `sample_frac=1.0` n'est pas
exécuté dans cette livraison ; les métriques ne sont pas des résultats à 100 %.
[Comparaison rejouée](assets/final-validation/model-comparison.json),
[gate appliqué](assets/final-validation/home-credit-gate.json).

Le candidat historique étendu passe la validation mais échoue au test consulté
une fois : AUC 0,74618, DI âge 0,79692, écart EO âge 0,25758. Il reste rejeté.
[Rapport et protocole historique](qualification.md).

Les six essais préspécifiés du [protocole v2](protocol-v2.md) dépassent 0,74
d'AUC validation, mais échouent sur EO âge. Son test demeure fermé.
L'[étude de développement Fairlearn](constrained-development.md) ultérieure
examine quatre réductions et huit scores déterministes. Le point satisfaisant
l'équité atteint DI âge 0,94501, EO âge 0,05894 et AUC 0,73665 : il échoue à
l'AUC minimale de 0,74. Aucun de ces points n'est promu.

Obtenir un champion réel demanderait une nouvelle étude bornée, puis un plan
de qualification figé, des preuves SHAP adaptées au modèle retenu et une
évaluation indépendante unique. La validation réutilisée pour le développement
ne constitue plus une nouvelle preuve indépendante. Aucune échéance ou réussite
de cette qualification n'est garantie par la livraison du code.

## Utilisation et reprise

```powershell
.\creditS\Scripts\Activate.ps1
python -c "import sys; print(sys.prefix)"
make up
make demo
make test
make lint
```

Accès : MLflow local `http://localhost:5000`, MLflow Docker
`http://localhost:5001`, Airflow `http://localhost:8081`, API
`http://localhost:8000/docs`. Les identifiants Airflow sont dans `.env` local,
jamais dans le dépôt. Airflow tourne uniquement dans Docker.

Pour arrêter les services de ce projet sans supprimer leurs volumes :

```powershell
.\creditS\Scripts\Activate.ps1
make down
```

Les services restent démarrés à la livraison. La CI GitHub Actions est définie
mais n'a pas été exécutée sur un runner distant : aucun remote Git n'est
configuré. La validation ci-dessus est locale, sur cette machine Windows.
Les limites de population, calibration, proxys et usages sont détaillées dans
la [Model Card](MODEL_CARD.md).
[Preuve de l'état final](assets/final-validation/execution.json).
