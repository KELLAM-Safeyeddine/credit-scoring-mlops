# Qualification Home Credit — recherche complémentaire

Campagne du 2 octobre 2026 : 18 essais LightGBM, six configurations et trois
puissances de repondération (1,0 ; 1,25 ; 1,5), avec `sample_frac=0.1`.
Les 14 760 observations de train et 4 920 de validation restent identiques
aux expériences précédentes. Les transformations et poids utilisent uniquement
le train. Les attributs sensibles restent exclus des 261 features du modèle.
Le test réservé n'a pas été lu.

Les 18 runs sont terminés dans l'expérience MLflow locale
[credit-scoring-qualification](http://localhost:5000/#/experiments/2).
Chaque run conserve paramètres, métriques, rapport Fairlearn, contrat de features,
encodeur et provenance DVC. Le hash du snapshot brut reste
`6d6f2659068bf71e19aece18bd8736eb.dir`.

| Essai | AUC validation | DI âge | Écart EO âge | DI genre | Écart EO genre | Résultat du filtrage |
|---|---:|---:|---:|---:|---:|---|
| 7, meilleure AUC | 0,74151 | 0,74882 | 0,11741 | 0,94879 | 0,08481 | Équité insuffisante |
| 13, meilleure AUC parmi les seuils équitables | 0,73872 | 0,80871 | 0,08158 | 0,98183 | 0,05469 | AUC sous 0,74 |
| 16, proche des deux limites | 0,73954 | 0,80122 | 0,10263 | 0,97945 | 0,02193 | AUC et EO insuffisants |

L'essai 13 utilise 300 arbres, 7 feuilles, un taux d'apprentissage de 0,05,
`min_child_samples=80`, `reg_lambda=5`, une puissance de repondération de 1,0
et un seuil global de refus de 0,56. Il respecte les contraintes d'utilité
et d'équité, mais manque le minimum AUC de 0,00128. Cet écart ne justifie
aucune exception au gate.

**Aucun candidat qualifiable dans cette campagne.** Aucun modèle enregistré ou
promu par ce script ; les alias Home Credit restent vides dans les deux Registry.
Les artefacts SHAP et le gate final n'ont pas été exécutés puisqu'aucun candidat
ne franchit le filtrage préalable. L'absence de champion demeure explicite.
Les résultats complets sont disponibles en [CSV](assets/qualification-search.csv)
et [JSON](assets/qualification-search.json), avec les identifiants de runs et
les configurations de chaque essai.

Vérification exécutée dans `creditS` : **104 tests réussis** en 34,29 secondes
(25 avertissements de dépendances), Ruff sans erreur et 80 fichiers Python
conformes au formatage. Les nouveaux tests contrôlent le budget, les paramètres
autorisés, les puissances non finies, chaque règle du filtrage et le rejet de
splits qui se recouvrent avant entraînement. L'incident d'encodage console
Windows de la première tentative est corrigé à l'entrée du script.

La recherche peut être relancée, toujours dans le venv :

```powershell
.\creditS\Scripts\Activate.ps1
python -c "import sys; print(sys.prefix)"
python -m src.models.qualify
python -m pytest -q
python -m ruff check .
```

Elle crée une nouvelle campagne privée dans `runtime/qualification/`, bornée
par `params.yaml`, sans changer les alias ni les résultats historiques du pipeline.
Le filtrage sélectionne sur validation ; il ne remplace pas le gate final avec
vérification des artefacts SHAP et comparaison au champion éventuel.

Les essais répétés sur la même validation peuvent surajuster la sélection.
Une suite raisonnable est une campagne préspécifiée sur davantage de données,
avec un budget mémoire adapté, puis une évaluation indépendante du candidat figé.
Le test actuellement réservé doit rester hors de toute sélection ou réglage.

## Campagne étendue à 20 % des données

Une seconde campagne double les données le 2 octobre 2026, avec environ 2,4 Go
de RAM libre au départ : 61 502 clients, dont 29 520 train, 9 840 validation,
9 840 test et 12 302 production. Tous les clients de la première campagne
conservent leur affectation. Seuls les clients jusque-là exclus sont répartis
de façon stratifiée entre les quatre splits. Le test est préparé et transformé
pour la cohérence des exports ; ses labels et scores ne servent à aucune
sélection, comparaison de modèles ou évaluation dans cette campagne.

Les catégories sont apprises à nouveau uniquement sur le train étendu :
le nouveau contrat contient **264 features**, identiques pour les quatre splits.
Il est distinct du contrat historique à 261 features et n'est pas déployé.
Les fichiers historiques `data/interim` et `data/processed` restent inchangés.

La campagne est préspécifiée à six essais : configurations à 7 feuilles de
la première campagne, repondération 1,0 / 1,25 / 1,5. Les gates sont inchangés.

| Essai | Puissance | AUC validation | DI âge | Écart EO âge | Seuil équitable et utile trouvé |
|---|---:|---:|---:|---:|---|
| 1 | 1,0 | 0,74318 | 0,76366 | 0,10728 | Non |
| 2 | 1,25 | 0,73862 | 0,78416 | 0,08580 | Non |
| 3 | 1,5 | 0,73272 | 0,80306 | 0,07348 | Oui, AUC insuffisante |
| 4 | 1,0 | 0,74332 | 0,75036 | 0,09432 | Non |
| 5 | 1,25 | 0,73811 | 0,79322 | 0,10124 | Non |
| 6 | 1,5 | 0,73152 | 0,80684 | 0,06916 | Oui, AUC insuffisante |

**Aucun candidat qualifiable** dans les six runs terminés de la campagne
`da2b18cd6ffe`. Les alias Home Credit restent vides aux ports 5000 et 5001.
Le gate final et SHAP ne sont pas exécutés sur ces candidats éliminés en amont.
Cette expérience ne prouve pas qu'un champion conforme est impossible :
augmenter l'échantillon seul ne résout pas ici le compromis performance/équité.
La validation étendue comprend des clients de la validation historique déjà
consultée ; elle ne constitue pas une évaluation indépendante finale.

Le snapshot est suivi dans un dépôt DVC sans SCM isolé, sous
`runtime/qualification-expanded-20/`, et poussé au même remote local :

- interim : `c6aa5659095e6f56e14037f470cadb74.dir` ;
- processed : `df0b7377bfe4fa4d60f8ac341224774e.dir`, 16 fichiers, 72 910 177 octets ;
- 22 fichiers poussés, incluant les manifests de répertoires.

Chaque run MLflow contient le hash DVC du processed étendu, les deux descripteurs,
le manifeste des splits, les hashes exacts des entrées train/validation et son
nouvel encodeur. Le `dvc.lock` historique est explicitement marqué comme snapshot
du pipeline initial, sans prétendre décrire le nouveau snapshot.
[Preuves et configurations](assets/qualification-expanded-20/search.json).

Pour récupérer le snapshot dans un clone ayant accès au même remote local :

```powershell
.\creditS\Scripts\Activate.ps1
$qualificationPython = (Resolve-Path .\creditS\Scripts\python.exe).Path
New-Item -ItemType Directory runtime/qualification-expanded-20 -Force | Out-Null
Copy-Item docs/assets/qualification-expanded-20/*.dvc runtime/qualification-expanded-20/
Copy-Item docs/assets/qualification-expanded-20/split_manifest.json runtime/qualification-expanded-20/
Push-Location runtime/qualification-expanded-20
# Initialisation seulement si .dvc n'existe pas encore.
if (-not (Test-Path .dvc)) {
    & $qualificationPython -m dvc init --no-scm
    $qualificationRemote = (Resolve-Path ../../../dvc-storage).Path
    & $qualificationPython -m dvc remote add -d local $qualificationRemote
}
& $qualificationPython -m dvc pull
Pop-Location
python -m src.models.qualify --params docs/assets/qualification-expanded-20/params.yaml
python -m pytest -q
```

Pour reconstruire une expansion dans un **nouveau** dossier, utiliser
`python -m src.data.expand_development --output runtime/une-nouvelle-campagne --fraction 0.2`,
puis `src.features.build_features --params` avec le `params.yaml` généré.
Le script refuse d'écraser un dossier existant et rejette les memberships qui
se recouvrent, les IDs absents de la source ou une fraction qui n'ajoute aucun client.
Le test de régression MLflow vérifie l'enregistrement de la provenance sur deux
runs successifs : une variable temporaire du premier run ne peut plus remplacer
le chemin du snapshot du suivant. La tentative interrompue reste visible dans
MLflow ; les six résultats du tableau proviennent uniquement de la relance terminée.

Validation finale exécutée dans `creditS` : **109 tests réussis en 30,92 secondes**,
25 avertissements connus, Ruff sans erreur et 82 fichiers Python formatés.
Les six runs ont le statut `FINISHED` et portent le hash DVC attendu.

## Campagne de réduction des proxys

Neuf essais supplémentaires sur le même snapshot à 20 %, dans la campagne
`8e316de7f5c3`, comparent trois profils et trois puissances de repondération
(0,5 ; 1,0 ; 1,25). LightGBM utilise 300 arbres, 15 feuilles,
`learning_rate=0.03`, `min_child_samples=100` et `reg_lambda=10`.
La sélection repose uniquement sur train/validation ; aucun résultat de test
n'est utilisé. Les règles du gate sont inchangées.

Le diagnostic utilise la corrélation de Pearson sur le train : `EXT_SOURCE_1`
avec l'âge vaut environ 0,606 ; les indicateurs de retraite et d'emploi sont
aussi fortement liés à l'âge. Ces associations ne constituent pas une preuve
causale de discrimination. [Diagnostic](assets/qualification-proxies/age_correlations.json).

- `external` neutralise `EXT_SOURCE_1` et `EXT_SOURCE_MEAN`, qui contient aussi
  le score externe 1. Les scores 2 et 3 restent disponibles séparément.
- `employment` neutralise 88 colonnes : catégories de revenu, employeur,
  profession, ancienneté, anomalie d'emploi et deux indicateurs associés.
- `combined` neutralise les deux ensembles, soit 90 colonnes.

`ProxyMask`, intégré au pipeline persisté, remplace ces entrées par zéro avant
l'imputation à l'entraînement et à l'inférence. Le contrat externe conserve
les 264 colonnes, leur ordre et les index. Les colonnes neutralisées ne peuvent
plus influer sur ce modèle et leurs contributions Tree SHAP sont nulles.
Cette neutralisation n'élimine pas les autres proxys présents dans les données.

| Essai | Profil | Puissance | AUC | DI âge | Écart EO âge | Seuil équitable et utile |
|---|---|---:|---:|---:|---:|---|
| 1 | external | 0,5 | 0,73843 | 0,69453 | 0,15177 | Non |
| 2 | employment | 0,5 | 0,74690 | 0,72452 | 0,17390 | Non |
| 3 | combined | 0,5 | 0,73431 | 0,75043 | 0,14794 | Non |
| 4 | external | 1,0 | 0,73119 | 0,79902 | 0,10728 | Non |
| 5 | employment | 1,0 | 0,73970 | 0,76955 | 0,13325 | Non |
| 6 | combined | 1,0 | 0,72890 | 0,79223 | 0,08387 | Non |
| 7 | external | 1,25 | 0,72913 | 0,81485 | 0,08025 | Oui, AUC insuffisante |
| 8 | employment | 1,25 | 0,73526 | 0,75455 | 0,09963 | Non |
| 9 | combined | 1,25 | 0,72459 | 0,80169 | 0,07407 | Oui, AUC insuffisante |

**Aucun candidat qualifiable.** Les neuf runs sont `FINISHED` dans MLflow et
les alias Home Credit restent vides dans les deux Registry. Les modèles
éliminés n'ont pas fait l'objet d'une promotion ni du gate final SHAP.
Le test de compatibilité SHAP a été exécuté sur une fixture entraînée avec
LightGBM ; il ne constitue pas un artefact d'explicabilité d'un champion réel.
Les masks exacts, métriques d'utilité, paramètres et rapports Fairlearn sont
loggés dans chaque run. Le snapshot DVC processed reste
`df0b7377bfe4fa4d60f8ac341224774e.dir`.

[Résultats complets](assets/qualification-proxies/search.json) et
[configuration reproductible](assets/qualification-proxies/params.yaml).
Après récupération du snapshot étendu documentée ci-dessus :

```powershell
.\creditS\Scripts\Activate.ps1
python -m src.models.qualify --params docs/assets/qualification-proxies/params.yaml
python -m pytest -q
python -m ruff check .
```

Cette campagne ne montre pas d'amélioration suffisante du compromis. Une piste
distincte à examiner serait une repondération ciblée sur l'âge seul, au lieu
des intersections genre/âge ; elle doit toujours auditer les deux attributs
et conserver les mêmes seuils. La répétition des campagnes sur cette validation
renforce la nécessité d'évaluer une seule fois le candidat figé sur le test réservé.

Validation exécutée dans `creditS` : **111 tests réussis en 39,53 secondes**,
27 avertissements de dépendances, Ruff sans erreur et 84 fichiers formatés.
Les tests vérifient l'invariance des scores aux valeurs masquées, leur attribution
SHAP nulle, la conservation du contrat et le rejet d'une entrée mal ordonnée.
