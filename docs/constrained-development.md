# Étude de développement Fairlearn à coûts métier

Cette étude exploratoire suit les six essais clos du protocole v2. Elle utilise
uniquement ses 44 281 lignes d'entraînement et 14 760 lignes de validation,
avec le contrat de 263 features. Elle ne constitue pas une extension des six
essais préspécifiés ni une qualification indépendante. La validation est désormais
réutilisée pour la recherche ; un résultat favorable exigerait un nouveau plan
de qualification avant toute ouverture de test ou promotion.

## Plan borné avant entraînement

Quatre ajustements `ExponentiatedGradient` Fairlearn **0.13.0** :
`TruePositiveRateParity` ou `EqualizedOdds`, chacun contraint sur l'âge seul ou
l'intersection genre–âge. L'objectif `ErrorRate` utilise les coûts normalisés
FP=1/11 et FN=10/11, correspondant aux coûts métier 1 et 10 du projet.
Chaque oracle LightGBM utilise 150 arbres, 15 feuilles, learning rate 0,05,
min_child_samples 100, reg_lambda 5, quatre threads, seed 20261003,
`class_weight=None` et les poids calculés par la réduction. Six itérations
maximum de l'algorithme externe, eps=0,01, difference_bound=0,02 ; ce budget
ne limite pas le nombre d'appels aux oracles et ne garantit pas la convergence.
L'imputation est ajustée uniquement sur le train.

Deux scores déterministes sont examinés pour chaque réduction : moyenne pondérée
des probabilités des arbres, et moyenne pondérée de leurs décisions binaires.
Le second est un score de vote, pas une probabilité de défaut calibrée ; le
premier n'est pas non plus calibré. Ils ne nécessitent aucun groupe à l'inférence.
Ils **n'héritent pas des garanties de la politique aléatoire Fairlearn**.
Chaque score reçoit donc son propre audit réel sur validation et un seuil global
parmi les 981 valeurs préexistantes. Une absence de seuil faisable est un échec,
même si le diagnostic de moindre pénalité fournit un seuil et une bonne AUC.

Le screening exige les mêmes limites AUC, DI, EO, recall, acceptation et supports.
Il ne remplace pas le gate complet : aucune preuve SHAP d'ensemble ni aucun test
indépendant n'est produit ici. Aucun modèle n'est enregistré ni promu.

API vérifiée dans les versions installées et la documentation officielle :
[ExponentiatedGradient 0.13](https://fairlearn.org/v0.13/api_reference/generated/fairlearn.reductions.ExponentiatedGradient.html),
[objectif de réduction à coûts métier](https://fairlearn.org/v0.13/user_guide/mitigation/reductions.html).

## Résultats exécutés le 3 octobre 2026

| Contrainte / groupes | Score | AUC validation | DI âge | Écart EO âge | Seuil faisable |
|---|---|---:|---:|---:|---|
| TPR / âge | Probabilités | 0,75404 | 0,79585 | 0,21688 | Non |
| TPR / âge | Vote | 0,71022 | 0,66831 | 0,20691 | Non |
| TPR / intersection | Probabilités | 0,74792 | 0,82763 | 0,16864 | Non |
| TPR / intersection | Vote | 0,70349 | 0,69818 | 0,23108 | Non |
| EqualizedOdds / âge | Probabilités | 0,74787 | 0,79133 | 0,12095 | Non |
| EqualizedOdds / âge | Vote | 0,70794 | 0,90346 | 0,10885 | Non |
| EqualizedOdds / intersection | Probabilités | 0,73665 | 0,94501 | 0,05894 | Oui |
| EqualizedOdds / intersection | Vote | 0,70244 | 0,84331 | 0,12654 | Non |

La moyenne des probabilités sous EqualizedOdds/intersection satisfait l'équité
et l'utilité au seuil global **0,492** : DI genre 0,97053, EO genre 0,03238.
Elle échoue toutefois au minimum AUC de 0,74. Aucun des huit points ne satisfait
l'ensemble des règles. Les autres seuils sont des diagnostics d'échec.
Les quatre runs sont `FINISHED`, avec 17 appels aux oracles chacun et environ
19 à 22 secondes d'ajustement. Le meilleur gap d'optimisation du run
EqualizedOdds/intersection vaut 0,05839 ; il ne justifie aucune affirmation de
convergence exacte ou de garantie sur les scores déterministes.

[Rapport complet](assets/constrained-development/report.json),
[définition préalable](assets/constrained-development/definition.json) et
[preuve d'exécution](assets/constrained-development/execution.json).
**133 tests réussis**, Ruff sans erreur, 93 fichiers Python formatés.
Les alias Home Credit sont vides sur les deux serveurs MLflow ; le test v2 reste
fermé. Les résultats sont archivés dans un dépôt DVC local isolé et poussés au
remote commun. Aucun calcul SHAP ni modèle enregistré n'est produit pour ces
points rejetés au screening.

## Traçabilité et vérification

Configuration dans `params.yaml`, section `constrained_development`. Le manifeste
et la configuration source du protocole v2 restent inchangés, vérifiés par SHA256.
Les memberships train/validation sont contrôlés avant ajustement ; les fichiers
X/y/sensitive test ne sont pas ouverts. MLflow conserve quatre runs dans
`credit-scoring-constrained-development`, avec métriques des deux scores,
audits complets, coûts, paramètres, nombre d'oracles, gap d'optimisation et hashes
des entrées exactes. Le registre de holdout v2 reste absent.

```powershell
.\creditS\Scripts\Activate.ps1
python -m pytest -q
python -m ruff check .
Get-Content runtime/constrained-development/report.json
```

Le script `python -m src.models.constrained_study` refuse d'écraser un résultat
existant. Les rapports permettent d'inspecter l'étude sans relancer la recherche.
