# Développement : capacité des réductions Fairlearn

J'ai étudié une augmentation de capacité des oracles Fairlearn sur le développement v2, sans utiliser le test pour choisir le modèle.

Cette étude suit les quatre ajustements de l'[étude initiale](constrained-development.md).
Son meilleur point respectant l'équité avait AUC 0,73665, sous 0,74.
Le budget de la nouvelle étude est fixé **avant entraînement à deux ajustements**,
sur les mêmes 44 281 lignes de train et 14 760 lignes de validation v2.
Le test v2 reste fermé, et les six essais du protocole prospectif clos ne sont
ni modifiés ni présentés comme étendus par cette recherche.

## Plan de développement

Les deux essais utilisent Fairlearn 0.13.0 `ExponentiatedGradient`,
`EqualizedOdds` sur les intersections genre–âge, objectif `ErrorRate` à coûts
FP=1/11 et FN=10/11, `eps=0.01`, `difference_bound=0.02`, `max_iter=12`.
Le nombre d'itérations ne limite pas le nombre d'appels aux oracles.
Les oracles LightGBM 4.6.0 utilisent 300 arbres, quatre threads, seed 20261003,
`class_weight=None`, reg_lambda 5, et les poids de la réduction.

| Essai | Feuilles | Learning rate | Min child samples |
|---|---:|---:|---:|
| 1 | 7 | 0,05 | 80 |
| 2 | 15 | 0,03 | 100 |

La médiane d'imputation est ajustée sur le train seulement. Les contraintes
utilisent les attributs d'audit séparés ; les 263 features excluent genre, âge,
identifiants, TARGET et ratio emploi/âge. Les configurations et budgets sont
validés avant lecture des données : au maximum quatre essais dans le moteur,
un à vingt tours externes, un à quatre threads, noms uniques et paramètres
autorisés. Cette campagne en utilise exactement deux.

Pour chaque réduction, moyenne pondérée des probabilités et score de vote
binaire sont audités séparément, avec le même seuil global choisi sur la grille
de validation de 981 valeurs. **Les garanties de la politique aléatoire Fairlearn
ne s'appliquent pas automatiquement à ces scores déterministes.** Ils restent
non calibrés. Le screening conserve AUC ≥ 0,74, DI ≥ 0,8, EO ≤ 0,10,
supports suffisants, recall ≥ 0,5 et bornes d'acceptation inchangées.

Un point qui passe ce screening est sauvegardé avec les seuls oracles à poids
positif, l'imputation figée, l'ordre des colonnes, le seuil, les audits et la
provenance. Un contrôle après sérialisation vérifie les scores sur seize lignes
de validation, tolérance 10⁻¹². Ces instantanés sont **de développement** :
ni enregistrement Registry ni promotion, aucune nouvelle preuve SHAP ou
évaluation de test à cette étape.

La validation a déjà servi à plusieurs études. Un succès à ce stade demanderait
un plan de qualification ultérieur figé et une évaluation indépendante unique.
Il ne constitue pas un succès du gate complet de production.

## Résultats exécutés le 3 octobre 2026

| Essai / score | AUC validation | DI âge | EO âge | Screening |
|---|---:|---:|---:|---|
| 7 feuilles / probabilités | **0,74725** | **0,80968** | **0,09605** | **Passe** |
| 7 feuilles / vote | 0,71866 | 0,94258 | 0,07764 | Échec AUC |
| 15 feuilles / probabilités | 0,74980 | 0,77902 | 0,07106 | Aucun seuil faisable |
| 15 feuilles / vote | 0,71117 | 0,76915 | 0,10442 | Échec AUC et équité |

Un seul point satisfait simultanément les règles du screening. Son seuil global
est **0,417**, DI genre **0,92951**, EO genre **0,01090**, recall **0,73552**,
acceptation **0,58550**, précision **0,14318**, PR-AUC **0,24547**. Sa forte part
de refus et l'absence de calibration limitent l'interprétation métier.
Les marges âge vis-à-vis des limites DI/EO restent faibles ; il ne faut pas
en déduire qu'un test indépendant réussira. Les autres seuils non faisables
restent des diagnostics de moindre pénalité.

Run sélectionné : **`2cb35a1c377449df9fb7bcf46d5cb53b`**, sept oracles à poids
positif dans l'instantané **2 017 144 octets**, SHA256
`d93e7be08ab2a6b179b4cf32ae5b8e9434a1365e7685771c5892c7f49a6194bc`.
La sérialisation a été contrôlée sur seize lignes, puis l'ensemble des métriques
et audits recalculé depuis l'instantané sur les **14 760 lignes de validation** :
résultats exactement identiques. Aucun recalibrage ni recherche de seuil dans
ce contrôle. Deux runs `FINISHED`, 35 appels aux oracles chacun, temps
d'ajustement environ 66,38 et 92,06 secondes ; best_gap 0,00736 et 0,01303.

**139 tests réussis**, 27 avertissements de dépendances, 49,48 secondes ; Ruff
conforme et 93 fichiers Python formatés. Le registre d'évaluation du test v2
reste absent, les alias Home Credit restent vides sur 5000 et 5001.
Les rapports et l'instantané sont suivis dans un dépôt DVC isolé, poussés au
remote commun ; le binaire n'est pas versionné dans Git.

[Définition préalable](assets/constrained-capacity-development/definition.json),
[rapport complet](assets/constrained-capacity-development/report.json),
[preuve d'exécution](assets/constrained-capacity-development/execution.json).
J'ai ensuite produit les explications SHAP adaptées à la moyenne de sept modèles
LightGBM et figé un plan de qualification avant l'évaluation du test v2.

## Configuration et vérification

`params.yaml`, section `constrained_capacity_development`, conserve le plan.
La définition développée est écrite dans `definition.json` avant les fits.
Le manifeste/configuration source v2 et les memberships train/val sont vérifiés
par hash. MLflow conserve les paramètres, audits et métriques dans l'expérience
`credit-scoring-constrained-development`. Le lock racine capturé est historique
pour cette étude : le descripteur `processed.dvc` du dépôt v2 isolé constitue
la provenance DVC des entrées effectives.

```powershell
.\creditS\Scripts\Activate.ps1
python -m pytest -q
python -m ruff check .
Get-Content runtime/constrained-capacity-development/report.json
```

La commande de première exécution est
`python -m src.models.constrained_study --study constrained_capacity_development`.
Elle refuse d'écraser une campagne existante. Les instantanés pickle locaux
ne doivent être chargés que depuis les artefacts de confiance de ce projet.

Référence API : [Fairlearn 0.13, réductions et objectif à coûts](https://fairlearn.org/v0.13/user_guide/mitigation/reductions.html).
