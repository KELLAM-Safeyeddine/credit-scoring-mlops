# Protocole prospectif Home Credit v2

Figé le **3 octobre 2026**, identifiant `home-credit-independent-v2`.
Ce protocole prépare une nouvelle campagne ; aucun modèle n'a encore été
entraîné et aucun résultat prédictif du nouveau test n'a été consulté.
Le précédent candidat reste rejeté : il passe la validation mais échoue sur
l'équité du test indépendant. [Historique et décision](qualification.md).

## Population et affectations

Les 61 502 clients des campagnes précédentes sont exclus de **tous** les
nouveaux splits, réserve de production comprise. Les exclusions proviennent
des quatre affectations de `data/interim` et de l'expansion à 20 % ; leurs
hashes d'identifiants sont conservés dans le manifeste.
Il reste 246 009 clients dans la table `application_train` de 307 511 lignes.
Le protocole en prélève 92 253, soit 30 % de la table source, avec seed 20261003.
Après réserve de production de 20 %, les données restantes sont stratifiées
60/20/20. La stratification utilise uniquement TARGET ; les CSV d'affectation
ne contiennent que `SK_ID_CURR`, sans cible ni attribut sensible.

| Split | Clients | Usage autorisé |
|---|---:|---|
| train | 44 281 | Fit des transformations, poids et modèles |
| val | 14 760 | Choix du seuil global et du candidat |
| test | 14 761 | Une évaluation du candidat figé |
| production | 18 451 | Réserve séparée pour monitoring ultérieur |

Les hashes et la disjonction réelle de tous les splits, y compris avec
l'historique, ont été vérifiés. Les features et leur vocabulaire seront
construits ensuite, avec fit uniquement sur train. Le nombre de colonnes du
futur modèle ne doit pas être supposé identique aux contrats des anciennes campagnes.

## Recherche préspécifiée

Six configurations distinctes : deux LightGBM et trois puissances de repondération
par âge seulement, **1,5 / 1,6 / 1,75**. Tous les audits continuent à vérifier
le genre et les quatre groupes d'âge. Les attributs sensibles restent exclus
des features et ne sont pas requis à l'inférence.

| Paramètre | Configuration A | Configuration B |
|---|---:|---:|
| n_estimators | 300 | 300 |
| num_leaves | 7 | 15 |
| learning_rate | 0,05 | 0,03 |
| min_child_samples | 80 | 100 |
| reg_lambda | 5 | 5 |

Class_weight=balanced, n_jobs=4 et random_state=20261003. Seuil unique, choisi
sur validation selon le coût métier configurable sous contraintes d'équité
et d'utilité, grille de 981 seuils de 0,01 à 0,99. Aucun nouveau profil de
features, puissance ou hyperparamètre ne sera ajouté après lecture des résultats.

Le candidat est sélectionné parmi les passes du gate par AUC validation
maximale ; à égalité exacte, recall maximal puis ordre des essais. Avant
lecture du test : bundle et seuil figés, artefacts SHAP présents et vérifiés,
provenance DVC enregistrée dans MLflow. Si aucun candidat ne passe la validation,
le test reste fermé. Si le candidat échoue sur test, aucun autre candidat ni
seuil n'est essayé sur ce même test et aucune promotion n'a lieu.

Les seuils restent inchangés : AUC ≥ 0,74 ; DI ≥ 0,8 et écart EO ≤ 0,10 pour
genre et âge ; supports ≥ 100 dossiers et ≥ 20 de chaque classe par groupe ;
recall ≥ 0,50, acceptation entre 0,10 et 0,95. Le gate exige aussi SHAP et une
inférence déterministe sans groupe sensible. La comparaison au champion éventuel
requiert les mêmes données de validation. Le Registry n'a actuellement aucun champion.

## Fichiers et contrôles

[Manifeste figé](assets/protocol-v2/protocol.json) et
[configuration prospective](assets/protocol-v2/params.yaml).
Le snapshot isolé est dans `runtime/protocol-v2` :

- memberships : `6e2530393a6960d3771bd4d72be7d06c.dir` ;
- protocol.json : `04571a1e6786262ab5dd2de2991c7255` ;
- params.yaml : `6f41af0f9bfe1e1a6966cd916f887884`.

Sept fichiers ont été poussés au remote DVC local. Les identifiants individuels
restent dans le snapshot DVC ; Git conserve les descripteurs et le manifeste.
Le script `src.data.freeze_protocol` refuse d'écraser le dossier existant.
Le runner vérifie le hash de la configuration et les affectations train/val
avant fit ; l'évaluateur vérifie celle du test. Modifier les règles ou remplacer
les IDs provoque un refus. Les contrôles locaux évitent les erreurs de protocole ;
ils ne constituent pas une protection contre un opérateur qui modifierait les fichiers.

Le registre du nouveau test sera
`runtime/qualification-holdouts/home-credit-independent-v2.json`.
Il n'existe pas encore. Le registre du test historique est conservé intact,
avec son candidat rejeté et ses résultats ; il n'est ni effacé ni réinitialisé.
Les prochains runs MLflow conserveront l'identifiant et le manifeste du protocole.

Vérification sur le workspace actuel :

```powershell
.\creditS\Scripts\Activate.ps1
$protocolPython = (Resolve-Path .\creditS\Scripts\python.exe).Path
Push-Location runtime/protocol-v2
& $protocolPython -m dvc status
Pop-Location
python -m pytest -q
python -m ruff check .
```

Sur un clone avec accès au même remote, créer le dossier `runtime/protocol-v2`,
y copier les trois descripteurs de `docs/assets/protocol-v2`, initialiser un
dépôt DVC sans SCM et configurer le remote local, puis faire `dvc pull` avec
le Python de `creditS`. Le snapshot restaure le manifeste, la configuration
et les quatre fichiers d'affectation ; il ne restaure pas une base MLflow.

Preuves exécutées : création des splits, vérification des SHA-256 et des
exclusions, résolution du nouveau registre, DVC push/status, **122 tests réussis
en 30,40 secondes**, 27 avertissements de dépendances, Ruff sans erreur et
89 fichiers Python formatés. Aucun fit, rapport de performance ou promotion
n'a été exécuté pour ce protocole.

Cette indépendance concerne les affectations de clients dans le même dataset
Kaggle. Les choix méthodologiques bénéficient des campagnes historiques ; ce
protocole ne fournit pas une validation temporelle, une validation bancaire
externe ou une garantie de généralisation. L'étape suivante préparera les
features à partir des affectations figées et lancera les six essais, après
validation utilisateur du protocole.
