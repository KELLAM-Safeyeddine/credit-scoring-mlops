# Qualification prospective sur de nouveaux clients : protocole v3

Figé le **3 octobre 2026**, `home-credit-independent-v3`. Il suit le rejet de
l'ensemble au [test v2](ensemble-qualification.md) : AUC conforme, mais DI âge
0,79575 et écart equal opportunity âge 0,11249 hors limites. Le registre v2
évalué reste intact. Ce nouveau protocole désigne une réserve de clients ;
il ne désigne pas le modèle v3 accepté de la démonstration synthétique.

## Affectations et indépendance

Les **153 755 clients** des campagnes précédentes sont exclus de tous les splits,
y compris les anciennes réserves de production. Les identifiants proviennent
de `data/interim`, `runtime/qualification-expanded-20/interim` et
`runtime/protocol-v2/memberships`. Les quatre affectations de chaque campagne
sont contrôlées ; les exclusions communes sont comptées une seule fois.

Le pool restant compte 153 756 clients. Le nouveau protocole en affecte
**123 004**, soit 40 % des 307 511 lignes sources, seed **20261004**.
La réserve de production prend 20 % ; le reste est stratifié 60/20/20.
Les exports d'affectation ne contiennent que `SK_ID_CURR`. TARGET sert seulement
à la stratification ; aucune performance ou distribution d'audit du nouveau
test n'a été consultée. Les features ne sont pas encore matérialisées.

| Split | Clients | Usage |
|---|---:|---|
| train | 59 041 | Fit des transformations, contraintes et modèles |
| val | 19 681 | Choix du seuil global et d'un candidat |
| test | 19 681 | Une évaluation du candidat figé |
| production | 24 601 | Monitoring séparé |

Il reste **30 752 clients non affectés**. Les hashes des affectations et la
disjonction avec l'historique sont vérifiés. Le protocole ne réinitialise pas
un ancien test : son registre propre, encore absent, est
`runtime/qualification-holdouts/home-credit-independent-v3.json`.

## Deux entraînements préspécifiés

Fairlearn 0.13.0 `ExponentiatedGradient`, `EqualizedOdds` sur les intersections
genre–âge, objectif à coûts FP=1/11 et FN=10/11, `eps=0.005`,
`difference_bound=0.01`, `max_iter=15`. Ce dernier limite les tours externes,
pas le nombre total d'appels aux oracles. Les oracles LightGBM 4.6.0 utilisent
`class_weight=None`, quatre threads et les poids de la réduction.

| Essai | Arbres | Feuilles | Learning rate | Min child samples | Reg lambda |
|---|---:|---:|---:|---:|---:|
| 1 | 300 | 7 | 0,05 | 100 | 5 |
| 2 | 500 | 7 | 0,03 | 120 | 10 |

Seule la moyenne pondérée des probabilités est admissible à la sélection.
Le vote binaire peut être rapporté comme diagnostic, sans constituer un candidat
supplémentaire. Le budget reste **deux fits** ; aucun hyperparamètre ou profil
de features ne sera ajouté après connaissance des résultats de cette campagne.
Imputation et vocabulaire sont ajustés uniquement sur train. Les attributs
sensibles, identifiants et ratio emploi/âge restent exclus des features.

Les limites du gate de production restent **AUC ≥ 0,74, DI ≥ 0,8 et écart
equal opportunity ≤ 0,10**, pour genre et âge. Pour éviter une sélection trop
proche des limites, cette campagne impose en plus des marges sur validation :
**AUC ≥ 0,745, DI ≥ 0,85 et écart EO ≤ 0,075**, avec les mêmes contraintes
de supports, recall et acceptation. Ces marges sont des critères préalables
de sélection, sans assouplissement du gate ni garantie de réussite sur test.

Le seuil global est choisi sur validation selon le coût métier sous ces
contraintes, grille de 981 valeurs entre 0,01 et 0,99. Parmi les candidats
admissibles : AUC validation maximale, puis recall maximal et ordre des essais.
Sans candidat admissible, aucun test. Avec un candidat, sérialisation, SHAP,
seuil et provenance sont figés avant l'évaluation unique. Un échec indépendant
interdit une promotion ou l'essai d'un autre candidat sur ce test.

Les scores demeurent non calibrés. Les garanties d'une politique aléatoire
Fairlearn ne s'étendent pas automatiquement au score déterministe ; le biais
doit être mesuré et contrôlé par les audits effectivement exécutés.
Les choix de méthode bénéficient des campagnes antérieures : l'indépendance
porte sur les clients, dans le même snapshot Kaggle, sans validation temporelle
ou bancaire externe. Multiplier des réserves n'efface pas ces limites.

## Fichiers et vérification de la préparation

[Configuration de préparation](../configs/protocol-v3.yaml),
[configuration effective figée](assets/protocol-v3/params.yaml),
[manifeste](assets/protocol-v3/protocol.json),
[preuve d'exécution](assets/protocol-v3/execution.json).

`src.data.freeze_protocol --params configs/protocol-v3.yaml` a été exécuté dans
`creditS`. Il refuse d'écraser un protocole existant. Les plans personnalisés
ont un budget validé avant toute écriture. Le dépôt DVC isolé
`runtime/protocol-v3` conserve les affectations et le plan : **8 fichiers poussés**
au remote local commun, statut propre. Les CSV d'identifiants ne sont pas dans Git.
Les empreintes du candidat v2 et son registre évalué restent conformes aux
preuves archivées. Aucun entraînement, aucune préparation des features ni
évaluation du nouveau test n'ont eu lieu à cette étape.
La suite complète passe : **149 tests réussis en 85,12 secondes**, 41
avertissements de dépendances, Ruff conforme et 97 fichiers Python formatés.
Les cinq nouveaux tests couvrent le gel réel d'un plan sur fixtures, le refus
d'écrasement et le rejet de budgets invalides avant toute écriture.

```powershell
.\creditS\Scripts\Activate.ps1
python -m pytest -q
python -m ruff check .
Get-Content runtime/protocol-v3/execution.json
$protocolPython = (Resolve-Path .\creditS\Scripts\python.exe).Path
Push-Location runtime/protocol-v3
& $protocolPython -m dvc status
Pop-Location
```

À la clôture de la définition, l'étape suivante prévue était la matérialisation
des features puis les deux configurations préspécifiées, après validation de
l'utilisateur. Son exécution est consignée ci-dessous ; le test reste fermé.

## Matérialisation après validation de l'utilisateur

La préparation a été exécutée sur les affectations figées. Pandera valide la
source et le descripteur raw DVC est contrôlé avant export. Les agrégations
bureau, previous_application et installments_payments portent sur les 123 004
clients, avec cutoff au jour 0 et traitement des historiques par chunks.
L'encodeur apprend le vocabulaire sur train seulement. Le contrat v3 compte
**264 features**, identiques entre les quatre splits ; il est distinct du
contrat v2 de 263 colonnes. L'imputation reste ajustée sur train au moment du fit.

Les fichiers de test sont matérialisés pour conserver un contrat reproductible,
sans calcul de scores ni audit prédictif sur le test. Genre, âge et ratio
emploi/âge sont exportés séparément pour l'audit et exclus des features.

- Interim : `cebb8a3b1eea9bf44cbe11b058542c17.dir`, quatre CSV, 69 354 699 octets.
- Processed : `fdec4513d9be07de9abdfe5a2d89130c.dir`, seize fichiers,
  145 819 983 octets.

Les snapshots sont suivis par DVC dans `runtime/protocol-v3` et poussés au
remote commun : 22 objets supplémentaires, dont les deux manifestes de répertoire.
La preuve de préparation et les descripteurs sont archivés avec le protocole.

Le runner applique les marges uniquement à la sélection : il ne réécrit pas
les limites du quality gate dans la configuration figée. Il vérifie que le
plan utilisé est exactement celui du protocole, refuse un budget différent,
et interdit au vote binaire diagnostique d'être retenu. Le choix final suit
AUC, recall puis ordre stable des essais. Aucun résultat de test ne participe
à ce choix. Les nouveaux tests couvrent ces invariants.

Validation du code : **152 tests réussis en 104,48 secondes**, 41 avertissements
de dépendances ; Ruff conforme et 97 fichiers Python formatés.

```powershell
.\creditS\Scripts\Activate.ps1
python -m pytest -q
python -m ruff check .
Get-Content runtime/protocol-v3/preparation_execution.json
```

Les commandes de première exécution utilisées sont :

```powershell
.\creditS\Scripts\Activate.ps1
python -m src.data.prepare_protocol --params runtime/protocol-v3/params.yaml
python -m src.features.build_features --params runtime/protocol-v3/params.yaml
python -m src.models.constrained_study --protocol runtime/protocol-v3/params.yaml --output runtime/protocol-v3/constrained-study
```

Le runner refuse d'écraser une étude existante ; ces commandes sont documentées
pour la reproductibilité et ne demandent pas de refaire la campagne terminée.

## Résultats des deux configurations

| Essai / score | AUC validation | DI âge | EO âge | Sélection v3 |
|---|---:|---:|---:|---|
| 300 arbres / probabilités | 0,741046 | 0,922616 | 0,063670 | Échec marge AUC |
| 500 arbres / probabilités | 0,740729 | 0,921256 | 0,057848 | Échec marge AUC |
| 300 arbres / vote | 0,717971 | 0,863622 | 0,050396 | Diagnostic non admissible |
| 500 arbres / vote | 0,721588 | 0,865488 | 0,073997 | Diagnostic non admissible |

Les deux scores probabilistes respectent les **règles numériques de production**
sur validation, dont AUC ≥ 0,74. Ils ne respectent pas la **marge de sélection
préspécifiée AUC ≥ 0,745** : ce résultat n'est pas présenté comme un échec
du minimum AUC du quality gate. Aucune règle n'est modifiée après observation
des résultats. **Aucun candidat n'est retenu**, aucun SHAP de qualification
supplémentaire n'est produit, aucun Registry n'est modifié et le test v3
reste fermé. Le statut final est `no_eligible_deterministic_score`.

Le premier score probabiliste utilise le seuil 0,494 : DI genre 0,997318,
EO genre 0,013395, recall 0,679043 et acceptation 0,657741.
Le second utilise 0,507 : DI genre 0,994150, EO genre 0,023603,
recall 0,668345 et acceptation 0,669072. Les audits vérifient aussi les supports.
Le biais est mesuré selon ces métriques ; ces résultats de validation
ne prouvent pas une qualification indépendante ni un usage bancaire.

Les deux runs exploités sont `FINISHED`, dans MLflow 3.10.1, expérience
`credit-scoring-constrained-development` :

- 300 arbres : `1f589efd597d44e7a4cd3f918a347908`, 44 appels aux oracles,
  environ 298,48 secondes, best_gap 0,01133.
- 500 arbres : `11a588a834e3474e8697fe39c2fc0363`, 44 appels aux oracles,
  environ 168,09 secondes, best_gap 0,01152.

Les best_gap restent supérieurs à `eps=0.005` au terme du budget externe
de quinze tours : cette étude ne prétend pas avoir atteint la précision
d'optimisation demandée. Le budget n'est pas augmenté après les résultats.
Le lock DVC racine loggé est un snapshot historique ; `processed.dvc` et
les SHA256 des entrées train/val identifient les données v3 effectivement utilisées.

### Incident numérique et reprise documentée

L'exécution initiale du second fit, run `8c0617f7198b4ebea66ced05962ad3d9`,
a terminé l'entraînement mais échoué lors du calcul des métriques, car une
moyenne pondérée pouvait produire une valeur supérieure à 1 par arrondi.
Le run demeure `FAILED` dans MLflow. La correction borne seulement la moyenne
finale à [0,1] dans la tolérance 10⁻¹² ; les probabilités invalides d'un oracle
sont toujours rejetées. Le test de régression reproduit exactement
`1.0000000000000002` avant correction et vérifie également ce refus.

Il y a donc **deux configurations distinctes et trois tentatives de fit**, dont
une reprise technique du second essai. Les paramètres, seed et SHA256 des
entrées sont identiques entre le run échoué et sa reprise ; aucun nouveau
profil ni candidat adaptatif n'est ajouté. Le premier résultat est conservé.
La reprise vérifie le statut `FAILED`, le nom du fit et ses paramètres avant
de repartir, et n'autorise qu'une reprise technique dans ce dossier.

Un checkpoint du second fit est désormais écrit immédiatement après
l'entraînement et avant l'évaluation : neuf oracles actifs, imputation et
contrat des colonnes, **4 259 919 octets**. Il permet de récupérer le résultat
d'ajustement après un incident ; **il n'est pas un candidat sélectionné**.
Les métriques ont été recalculées depuis ce checkpoint sur les 19 681 lignes
de validation et sont exactement identiques au rapport et aux logs MLflow.
Le binaire est conservé par DVC et exclu de Git.
Les preuves de préparation, de clôture et le répertoire d'étude sont ajoutés au
snapshot DVC : **12 objets supplémentaires poussés**, statut du dépôt isolé propre.

[Définition exécutée](assets/protocol-v3/study-definition.json),
[rapport complet](assets/protocol-v3/study-report.json),
[reprise technique](assets/protocol-v3/technical_retry.json),
[préparation](assets/protocol-v3/preparation_execution.json),
[preuve de clôture](assets/protocol-v3/study_execution.json).

Vérification finale après correction : **153 tests réussis en 48,54 secondes**,
41 avertissements de dépendances ; Ruff conforme. Le registre v3 reste absent,
le registre v2 évalué et ses empreintes restent inchangés, les alias Home Credit
locaux restent vides. Les services Docker de ce projet n'ont pas été arrêtés.
Cette campagne est close sans candidat ; le plan figé n'est pas prolongé
par des essais supplémentaires ni par un changement rétrospectif des marges.

```powershell
.\creditS\Scripts\Activate.ps1
Get-Content runtime/protocol-v3/constrained-study/report.json
Get-Content runtime/protocol-v3/study_execution.json
python -m pytest -q
python -m ruff check .
```
