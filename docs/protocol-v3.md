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

La prochaine étape, après validation de l'utilisateur, matérialisera les features
et exécutera uniquement les deux fits préspécifiés. Elle laissera le test fermé.
