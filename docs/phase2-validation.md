# Phase 2 — données et DVC

J'ai extrait et validé les données, puis mis en place leur suivi DVC et un premier pipeline reproductible.

Ce compte rendu décrit les vérifications de cette phase au moment de son exécution. Le [bilan définitif](project-completion.md) présente le champion final et les dernières validations.

L'archive de la racine a été extraite vers data/raw. Elle contient huit tables
opérationnelles, le dictionnaire HomeCredit_columns_description.csv et le fichier
optionnel sample_submission.csv. Les neuf CSV requis sont vérifiés avant extraction.
Le manifeste inclut le SHA-256 de l'archive. Le CRC ZIP est contrôlé pendant la
lecture ; les exécutions suivantes vérifient le hash source et les tailles, sans
réécrire une extraction complète. DVC conserve ensuite les hashes des CSV.

## Pipeline exécutable de cette phase

prepare → featurize → train → simulate_production

- Pandera 0.26.1 valide application_train avant tout échantillonnage : identifiants
  uniques et non nuls, TARGET dans {0,1}, revenu positif, genre M/F/XNA, naissance négative.
- sample_frac=0.1, random_state=42. Une réserve production indépendante de 20 %
  est prélevée d'abord ; les 80 % restants sont divisés en 60/20/20, stratifiés.
- SK_ID_CURR est conservé comme index. La sentinelle d'emploi est remplacée par
  NaN et un indicateur. L'âge et ses groupes sont exportés uniquement pour l'audit.
- Features numériques de référence : 104 colonnes, identiques sur les quatre splits.
  TARGET, SK_ID_CURR, CODE_GENDER, DAYS_BIRTH, AGE_YEARS et AGE_GROUP sont exclus.
- Imputation médiane et standardisation sont ajustées uniquement sur le train.
  Le seuil minimise le coût de validation (défaut accepté : 10 ; bon dossier refusé : 1).
  Le test reste intact ; application_test n'est jamais utilisé pour entraîner ou évaluer.
- Régression logistique locale, modèle DVC models/baseline.pkl ; aucune promotion.
- Huit lots normaux, chacun avec features, prédictions, audit et labels séparés.
  Le manifeste indique la disponibilité des labels avec deux lots de retard.

Les ratios métier et agrégations seront ajoutés en phase 3. Les switches des
agrégations sont déjà configurés mais ne sont pas encore opérationnels. La
comparaison des quatre familles et MLflow arriveront en phase 4, l'audit d'équité
et le gate dans leurs phases dédiées. Les injections de drift seront ajoutées en
phase 10 ; cette phase ne prétend pas démontrer les alertes Evidently.

## Résultats exécutés

| Vérification | Résultat |
|---|---|
| Validation Pandera complète | 307 511 lignes ; 8,0729 % de défauts |
| Extraction répétée | ignorée, fichiers conservés |
| Train / validation / test | 14 760 / 4 920 / 4 920 |
| Réserve production | 6 151 identifiants distincts |
| AUC validation de référence | 0,7345016877 |
| PR-AUC (average precision) | 0,2150567035 |
| Seuil de défaut | 0,09 |
| dvc repro | quatre stages exécutés sans erreur |
| Régénération | 54 fichiers identiques par SHA-256 |
| Isolation | aucun chevauchement d'identifiants entre les quatre splits |
| Contrats | X/y/audit alignés ; colonnes identiques, y séparé |
| pytest | 19 tests réussis |
| Ruff | lint et format conformes |

Pour tester une véritable régénération, data/interim et data/processed ont été
déplacés vers .phase2-repro-check/ (ignoré), puis dvc repro --force --no-run-cache
a exécuté les commandes. Les empreintes des 54 fichiers générés ont été comparées
à celles du premier run. Les dossiers sauvegardés sont conservés pour inspection.

## Stockage

Remote DVC par défaut : storage, dossier ../dvc-storage à côté du projet.
Données brutes : 2 684 262 587 octets, 11 fichiers (10 CSV + manifeste).
Hash DVC du répertoire brut : 6d6f2659068bf71e19aece18bd8736eb.dir.
Le modèle conserve cette référence de provenance. Les artefacts intermédiaires,
features, lots, modèle et métriques sont également mis en cache DVC et envoyés au remote.
Aucun CSV n'est versionné dans Git ; seuls les métadonnées DVC et le code le sont.

## Commandes Windows PowerShell

```powershell
.\creditS\Scripts\Activate.ps1
python -c "import sys; print(sys.prefix)"
make data
python -m dvc repro
python -m scripts.verify_data_pipeline
python -m dvc push
python -m dvc status
python -m dvc status --cloud
make test
make lint
```

Les stages DVC utilisent explicitement creditS/Scripts/python.exe : ils fonctionnent
même si le shell d'exécution ne conserve pas l'activation. Sur Linux/macOS, activer
creditS/bin/activate puis adapter runtime.python dans params.yaml vers creditS/bin/python.
Le hash des datasets reste indépendant de ce chemin ; changer le runtime invalide
volontairement les stages dans dvc.lock.

La restauration se fait avec python -m dvc pull. Le remote local est propre à cette
machine ; pour collaborer, il faudra un remote partagé, éventuellement MinIO.

Vérifications finales : dvc push a envoyé les sorties ; dvc status --cloud confirme
« Cache and remote storage are in sync ». dvc pull et make train sont à jour.
Les tests finaux confirment 19 réussites ; Ruff check et format passent.
