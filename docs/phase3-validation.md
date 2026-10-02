# Phase 3 — nettoyage et features

## Contrat final

261 colonnes ordonnées : 122 numériques et 139 indicateurs catégoriels issus de
15 variables. Les sorties X/y/sensitive sont alignées par SK_ID_CURR sur train,
validation, test et production. feature_columns.json conserve l'ordre exact ;
feature_encoder.pkl et category_vocabulary.json conservent l'encodage appris.

Le choix utilisateur réserve strictement l'âge à l'audit : EMPLOYED_AGE_RATIO
est calculé et exporté dans sensitive_*.csv, puis exclu du modèle avec
CODE_GENDER, DAYS_BIRTH, AGE_YEARS et AGE_GROUP. TARGET et SK_ID_CURR ne sont
jamais des features. Le ratio âge/emploi reste disponible pour l'analyse.

clean_application est réutilisable et idempotent : la sentinelle 365243 devient
NaN et EMPLOYED_ANOMALY=1 ; AGE_GROUP utilise les intervalles [0,30), [30,45),
[45,60), [60,+inf). La réserve production et les splits de la phase 2 sont
inchangés : les cinq fichiers intermédiaires restent identiques par SHA-256.

## Ratios

| Variable | Calcul | Sortie |
|---|---|---|
| CREDIT_INCOME_RATIO | AMT_CREDIT / AMT_INCOME_TOTAL | modèle |
| ANNUITY_INCOME_RATIO | AMT_ANNUITY / AMT_INCOME_TOTAL | modèle |
| CREDIT_TERM | AMT_CREDIT / AMT_ANNUITY | modèle |
| EXT_SOURCE_MEAN | moyenne des EXT_SOURCE_1/2/3 disponibles | modèle |
| EMPLOYED_AGE_RATIO | DAYS_EMPLOYED / DAYS_BIRTH nettoyés | audit uniquement |

Un dénominateur nul/manquant produit NaN. La moyenne EXT_SOURCE reste manquante
si aucune des trois sources n'est disponible. Il n'y a pas de valeur infinie.

## Agrégations historiques

- bureau : nombre de crédits, part active, crédit total, dette totale, retard maximal.
- previous_application : nombre de demandes, taux de refus, montants moyens demandé et de crédit.
- installments_payments : nombre d'échéances, nombre de versements, retard moyen,
  part des échéances payées tard et écart moyen montant dû moins montant payé.

Les versements sont regroupés par (SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_VERSION,
NUM_INSTALMENT_NUMBER). Le montant dû est pris une fois (maximum observé), les
versements sont additionnés, et le retard est max(dernière date de versement -
date d'échéance, 0). Cela évite de doubler le montant dû lors d'un paiement fractionné.
L'écart peut être négatif si la somme versée dépasse le montant dû.

Les lignes postérieures au jour d'application sont exclues : DAYS_CREDIT <= 0,
DAYS_DECISION <= 0, et pour les paiements DAYS_INSTALMENT <= 0 ainsi que
DAYS_ENTRY_PAYMENT <= 0. Des dates manquantes ne prouvent pas la disponibilité
historique et sont exclues. La granularité journalière ne permet pas de vérifier
l'ordre exact des opérations au cours d'une même journée.

Un historique absent donne un compteur zéro et des statistiques manquantes.
Une somme entièrement manquante reste manquante, sans fausse dette zéro.
Les moyennes sont calculées à partir de sommes et de comptes valides, sans
moyenne non pondérée des moyennes de chunks.

## Fuites de données et mémoire

Ces agrégations sont des caractéristiques propres à chaque client ; elles
n'apprennent aucune statistique sur la population validation/test/production.
Le vocabulaire OneHotEncoder est ajusté uniquement sur train, avec catégories
inconnues ignorées. Un token constant représente les catégories manquantes.
La médiane d'imputation et la standardisation restent ajustées uniquement sur
train dans le pipeline du modèle. Aucun label de validation/test ne participe
au feature engineering ; les labels test ne sont pas utilisés pour choisir le modèle.

Les fonctions lisent uniquement les colonnes requises, par chunks de 100 000
lignes : int32 pour les identifiants, float32 pour les jours et catégories pour
les statuts. Les montants sont agrégés en float64 pour limiter la perte de précision,
puis les features exportées sont réduites en float32. Seuls les 30 751 identifiants
échantillonnés sont conservés. Les paiements sont répartis dans 64 partitions sur
disque puis regroupés par échéance ; les fichiers temporaires sont nettoyés.
Les paramètres chunksize/installment_partitions permettent d'adapter la mémoire.

Les NaN dus aux sources, aux dénominateurs invalides ou aux historiques absents
sont attendus et dénombrés dans feature_manifest.json. La vérification compare
ces nombres aux exports. Aucun inf n'est présent ; les sorties de l'imputation
train-fitted sont finies sur les quatre splits.

## Résultats exécutés

| Contrôle | Résultat |
|---|---|
| Train / validation / test / production | 14 760 / 4 920 / 4 920 / 6 151 |
| Contrat de features | 261 colonnes identiques ; X/y/audit alignés |
| Encoder rechargé | classe src.features.encoding.FeatureEncoder |
| Tests | 29 réussites, sans avertissement |
| Ruff | lint et format conformes |
| dvc repro | quatre stages exécutés |
| Réexécution sans cache d'exécution | 56 fichiers identiques par SHA-256 |
| Conservation des splits | 5 fichiers interim identiques à la phase 2 |
| Remote DVC | cache et storage synchronisés |
| EDA | 4 cellules exécutées sur le train uniquement |

La référence logistique fixe obtient AUC validation 0,721025 et PR-AUC 0,198446,
au seuil métier 0,09. L'AUC est inférieure à 0,734502 en phase 2 : davantage de
features ne garantit pas une amélioration pour un modèle linéaire. Aucun ajustement
sur le test ni promotion ne compense cette baisse. La comparaison de familles,
le réglage et MLflow seront réalisés en phase 4. L'audit d'équité reste à faire.

## Commandes Windows PowerShell

```powershell
.\creditS\Scripts\Activate.ps1
python -c "import sys; print(sys.prefix)"
make train
python -m scripts.verify_data_pipeline
make test
make lint
python -m dvc status --cloud
```

Les snapshots de reproductibilité sont ignorés dans .phase3-repro-check/.
Les CSV, le vocabulaire, l'encodeur, le modèle et les lots sont versionnés via
DVC ; Git conserve le code, les métadonnées DVC et les résultats agrégés de l'EDA.
