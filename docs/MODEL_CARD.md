# Model Card — état après la phase 5

Projet pédagogique Home Credit Default Risk, sample_frac=0.1. Aucun modèle
n'est promu ni destiné à prendre des décisions de crédit réelles à ce stade.
Le dataset Kaggle et les seuils choisis ne démontrent pas une représentativité
pour une population réelle. La carte sera complétée pendant la phase 12.

## Données et modèle

Train 14 760, validation 4 920, test réservé 4 920, production isolée 6 151.
Version brute DVC : `6d6f2659068bf71e19aece18bd8736eb.dir`.
261 features ; encodage, imputation et statistiques ajustés sur le train.
CODE_GENDER, DAYS_BIRTH, AGE_YEARS, AGE_GROUP et EMPLOYED_AGE_RATIO ne sont pas
des features. Les tables sensibles servent à l'audit et à l'étude des mitigations.
La repondération estime ses poids uniquement sur les intersections genre/âge
du train et utilise un seuil global à l'inférence, sans attribut sensible.

## Résultats provisoires, validation uniquement

| Métrique | LightGBM baseline | Repondération, puissance 1.5 |
|---|---:|---:|
| ROC-AUC | 0.7487 | 0.7311 |
| Seuil global de refus | 0.49 | 0.55 |
| Recall du défaut | 0.6146 | 0.5214 |
| Disparate impact acceptation, genre | 0.8901 | 0.9871 |
| Disparate impact acceptation, âge | 0.5996 | 0.8142 |
| Écart de TPR défaut, genre | 0.0722 | 0.0150 |
| Écart de TPR défaut, âge | 0.1974 | 0.0921 |

Positif = défaut prédit = refus. Taux de sélection = acceptation = 1 - prédiction.
Disparate impact = minimum/maximum du taux d'acceptation des groupes comparables.
Equal opportunity difference = maximum-minimum du TPR de refus parmi les défauts
réels. Equalized odds difference = maximum de l'écart TPR et de l'écart FPR.
Les métriques et effectifs détaillés par groupe sont dans
`reports/fairness/fairness_report.json`, conservé par DVC et MLflow.

Le candidat respecte DI >= 0.8 et écart TPR <= 0.10 sur genre et âge, avec une
perte d'AUC de 0.0175, inférieure à 0.02. Il ne respecte pas le minimum AUC
0.74 du futur quality gate. Cette étude de mitigation n'est pas une promotion.
Le biais est mesuré et contrôlé selon ces métriques sur cette validation ;
aucune absence générale de biais n'est affirmée.

Comparaisons limitées aux groupes ayant au moins 100 observations et 20 exemples
de chaque classe. F/M et les quatre tranches d'âge sont requis. XNA est affiché,
mais n'a qu'un client en validation et aucun défaut : son TPR est non estimable,
son équité n'est pas certifiée. La couverture genre des comparaisons est 4919/4920.
Les faibles effectifs de défauts, particulièrement chez les 60+, rendent les
écarts incertains ; les seuils franchis de peu ne garantissent pas leur stabilité.
Les paramètres, méthodes et seuils sont sélectionnés sur cette même validation :
une évaluation indépendante sur le test reste nécessaire.

## Explicabilité et proxys

SHAP TreeExplainer 0.49.1 explique le score brut du LightGBM, en log-odds du
défaut. Une contribution positive augmente le score de défaut. Les explications
reconstruisent la sortie du modèle à une erreur inférieure à 1e-6 ; il s'agit
d'associations du modèle, pas de causes ni d'une preuve d'équité. Les valeurs
imputées sont expliquées, et ne doivent pas être présentées comme des valeurs
déclarées par le client. Summary, bar et trois waterfalls sont archivés par run.
Les scores, notamment avec class_weight=balanced, ne sont pas des PD calibrées.

Un classifieur auxiliaire prédit le genre connu à partir des features autorisées
avec AUC validation 0.8734. Les coefficients standardisés et corrélations du
train sont dans `reports/fairness/proxy_report.json`. Ils signalent des proxys
potentiels, sans démontrer qu'une variable particulière cause une discrimination.

## Limites des comparateurs Fairlearn

ThresholdOptimizer est ajusté sur une réserve de calibration dans le train,
séparée des observations utilisées pour son estimateur. Ses seuils utilisent
les intersections genre/âge à l'inférence et ses décisions sont randomisées,
comme décrit par la [documentation Fairlearn 0.13](https://fairlearn.org/v0.13/user_guide/mitigation/postprocessing.html).
L'emploi du genre et de l'âge dans une décision réelle soulève une limite
juridique potentielle : l'admissibilité de ces seuils doit être étudiée pour
la juridiction et le contexte d'usage. Ce comparateur reste réservé à l'étude,
et ne satisfait pas la règle de production sans attribut sensible.
Les groupes absents ou insuffisamment représentés en calibration utilisent le
seuil global de secours, avec cette limite conservée dans la politique.

ExponentiatedGradient utilise une mixture randomisée de classifieurs entraînés
sous contrainte EqualizedOdds. Son score est une probabilité de refus de la
mixture, pas une probabilité calibrée de défaut. Son AUC ne doit donc pas être
interprétée comme exactement la même quantité que le score du modèle de base.
Les deux comparateurs sont évalués sur leurs décisions effectivement tirées,
avec seed=42 ; la randomisation introduit une variabilité supplémentaire.
Ils ne sont pas retenus dans cette exécution. En particulier, une solution
qui accepte presque tout échoue aux règles d'utilité : recall >= 0.50 et taux
d'acceptation compris entre 0.10 et 0.95.

## Travail restant

Quality gate et Registry, audit indépendant du test, API et supervision sont
les prochaines phases. Le cadre réglementaire détaillé, les mesures de contrôle
humain, la protection des données et la surveillance seront documentés avant
toute prétention d'utilisation réelle. Aucun déploiement réel n'est validé ici.
