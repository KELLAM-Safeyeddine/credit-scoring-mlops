# Étude de développement v3 à quinze feuilles

J'ai mené une étude de développement séparée sur les données v3 pour augmenter la capacité du modèle sous contraintes Fairlearn.

Deux configurations supplémentaires, déclarées avant leurs fits, utilisent
uniquement le train et la validation v3 : 59 041 / 19 681 clients et 264 features.
La campagne prospective v3 initiale reste close sans candidat. Cette étude est
séparée, adaptative et réutilise sa validation ; elle ne constitue donc pas une
nouvelle qualification indépendante. Le test v3 n'est ni lu ni évalué, son
registre d'évaluation reste absent et le registre v2 évalué demeure inchangé.

## Plan et résultats exécutés

Le plan figure dans `params.yaml`, section `constrained_refinement_development`,
et dans la [définition archivée](assets/v3-refinement-development/definition.json).
Il contient exactement deux fits LightGBM à 15 feuilles, 300 puis 500 arbres,
sous Fairlearn ExponentiatedGradient / EqualizedOdds sur les intersections
genre–âge. Budget : vingt tours externes, `eps=0.005`, `difference_bound=0.01`.
Les coûts normalisés sont FN 10/11 et FP 1/11, sans `class_weight` supplémentaire.
Le seed, les données et les règles de production du parent v3 restent figés.
L'imputation est ajustée sur le train uniquement. Genre, âge et ratio emploi/âge
sont exclus des features ; les attributs sensibles servent à la mitigation et
à l'audit, sans être nécessaires à l'inférence du score déterministe.

La sélection conserve AUC ≥ 0,745, DI ≥ 0,85 et écart EO ≤ 0,075, en plus des
supports et contraintes métier. Seule la moyenne pondérée des probabilités
est éligible ; le vote dur demeure diagnostique. Les seuils sont choisis sur
validation selon les coûts et contraintes, sans consultation du test.

| Arbres / score | AUC validation | Seuil | DI âge | Écart EO âge | DI genre | Écart EO genre | Sélection |
|---|---:|---:|---:|---:|---:|---:|---|
| 300 / probabilités | 0,747646 | 0,424 | 0,838529 | 0,064687 | 0,983443 | 0,017971 | Refus : DI âge < 0,85 |
| 500 / probabilités | 0,751512 | 0,397 | 0,852035 | 0,065074 | 0,995486 | 0,027496 | Retenu pour développement |
| 300 / vote dur | 0,728151 | 0,474 | 0,937383 | 0,074411 | 0,999518 | 0,041831 | Mode non éligible, AUC insuffisante |
| 500 / vote dur | 0,729421 | 0,393 | 0,855215 | 0,071003 | 0,990004 | 0,005421 | Mode non éligible, AUC insuffisante |

Le modèle retenu a un recall de 0,787288, une PR-AUC de 0,237853 et un taux
d'acceptation de 0,536660. Le positif signifie **1 = défaut prédit = refus** ;
le DI compare les taux d'acceptation `1 - prédiction`, et l'écart EO compare
les TPR de refus parmi les défauts réels. Le biais est mesuré et contrôlé selon
ces métriques sur cette validation. Aucun quality gate complet ni promotion
n'est déclaré : les preuves SHAP et le test indépendant restent nécessaires.

Deux runs MLflow sont `FINISHED`, sans reprise :

- 300 arbres : `34ad29ceeda349b99cdfeb62ff9949c7`, 147,84 s, 59 appels aux oracles,
  best_gap 0,007271.
- 500 arbres : `1e1e19a2d91444d097ded1dc0770e1b8`, 213,21 s, 59 appels aux oracles,
  best_gap 0,009843.

Les best_gap dépassent `eps` au terme du budget ; la précision demandée n'est
pas atteinte. Les garanties de la politique randomisée Fairlearn ne sont pas
transférées à la moyenne déterministe : celle-ci est auditée explicitement.
Le score n'est pas calibré. Le DI âge reste proche de la marge ; sa robustesse
hors validation n'est pas établie. Aucun essai supplémentaire n'est ajouté.

## Preuves et vérification

L'instantané retenu contient neuf oracles actifs, chacun à 500 arbres,
l'imputation train et la liste ordonnée des colonnes. Taille : **8 191 693 octets**,
SHA256 `fe480d11ede2cd39f609ff57a56f0041f86a75aa0f5b26b90603540e64914e68`.
Ses métriques et audits recalculés après rechargement sur les 19 681 lignes
de validation sont exactement identiques au rapport et aux logs MLflow.
Les checkpoints et l'instantané sont conservés par DVC, exclus de Git.
Le dépôt DVC isolé `runtime/v3-refinement-development` archive aussi le plan,
les rapports, le lock historique et les copies du protocole source : dix objets
supplémentaires poussés vers `../dvc-storage`, statut propre après le push.
Les alias Home Credit restent vides ; aucun déploiement réel n'est effectué.

[Rapport complet](assets/v3-refinement-development/report.json),
[preuve d'exécution](assets/v3-refinement-development/execution.json).
Validation logicielle : **154 tests réussis en 38,77 s**, 41 avertissements de
dépendances, Ruff et format conformes. Le test ajouté vérifie qu'un plan de
développement séparé ne modifie pas le parent et exige son statut explicite.

```powershell
.\creditS\Scripts\Activate.ps1
Get-Content runtime/v3-refinement-development/report.json
Get-Content runtime/v3-refinement-development/execution.json
python -m pytest -q
python -m ruff check .
python -m dvc status
```

J'ai ensuite empaqueté l'ensemble, produit les preuves SHAP et figé un plan
d'évaluation unique avant l'ouverture du test v3.
