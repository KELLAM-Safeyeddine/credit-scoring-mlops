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
