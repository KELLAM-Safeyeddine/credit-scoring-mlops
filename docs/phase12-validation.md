# Preuves — phase 12

J'ai documenté l'architecture, les résultats, les commandes de lancement et les limites du modèle.

Ce compte rendu décrit les vérifications de cette phase au moment de son exécution. Le [bilan définitif](project-completion.md) présente le champion final et les dernières validations.

Validation le 2 octobre 2026 sur Windows dans le venv racine `creditS`
Python 3.11.9. Les résultats Home Credit et synthétiques sont explicitement séparés.

- README remplacé par un état actuel : pitch, schéma Mermaid, bootstrap du venv,
  quickstart DVC/Docker, entraînement, gates, résultats, monitoring et limites.
- `docs/architecture.md` complété : flux, responsabilités, contrats, persistance,
  environnements séparés et comportement d'échec.
- Model Card complétée : hash DVC, splits, performance globale et par groupe,
  biais avant/après mitigation, supports, proxys, SHAP, contrats API, monitoring,
  réentraînement et conditions préalables à un usage réel.
- Graphique réel de l'étude de mitigation publié dans `docs/assets/`, identique
  à l'artefact DVC/MLflow et inspecté visuellement. La légende distingue la limite
  de perte d'AUC de l'étude du minimum AUC 0,74 du gate.
- Les tableaux reprennent les CSV et bundles de modèles enregistrés, dont les
  AUC par groupe et les effectifs ; aucune métrique indépendante de test inventée.
- Anciennes mentions futures dans les docs API/Airflow remplacées par le
  fonctionnement actuel et ses limites. Les preuves des phases précédentes
  restent des comptes rendus historiques.
- Sources réglementaires consultées : article 22 publié par la CNIL et AI Act
  consolidé au 27 juillet 2026 sur EUR-Lex. Dates, champ et seuils techniques
  ne sont pas présentés comme une certification de conformité.
- Revue du quickstart : suppression de la dépendance à l'existence d'un modèle
  `credit-scoring` pour lancer la démonstration synthétique sur une base neuve.
  Test sur un vrai Registry SQLite temporaire absent puis présent ; une panne
  technique reste propagée et n'est pas confondue avec un modèle absent.
- `make test` : **96 tests réussis**, 25 avertissements connus, en 43,75 secondes.
- `python -m dvc pull` réussi : données et artefacts à jour dans le remote local.
- `docker compose up -d` réussi ; `/live` répond **200**, `/health` **503**,
  conformément à l'absence de champion Home Credit. Les cinq services restent sains.
- `make demo` réexécuté avec succès : `demo__6b9708de0924` et DAG enfant
  `retraining__demo__6b9708de0924`. V1 accepté, drift Evidently, v2 rejeté pour
  genre, v3 repondéré accepté, API rechargée ; `champion=3`, `previous=1` dans
  le modèle synthétique séparé. Les alias Home Credit restent inchangés.
- Contrôle des liens locaux et blocs Markdown, Ruff, formatage et diff Git exécutés.

Les sorties détaillées de cette démonstration sont dans
`runtime/demo/6b9708de0924/validation.json` (dossier privé ignoré).
Le workflow GitHub Actions est prêt mais n'a pas été exécuté à distance,
faute de remote Git configuré. Le test indépendant Home Credit reste réservé.

```powershell
.\creditS\Scripts\Activate.ps1
make test
make lint
python -m ruff format --check .
python -m dvc pull
docker compose up -d
make demo
```

Les douze phases produisent un projet pédagogique documenté et une démonstration
complète de la chaîne logicielle. Elles ne fournissent pas un champion Home Credit
qualifié : AUC insuffisante après mitigation, absence de validation indépendante
finale et limites réglementaires/opérationnelles explicitement documentées.
