# Preuves — phase 9

J'ai vérifié les images Docker, les healthchecks et le démarrage de la stack Compose.

Ce compte rendu décrit les vérifications de cette phase au moment de son exécution. Le [bilan définitif](project-completion.md) présente le champion final et les dernières validations.

Validation exécutée sur Windows avec `creditS` Python 3.11.9, Docker Engine
29.4.2 et Docker Compose 5.1.3, conteneurs Linux.

- `docker compose config --quiet` réussi.
- Image `credit-scoring-api:0.1.0` construite par le Dockerfile multi-stage.
  Base Python vérifiée : `sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317`.
- `pip check` réussi pendant le build et dans le conteneur d'inférence.
- `make up` exécuté : API, MLflow, PostgreSQL, webserver et scheduler Airflow
  démarrés et **healthy**. Les UIs sur 5001 et 8081 répondent HTTP 200.
- Test Docker réel isolé : MLflow HTTP dans un conteneur séparé, modèle LightGBM
  synthétique, artefacts téléchargés depuis le serveur par l'API sans montage des
  fichiers du Registry dans le conteneur API.
- `/health`, `/predict`, `/explain` et `/model-info` fonctionnent avec ce champion
  synthétique ; prédiction et explication répondent 200 avec le même score et la
  même version 1 ; trois raisons SHAP et attribut sensible rejeté à 422.
- UID 10001, Airflow absent, venv local absent et aucun `.env` copié dans l'image.
  Racine en lecture seule, tmpfs, capabilities supprimées et volume de logs actifs.
- Deux événements JSONL écrits ; l'identifiant original du client est absent.
- Nettoyage des conteneurs, réseau et volumes temporaires du test effectué.
- `make test` : **84 tests réussis**, 24 avertissements de bibliothèques connus.
  Ruff et vérification du formatage réussis.

Le test initial a exposé une fermeture transitoire de connexion avant le démarrage
d'Uvicorn. Le vérificateur attend désormais aussi ces erreurs de transport, avec
timeout, et conserve les logs d'échec avant le nettoyage. Le test final HTTP
complet a réussi. Rapport privé : `runtime/packaging/validation.json`.

Le Registry **Home Credit reste inchangé** : versions 1–4 rejetées, alias vides.
`/live` de l'API principale répond 200 ; `/health` répond 503 conformément à
l'absence de champion accepté. Le health 200 du test isolé ne constitue pas une
qualification du modèle Home Credit.

```powershell
.\creditS\Scripts\Activate.ps1
make up
python scripts/verify_packaging.py
make test
make lint
python -m ruff format --check .
```

La surveillance Evidently complète commence en phase 10 après validation.
