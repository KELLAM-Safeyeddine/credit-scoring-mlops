# Packaging Docker — phase 9

J'ai empaqueté l'API et Airflow dans des images distinctes, avec une stack Compose et des volumes persistants.

## Lancer et vérifier sous Windows

```powershell
.\creditS\Scripts\Activate.ps1
make up
docker compose ps
curl.exe -i http://localhost:8000/live
curl.exe -i http://localhost:8000/health
python scripts/verify_packaging.py
```

`make up` prépare `.env` sans remplacer les secrets existants, construit les
images et attend les healthchecks. Après cette préparation, `docker compose up -d`
peut aussi relancer la stack. Docker Desktop doit utiliser des conteneurs Linux.

| Service | Accès depuis Windows | Persistance |
|---|---|---|
| FastAPI | http://localhost:8000/docs | volume `api_logs` |
| MLflow | http://localhost:5001 | `runtime/mlflow`, `runtime/mlartifacts` |
| Airflow webserver | http://localhost:8081 | PostgreSQL et volume `airflow_logs` |
| Airflow scheduler | réseau Compose interne | PostgreSQL et volume `airflow_logs` |
| PostgreSQL 14.17 | réseau Compose interne | volume `postgres_data` |

Tous partagent le réseau Compose du projet `credit-scoring-mlops`. Les services
HTTP sont publiés sur `127.0.0.1`. Aucun service d'un autre projet n'est modifié.
Les ports sont configurés dans `.env`. Le Registry Docker est indépendant du
Registry local sur 5000, initialisé une seule fois par une sauvegarde SQLite.
Les volumes et bases existants sont conservés lors d'un redémarrage ou d'un `make down`.
MinIO et une UI Evidently sont optionnels et ne sont pas ajoutés à cette stack.

## Image API

`api/Dockerfile` utilise deux étapes sur Python 3.11.9 slim Bookworm, avec le
digest de base effectivement téléchargé. La première installe les dépendances
épinglées dans son propre venv `/opt/creditS` ; la seconde copie ce venv et le
code. **Le venv `creditS/` du poste n'est jamais copié.** `.env`, l'archive Kaggle,
les données et les artefacts locaux sont exclus du contexte par `.dockerignore`.

L'utilisateur `credit` a l'UID/GID 10001. `libgomp1` permet l'exécution LightGBM.
L'API n'installe pas Airflow. Ses dépendances directes sont dans
`api/requirements.txt`, avec des contraintes transitives dans
`api/requirements-lock.txt`, cohérentes avec le worker ML déjà validé. Les
contraintes inutilisées n'installent pas de packages supplémentaires.

Compose utilise un système de fichiers racine en lecture seule, un `/tmp`
temporaire pour les caches et téléchargements MLflow, aucun privilège Linux
additionnel et un volume persistant pour les événements JSONL. Les caches
Matplotlib et Numba sont dirigés vers `/tmp`. Un seul worker Uvicorn écrit les logs.
L'image contient son propre `HEALTHCHECK`, et l'API dépend du healthcheck MLflow.
Airflow dépend de son initialisation et PostgreSQL.

Le cache optionnel `.docker-wheels/` accélère les constructions. Si `.complete`
est présent, les installations utilisent les wheels sans réseau. Sans ce cache,
pip télécharge les mêmes versions épinglées. Le cache est monté en lecture seule
pendant la construction et n'est pas copié dans l'image finale. Une construction
nécessite aussi l'accès à l'image de base et aux paquets système Debian.

## Disponibilité et modèle

`/live` répond 200 si le serveur traite les requêtes. Le healthcheck Docker utilise
cette disponibilité du processus. `/health` exprime la disponibilité du scoring :
200 avec un champion vérifié, 503 sans champion accepté.

**J'ai désormais déployé le champion Home Credit version 6 dans le Registry Docker.**
J'ai vérifié la santé de la stack et les routes de scoring avec des réponses 200.
Sur une installation neuve, le scoring reste à 503 tant qu'un champion valide
n'est pas chargé ; je n'ai pas contourné le gate pour obtenir une réponse 200.
Après une promotion valide, utiliser le rechargement décrit dans `docs/api.md`.
Une mise à jour du code nécessite une reconstruction ; les modèles sont chargés
depuis le Registry, sans être intégrés à l'image.

## Validation reproductible

`scripts/verify_packaging.py`, lancé avec le Python local de `creditS`, vérifie :

- un serveur MLflow HTTP séparé, créé uniquement pour le test ;
- un modèle LightGBM synthétique enregistré et accepté dans ce Registry isolé ;
- le téléchargement des artefacts par une API Docker distincte ;
- health, prédiction et explication à 200, score/version cohérents et trois raisons SHAP ;
- rejet à 422 d'un attribut sensible et journalisation sans identifiant original ;
- UID 10001, absence d'Airflow, du venv local et d'un fichier `.env` dans l'image ;
- `pip check`, les cinq healthchecks Compose et l'accès aux UIs MLflow/Airflow.

Les requêtes synthétiques ne vont pas dans le Registry ou les logs Home Credit.
Les conteneurs, le réseau et les volumes temporaires portent des noms uniques et
sont supprimés à la fin du test. Le rapport reste dans
`runtime/packaging/validation.json`, hors Git ; un éventuel diagnostic d'échec
est enregistré dans `runtime/packaging/fixture-error.log`.

## Arrêter sans perdre les données

```powershell
.\creditS\Scripts\Activate.ps1
make down
```

Cette commande conserve les volumes. Pour servir localement avec `make serve`,
arrêter d'abord le service Docker API afin de libérer le port 8000.
