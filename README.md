# Lancer le projet de scoring de crédit

Prérequis : Git, Python **3.11.9 ou plus récent dans la série 3.11**, Docker
avec conteneurs Linux et Docker Compose. Airflow fonctionne uniquement dans Docker.

## Installation Windows (PowerShell)

```powershell
py -3.11 -m venv creditS
.\creditS\Scripts\Activate.ps1
python scripts/setup.py
```

## Installation Linux/macOS

```bash
python3.11 -m venv creditS
source creditS/bin/activate
python scripts/setup.py
```

Les dépendances sont installées uniquement dans `creditS`. Les images Docker
utilisent leurs propres fichiers de dépendances et locks.

## Démarrage

Placer l'archive ZIP Kaggle **Home Credit Default Risk** à la racine du dépôt,
puis démarrer Docker. Le nom de l'archive est détecté automatiquement.

```powershell
.\creditS\Scripts\Activate.ps1
python scripts/start_airflow.py
docker compose ps
Invoke-RestMethod http://localhost:8000/live
Invoke-RestMethod http://localhost:8000/health
```

Sur Linux/macOS, commencer par `source creditS/bin/activate`, puis exécuter
`python scripts/start_airflow.py`. Utiliser `curl` pour vérifier les endpoints.

Le démarrage génère les secrets manquants dans `.env`, construit les images
et attend les healthchecks. Les identifiants Airflow sont conservés dans `.env`.

| Service | Adresse |
|---|---|
| Interface de scoring | http://localhost:8000/ |
| Documentation de l'API | http://localhost:8000/docs |
| MLflow Docker | http://localhost:5001 |
| Airflow | http://localhost:8081 |

Pour lancer uniquement l’interface et l’API en local, avec le Registry MLflow local :

```powershell
.\creditS\Scripts\Activate.ps1
python scripts/mlflow_server.py
python scripts/serve_api.py
```

L’interface est accessible sur http://localhost:8000/. Elle permet de saisir un dossier,
de charger un exemple fictif et d’afficher le score et les trois raisons SHAP.
Le port 8000 doit être libre : arrêter l’API Docker avant de lancer ce serveur local.

L'API charge `credit-scoring@champion` depuis le Registry Docker. Les données,
modèles et bases MLflow ne sont pas inclus dans Git. Sur une installation neuve,
`/live` répond 200 ; `/health`, `/predict` et `/explain` restent à 503 tant qu'un
champion accepté n'est pas disponible. Les volumes existants conservent les modèles.

## Commandes du pipeline

Après activation de `creditS`, les cibles `make data`, `make train`, `make fairness`,
`make promote`, `make monitor` et `make demo` permettent d'exécuter le pipeline.
`make train` démarre MLflow local sur http://localhost:5000 ; ce Registry est distinct
du Registry Docker. Toute promotion reste soumise au quality gate.

Pour restaurer les données avec DVC, configurer un remote accessible et lancer
`python -m dvc pull` après activation de `creditS`. Le remote par défaut est
le dossier local voisin `../dvc-storage` ; il n'est pas hébergé sur GitHub.

## Arrêt et redémarrage

```powershell
.\creditS\Scripts\Activate.ps1
docker compose down
docker compose up -d
```

`docker compose down` conserve les volumes et les modèles. Ne pas ajouter `-v`
pour un arrêt ordinaire. Ne jamais versionner `.env`, `creditS`, les archives
Kaggle, les données ou les bases MLflow.
