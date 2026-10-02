# Scoring de crédit explicable et équitable

Chaîne imposée : DVC → MLflow → Airflow → Docker → FastAPI → Evidently.
Projet pédagogique, en construction phase par phase. Phases 1 et 2 réalisées : environnement, données validées et pipeline DVC de référence.
Les DAGs, Dockerfiles et services restent à implémenter dans leurs phases dédiées.

## Installation Windows PowerShell

Python 3.11.9 ou une version 3.11 plus récente est recommandé. Python 3.11 est utilisé explicitement (le Python système peut être différent).
Création initiale, seule commande de bootstrap exécutée hors du venv :

```powershell
if (-not (Test-Path creditS\Scripts\python.exe)) {
    if (Test-Path .runtime\python311\tools\python.exe) {
        & .runtime\python311\tools\python.exe -m venv creditS
    } else {
        py -3.11 -m venv creditS # Python 3.11.9+ requis
    }
}
.\creditS\Scripts\Activate.ps1
python -c "import sys; print(sys.prefix)"
python -m pip install gnumake==4.4.1
make setup
make test
make lint
```

GNU Make est fourni localement par gnumake dans creditS ; aucune installation globale.
Avec GNU Make déjà installé, `make setup` crée directement creditS s'il est absent.
Sur cette machine, un runtime Python 3.11.9 signé par la Python Software Foundation est conservé dans `.runtime/python311/tools/` et ignoré par Git et Docker ; le Makefile le préfère pour créer creditS. Evidently ne pouvait pas être importé avec le Python 3.11.0 initial.

Les cibles utilisent toujours le Python de creditS et ne nécessitent pas d'activation persistante.
Airflow ne sera installé que dans Docker.

## Linux / macOS

Installer GNU Make comme prérequis système ; `make setup` utilise Python 3.11.

```bash
python3.11 -m venv creditS
source creditS/bin/activate
make setup
make test
```

Sous Linux/macOS, adapter les deux entrées des hooks locaux dans
`.pre-commit-config.yaml` vers `creditS/bin/python`.

## État et conventions

- Archive Kaggle conservée à la racine et ignorée ; extraction idempotente via `make data`.
- `sample_frac: 0.1` en développement ; `1.0` pour les expériences finales.
- `CODE_GENDER` exclu des features ; genre et âge réservés à l'audit.
- `TARGET=1` signifie défaut ; acceptation = 1 − décision de défaut.
- Copie manuelle de `.env.example` vers `.env` ; renseigner les secrets localement.
- Docker Desktop doit être démarré avant les phases utilisant les conteneurs.
- `make data` extrait et valide ; `make train` exécute les quatre stages DVC.
- Les cibles up/down/demo restent à implémenter dans leurs phases dédiées.
- Dépendances directes épinglées dans `requirements.txt`, inventaire complet de
  l'environnement Windows dans `requirements-lock-windows.txt` après installation.
- Les données et les environnements ne doivent jamais être versionnés dans Git.

Les résultats de performance et d'équité seront documentés après mesure.




## Données et pipeline DVC (phase 2)

```powershell
.\creditS\Scripts\Activate.ps1
make data
python -m dvc repro
python -m scripts.verify_data_pipeline
python -m dvc push
python -m dvc status --cloud
```

Sur une copie du dépôt avec accès au remote, utiliser `python -m dvc pull` pour
restaurer les données et sorties. Remote local actuel : `../dvc-storage`.
Les stages exécutent explicitement le Python de creditS. Sous Linux/macOS,
adapter `runtime.python` dans `params.yaml` vers `creditS/bin/python`.

La réserve production est isolée avant le split stratifié 60/20/20 du reste.
Sur sample_frac=0.1 : train 14 760, validation 4 920, test 4 920, production 6 151.
Le modèle logistique numérique de référence atteint AUC validation 0,7345,
PR-AUC 0,2151, seuil métier 0,09. Il n'est pas promu : les évaluations d'équité,
SHAP et le quality gate n'ont pas encore été implémentés.
Les huit lots sont normaux à ce stade ; les scénarios de drift arriveront en phase 10.
Les agrégations et ratios arriveront en phase 3, les expériences MLflow en phase 4.

Preuves d'exécution et limites : [docs/phase2-validation.md](docs/phase2-validation.md).
Le test de régénération a confirmé 54 fichiers identiques par SHA-256 après
réexécution forcée sans cache d'exécution.
