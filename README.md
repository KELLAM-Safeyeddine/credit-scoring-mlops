# Scoring de crédit explicable et équitable

Chaîne imposée : DVC → MLflow → Airflow → Docker → FastAPI → Evidently.
Projet pédagogique, en construction phase par phase. Phase 1 : environnement et squelettes.
Les modules métier, DAGs, Dockerfiles et services sont encore à implémenter.

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

- Archive Kaggle conservée à la racine et ignorée ; extraction prévue en phase 2.
- `sample_frac: 0.1` en développement ; `1.0` pour les expériences finales.
- `CODE_GENDER` exclu des features ; genre et âge réservés à l'audit.
- `TARGET=1` signifie défaut ; acceptation = 1 − décision de défaut.
- Copie manuelle de `.env.example` vers `.env` ; renseigner les secrets localement.
- Docker Desktop doit être démarré avant les phases utilisant les conteneurs.
- Les cibles data/train/up/down/demo échouent explicitement tant que leur phase
  n'est pas implémentée.
- Dépendances directes épinglées dans `requirements.txt`, inventaire complet de
  l'environnement Windows dans `requirements-lock-windows.txt` après installation.
- Les données et les environnements ne doivent jamais être versionnés dans Git.

Les résultats de performance et d'équité seront documentés après mesure.


