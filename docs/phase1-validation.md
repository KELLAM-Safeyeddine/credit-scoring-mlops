# Phase 1 — initialisation

J'ai initialisé le dépôt et l'environnement `creditS`, puis vérifié les dépendances et les outils de développement.

Ce compte rendu décrit les vérifications de cette phase au moment de son exécution. Le [bilan définitif](project-completion.md) présente le champion final et les dernières validations.

## Choix

- Windows PowerShell, Python 3.11.9 (runtime local signe PSF), environnement racine `creditS`.
- Airflow exclusivement dans Docker ; aucune dépendance apache-airflow locale.
- Dépendances directes épinglées ; contraintes transitives Windows sauvegardées
  après installation et réutilisées par `make setup`.
- GNU Make installé dans le venv via gnumake, sans modification système.
- J'ai suivi le plan du projet et documenté les choix de mise en œuvre.
- Aucun CSV extrait ni pipeline métier implémenté pendant cette phase.

## Vérifications

Les résultats exécutés sont consignés ci-dessous. `phase1-environment.json` contient les
versions réellement importées. Les commandes ci-dessous doivent être exécutées
à la racine du projet.

```powershell
.\creditS\Scripts\Activate.ps1
python -c "import sys; print(sys.prefix)"
make setup
python -m pip check
python scripts/check_environment.py
make lint
make test
python -m pre_commit run --all-files
git status --short
```

Docker et Compose sont présents ; le moteur Docker était arrêté à l'inspection.
Son démarrage n'est pas nécessaire à la phase 1.


## Résultats exécutés

- Création de creditS depuis zéro via make setup : réussie, Python 3.11.9.
- pip mis à jour vers 26.2.1 ; dépendances directes épinglées installées.
- pip check : aucun conflit.
- Imports NumPy, pandas, sklearn, LightGBM, XGBoost, SHAP, Fairlearn,
  Evidently, Pandera, DVC, MLflow et FastAPI : réussis.
- dvc init : réussi ; aucun CSV extrait à cette phase.
- make test : 3 tests réussis.
- Ruff check et Ruff format --check : réussis.
- Hooks pre-commit Ruff (check et format) : réussis sur tous les fichiers.
- make setup final (avec les correctifs) : réussi ; imports vérifiés à chaque setup.

## Compatibilité constatée et corrigée

- Python 3.11.0 produisait une erreur de typing à l'import d'Evidently 0.7.20.
  Python 3.11.9 a corrigé l'import. La distribution NuGet officielle est locale
  dans .runtime, ignorée ; la signature de python.exe est valide (PSF).
- DVC 3.63.0 échouait avec pathspec 1.1.1 ; pathspec est épinglé à 0.12.1.
- SciPy 1.15.3 est explicitement épinglé pour la compatibilité de la résolution.
- Le test depuis zéro a utilisé le cache local des paquets téléchargés ; ce cache
  est ignoré. Une installation ordinaire utilise PyPI.

Distribution Python locale : https://www.nuget.org/packages/python/3.11.9
Documentation : https://docs.python.org/3.11/using/windows.html#the-nuget-org-packages

