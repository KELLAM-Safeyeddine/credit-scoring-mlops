# Preuves — phase 8

Environnement Windows, `creditS` Python 3.11.9, FastAPI 0.119.0,
Pydantic 2.12.3, MLflow 3.10.1, SHAP 0.49.1. Aucune installation globale
ou installation locale d'Airflow.

- `make test` exécuté : **84 tests réussis**, dont **21 tests API**.
- Ruff : contrôle et formatage du dépôt réussis.
- TestClient utilise un vrai Registry SQLite temporaire avec modèles LightGBM,
  artefacts d'encodeur et SHAP, enregistrement et alias réels MLflow.
- `/predict` et `/explain` : scores, décisions, seuils et versions identiques ;
  trois raisons SHAP en français.
- Payloads invalides, attributs d'audit, champs inconnus, NaN et Infinity : 422.
- Les ratios du service correspondent au calcul offline ; l'ordre des colonnes
  est préservé, les valeurs absentes restent manquantes avant imputation.
- Journaux : UUID distincts, pseudonyme HMAC stable, identifiant original absent,
  valeurs manquantes représentées par null et colonnes alignées sur le modèle.
- Un nouveau modèle accepté est promu en version 2 dans le Registry de test.
  L'API conserve la version 1 jusqu'au rechargement autorisé, puis prédit et
  explique avec la version 2 et son nouveau seuil.
- Un alias rejeté est refusé ; un échec de rechargement conserve le modèle déjà
  vérifié. Un serveur sans champion renvoie 503 et reste disponible pour reload.
- Serveur Uvicorn réellement lancé sur `127.0.0.1:8000`, Registry local 5000 :
  health 503 attendu, OpenAPI accessible et cinq routes exposées. Preuve privée
  sauvegardée dans `runtime/api/phase8-http-validation.json`.

Les métriques d'acceptation des fixtures sont **synthétiques** : elles valident
les mécanismes logiciels et ne constituent pas une qualification du modèle
Home Credit. Les Registry réels conservent leurs candidats rejetés ; aucun alias
champion de production n'a été créé pendant cette phase. Les métriques offline
ne certifient pas le mode d'entrée partielle de l'API.

24 avertissements de bibliothèques lors de la suite complète : dépréciations
Starlette/AnyIO et Evidently/Litestar, noms de colonnes LightGBM et inférence de
schéma MLflow sur données synthétiques. Aucun échec de test.

```powershell
.\creditS\Scripts\Activate.ps1
python -c "import sys; print(sys.prefix)"
make test
make lint
python -m ruff format --check .
make serve
```

Docker API et intégration Compose seront réalisés en phase 9 après validation.
