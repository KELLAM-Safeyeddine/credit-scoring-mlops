# API FastAPI — phase 8

Le service charge exclusivement l'alias `credit-scoring@champion`. Il vérifie
le tag accepté, le résultat du gate, ses règles, le run terminé, les artefacts
SHAP, l'encodeur, l'ordre des colonnes et le seuil du modèle MLflow 3.10.1.
La logique d'inférence et de journalisation est dans `src/serving/`.

## Démarrage Windows

```powershell
.\creditS\Scripts\Activate.ps1
make serve
```

Le script crée une clé HMAC et un jeton de rechargement dans `.env` si nécessaire,
puis démarre Uvicorn sur `127.0.0.1:8000`, avec un seul worker. Les secrets ne sont
jamais affichés ni versionnés. Le Registry est celui de `MLFLOW_TRACKING_URI`,
ou celui de `params.yaml` (localhost:5000). Le Registry Docker sur 5001 est
indépendant : sélectionner explicitement cette URI pour l'utiliser.
Depuis la phase 9, `make up` sert déjà l'API Docker sur 8000. Arrêter ce service
avec `docker compose stop api` avant de lancer le serveur local sur le même port.
Docker utilise `http://mlflow:5000` et peut sélectionner le nom du modèle avec
`MLFLOW_MODEL_NAME` ; par défaut, il s'agit de `credit-scoring`.

```powershell
.\creditS\Scripts\Activate.ps1
Invoke-RestMethod http://localhost:8000/health
$payload = @{
  customer_id = 'client-demo'
  AMT_INCOME_TOTAL = 180000.0
  AMT_CREDIT = 450000.0
  AMT_ANNUITY = 25000.0
  DAYS_EMPLOYED = -1500
  EXT_SOURCE_1 = 0.6
  EXT_SOURCE_2 = 0.7
  EXT_SOURCE_3 = 0.5
} | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/predict -Method Post -ContentType application/json -Body $payload
Invoke-RestMethod http://localhost:8000/explain -Method Post -ContentType application/json -Body $payload
```

**État actuel : aucun champion accepté dans les Registry réels.** Ces appels
répondent 503 jusqu'à une promotion valide. Le service démarre néanmoins et
peut recevoir une demande de rechargement après promotion. `/docs` reste accessible.

## Contrat et explications

`POST /predict` et `POST /explain` acceptent trois montants strictement positifs,
un identifiant client, et des champs facultatifs : prix du bien, ancienneté
professionnelle négative et trois scores externes entre 0 et 1. Aucune conversion
de chaînes numériques n'est autorisée. Les champs supplémentaires, TARGET,
genre, âge et variables dérivées de l'âge sont rejetés avec 422.

L'objet facultatif `history` fournit les 14 agrégats bureau, demandes précédentes
et échéances, calculés en amont selon les mêmes définitions et le même cutoff
que `src/features/aggregations.py`. Aucun accès au dataset Kaggle n'a lieu à la
requête. Omettre un historique signifie « information manquante », pas « zéro ».

Les ratios financiers et la moyenne des scores disponibles sont recalculés.
Les autres variables numériques sont laissées à NaN et imputées par le pipeline
entraîné. Les catégories absentes utilisent le token figé de l'encodeur ; les
catégories non apprises correspondent à des indicateurs nuls. Le contrat complet
et ordonné du modèle est conservé, même avec ce sous-ensemble d'entrées.
`missing_inputs` rend visibles les informations absentes.

**Les performances et métriques d'équité offline ne qualifient pas ce mode
d'entrée partielle.** Avant usage réel, mesurer ces métriques avec les mêmes
champs disponibles que dans les requêtes et enrichir le contrat si nécessaire.
Les tests vérifient le comportement logiciel ; ils ne valident pas un usage
bancaire du modèle synthétique de test.

La sortie contient `default_score`, `predicted_default` (1 = refus), la décision,
le seuil, la version et un identifiant de requête. Le score n'est pas une
probabilité de défaut calibrée. SHAP explique les valeurs transformées/imputées
avec trois contributions et des libellés français ; les contributions sont des
associations du modèle, exprimées dans les unités brutes du modèle, pas des causes.

`GET /model-info` expose version, date, run, seuil, métriques, hash du train,
hash d'évaluation, hash DVC brut et hash du snapshot `dvc.lock` s'il est disponible.
`GET /health` renvoie 200 avec un modèle vérifié et 503 en son absence.
`GET /live` vérifie uniquement que le processus répond. Le healthcheck Docker
utilise cette route pour permettre un rechargement même sans champion initial.

## Rechargement et surveillance

`POST /model/reload` exige l'en-tête `X-Reload-Token` contenant `API_RELOAD_TOKEN`.
La promotion et le rechargement sont deux opérations distinctes. Chaque requête
conserve un seul snapshot pour sa prédiction, ses explications et son journal.
Un chargement invalide conserve le modèle précédent, signale l'échec (503) et
laisse ce dernier servir. Le health indique alors un état de rechargement dégradé.
Le lancement local reste lié à l'interface loopback ; l'exposition Docker arrive
en phase 9.

Les requêtes réussies produisent un événement JSONL dans le chemin configurable
`api.log_path` (ou `API_LOG_PATH`) avec timestamp UTC, UUID, pseudonyme HMAC-SHA256,
score, décision, version/run et **les features encodées avant imputation**. Les
NaN deviennent JSON null, sans changer le contrat des colonnes. L'identifiant
client original n'est ni retourné ni journalisé. Une erreur d'écriture empêche
une réponse de scoring réussie. Un seul worker doit écrire ce journal local.

La pseudonymisation n'est pas une anonymisation ; les features restent sensibles
au sens de confidentialité. Le monitoring pourra lire ces événements en phase 10.
Les attributs d'audit doivent être joints via une source autorisée séparée :
ils ne figurent pas dans les requêtes ou features du service. La rotation de clé
change les pseudonymes ; la conservation et les contrôles d'accès restent à définir
pour tout déploiement réel.

```powershell
.\creditS\Scripts\Activate.ps1
python -m pytest tests/test_api.py -q
make test
make lint
```
