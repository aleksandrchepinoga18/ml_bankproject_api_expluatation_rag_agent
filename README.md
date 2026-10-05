# ML Bank Scoring Pipeline

Production-ready ML system for bank risk scoring with:
- Flask API for real-time predictions
- SHAP-based saved explanations without LLM or document retrieval
- Operator mode for reading generated explanations
- Drift monitoring (PSI, KS-test)
- Automatic retraining on data drift
- Quality tracking (ROC-AUC, F1)

## 🚀 Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run API
```bash
python app/api.py
```

### Run FastAPI Agent API

The FastAPI service exposes the RAG Agent while the legacy Flask `/predict` and `/explain` API remains available through `python app/api.py`.

Check installed dependency versions:

```bash
python -c "import importlib.metadata as m; print('fastapi', m.version('fastapi')); print('uvicorn', m.version('uvicorn')); print('pydantic', m.version('pydantic'))"
```

Expected compatible versions for this project:

```text
fastapi==0.111.0
uvicorn==0.30.0
pydantic==2.7.4
```

Local deterministic smoke test without Ollama:

```bash
AGENT_LLM_PROVIDER=disabled python scripts/smoke_fastapi.py
```

Run the FastAPI server:

```bash
AGENT_LLM_PROVIDER=disabled uvicorn app.fastapi_app:app --host 127.0.0.1 --port 8000
```

Use `AGENT_LLM_PROVIDER=ollama` and `OLLAMA_MODEL=qwen2.5:7b-instruct` when local Ollama is available.

FastAPI endpoints:

```text
GET  /health/live
GET  /health/ready
GET  /version
POST /analyze
POST /chat
```

Example document request:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"question":"Show document sources for wallet 0x01da6f3b20d0540f24d390e28195ad7311516739","wallet_address":"0x01da6f3b20d0540f24d390e28195ad7311516739"}'
```

Docker Compose local run:

```bash
cp .env.example .env
docker compose up --build
```

### Test API (example)
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"first_tx_timestamp": 1615161978.0, "last_tx_timestamp": 1627349954.0, ...}'
```

## Saved SHAP Explanations

The project includes an explanation layer without LLM. It is not a full RAG system because it does not retrieve documents from a knowledge base. When `/predict` is called, the API calculates the risk probability for class `1`, builds SHAP explanations for the same input row and model class, checks business rules, and saves the explanation to:

```text
monitoring/explanations/{user_id}.json
```

Class `1` means a risky wallet. The saved decision uses the trained threshold from `models/lightgbm_best_threshold.pkl`:

```python
decision = "отказ" if risk_probability >= best_threshold else "одобрение"
```

### What `/predict` Saves

`/predict` accepts both payload styles.

Existing top-level feature format:

```json
{
  "user_id": "TEST_USER_3",
  "first_tx_timestamp": 1615161978.0,
  "wallet_age": 259200,
  "risky_tx_count": 28
}
```

Explicit feature wrapper format:

```json
{
  "user_id": "TEST_USER_3",
  "features": {
    "first_tx_timestamp": 1615161978.0,
    "wallet_age": 259200,
    "risky_tx_count": 28
  }
}
```

For each item, `/predict`:

1. Keeps `user_id` and `wallet_address` as separate fields. For backward compatibility, if `user_id` is missing it falls back to `wallet_address`; if neither exists, it generates a safe id.
2. Reads features from `features` when present, otherwise from the current top-level payload.
3. Reorders columns with saved `feature_names` and applies train-only missing-value medians from `models/lightgbm_preprocessing.pkl`.
4. Calculates `risk_probability = model.predict_proba(X)[:, 1]`.
5. Calculates SHAP values with `TreeExplainer`.
6. Saves `top_features` and `rule_triggers` to JSON.
7. Returns the prediction response with `prediction`, `risk_probability`, and `user_id`.

Example explanation file:

```json
{
  "user_id": "TEST_USER_3",
  "score": 0.4026454212102552,
  "decision": "отказ",
  "top_features": [
    {
      "feature": "time_since_first_deposit",
      "value": 1021.0,
      "impact": 0.4384325840734663
    },
    {
      "feature": "market_adxr",
      "value": 47.10999338,
      "impact": 0.18635984285997645
    }
  ],
  "rule_triggers": [],
  "timestamp": "2026-05-08T19:59:05.054436"
}
```

`top_features` contains the top positive SHAP factors by default. Positive SHAP impact means the feature increased the model's risk score for class `1`.

Business rules are saved separately in `rule_triggers`, so important rules are not hidden when a feature does not make it into the SHAP top factors:

```json
{
  "rule_triggers": [
    {
      "rule": "risky_tx_count_gt_20",
      "feature": "risky_tx_count",
      "value": 28.0,
      "message": "обнаружено 28 рисковых транзакций"
    },
    {
      "rule": "wallet_age_lt_7_days",
      "feature": "wallet_age",
      "value": 259200.0,
      "message": "кошелёк создан менее недели назад (3 дней)"
    }
  ]
}
```

### Implementation Requirements

1. `score` is the risk probability from `model.predict_proba(X)[:, 1]`, meaning class `1`.
2. If `score >= best_threshold`, `decision = "отказ"`.
3. If `score < best_threshold`, `decision = "одобрение"`.
4. `/predict` must not break the old response format used by existing clients.
5. `user_id` should be passed explicitly for meaningful explanation lookup. For backward compatibility, when an old client omits `user_id`, the API generates a safe id and returns it in the response.
6. SHAP must be calculated with features in the same order used during model training.
7. `feature_names` are loaded from `models/lightgbm_feature_names.pkl`.
8. If an explanation file is missing, `/explain` returns `404`.
9. If an explanation is found, `/explain` returns a Russian text explanation.
10. Business rules must be checked against the raw feature values, not only against `top_features`.

### Test Explanations End-to-End

Open two terminals from the project root.

1. Start the API in terminal 1.

```bash
python app/api.py
```

2. In terminal 2, call `/predict` with a stable `user_id`. This creates the explanation JSON.

```bash
python -c "import pandas as pd, requests; row=pd.read_csv('monitoring/reference/reference_features.csv').iloc[3].to_dict(); row['user_id']='TEST_USER_3'; r=requests.post('http://127.0.0.1:5000/predict', json=row); print(r.status_code); print(r.json())"
```

Equivalent curl shape for real clients:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id":"TEST_USER_3","features":{"wallet_age":259200,"risky_tx_count":28}}'
```

In practice, include all model features in `features`; the short curl above shows only the payload shape.

3. Check that the JSON file was created.

```bash
ls monitoring/explanations/
```

On Windows PowerShell:

```powershell
Get-ChildItem monitoring/explanations
```

4. Request the text explanation.

```bash
python -c "import requests; r=requests.post('http://127.0.0.1:5000/explain', json={'user_id':'TEST_USER_3'}); print(r.status_code); print(r.json())"
```

Equivalent curl:

```bash
curl -X POST http://127.0.0.1:5000/explain \
  -H "Content-Type: application/json" \
  -d '{"user_id":"TEST_USER_3"}'
```

5. Check the operator CLI.

```bash
python operator_mode.py
```

Enter:

```text
TEST_USER_3
```

Expected flow:

```text
/predict -> saves monitoring/explanations/TEST_USER_3.json
/explain -> reads that JSON and returns user_id, decision, score, and a Russian explanation
```

Important: `/explain` can return an explanation only after `/predict` has already been called for the same `user_id`.

### Operator Mode

After the API is running and `/predict` has created an explanation, start the operator CLI in another terminal:

```bash
python operator_mode.py
```

Enter the user id without quotes:

```text
TEST_USER_3
```

The script calls `/explain` and prints the explanation. If there is no saved explanation yet, it asks you to call `/predict` first.

### Explanation Implementation Notes

- `train_pipeline.py` saves feature names to `models/lightgbm_feature_names.pkl`.
- `train_pipeline.py` saves preprocessing metadata to `models/lightgbm_preprocessing.pkl`.
- `src/inference.py` loads the model, SHAP explainer, feature names, preprocessing metadata, and threshold.
- `app/api.py` calls SHAP inside `/predict` and saves explanations.
- `/explain` validates `user_id`, reads the saved JSON, uses `rule_triggers` first, and adds SHAP factors as extra context.
- `operator_mode.py` is a simple CLI wrapper around `/explain`.

### Model Validation Notes

- The prediction target is wallet risk for wallets not seen during training.
- The data unit is a row-level wallet observation; one `wallet_address` can have multiple rows.
- Train/validation/test are split by `wallet_address`, so the same wallet cannot appear in multiple splits.
- The operating threshold is selected on validation by F1. Test metrics are computed once using that fixed threshold.
- `results/LightGBM_top50_risky_wallets.csv` contains 50 unique wallets using `max(row_probability)` per wallet and does not include `target`; `results/LightGBM_top50_risky_wallets_evaluation.csv` keeps label columns for offline evaluation only.

## 🔍 Monitoring

### Check for drift and retrain if needed
```bash
python -m monitoring.retrain_if_needed
```

### Simulate labels (for testing only)
```bash
python monitoring/simulate_labels.py
```

## 📁 Project Structure
- `app/` — Flask API
- `src/` — Model training pipeline
- `operator_mode.py` — CLI for operator explanations
- `monitoring/` — Drift detection, quality, retraining, saved explanations
- `models/` — Saved models (not in Git)
- `data/` — Raw data (not in Git)

## ⚠️ Note
- Data and models are excluded from Git (see `.gitignore`)
- Use environment variables for database credentials in production
