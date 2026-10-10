# Interview Demo

## One-minute architecture

Wallet Risk Analyst is a production-like local ML/RAG service for crypto wallet risk analysis.

- FastAPI exposes `/analyze`, `/chat`, `/health/live`, `/health/ready`, `/version`, and `/metrics`.
- The Agent runs through LangGraph and selects allow-listed tools: scoring, SHAP-style explanation, and retrieval.
- Scoring uses the existing LightGBM artifact, saved feature order, preprocessing, and threshold.
- Retrieval uses a versioned demo corpus with `document_id`, `chunk_id`, `wallet_address`, `source_kind`, corpus version, and index version.
- The answer layer separates model scores from document evidence and refuses to assert proven fraud.
- Observability writes local structured JSONL logs, local metrics snapshots, a local dashboard, and file-only test alerts.

## Demo request

Use the terminal demo client when the Docker API is already running:

```bash
WALLET=0x01da6f3b20d0540f24d390e28195ad7311516739

python scripts/demo_agent.py \
  --endpoint analyze \
  --wallet-address "$WALLET" \
  --features-file docs/demo_features_valid.json \
  --question "Explain risk score, factors, and document sources for wallet $WALLET"
```

`docs/demo_features_valid.json` is a synthetic/demo feature row from the checked project example. It is not real wallet evidence.

The FastAPI Swagger UI is available at `http://127.0.0.1:8080/docs`. No web chat UI for `/chat` exists yet; the monitoring dashboard is a metrics viewer, not a chat interface.

PowerShell equivalent:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/analyze `
  -Method Post `
  -ContentType "application/json" `
  -Body '{
    "question":"Explain risk score, factors, and document sources for wallet 0x01da6f3b20d0540f24d390e28195ad7311516739",
    "wallet_address":"0x01da6f3b20d0540f24d390e28195ad7311516739",
    "features":{"risky_tx_count":30,"wallet_age":10}
  }'
```

Expected talking points:

- `intent=combined`
- scoring tool returns a model score and model/schema version
- SHAP tool returns top model factors when features are present
- retrieval returns sources with `document_id` and `chunk_id`
- answer includes limitations and does not claim real-world fraud

## Tools and sources

- `scoring`: computes the row/wallet risk model output or returns `insufficient_data`.
- `shap`: explains the current model score for the supplied feature row.
- `retrieval`: returns wallet-linked or methodology chunks. Documents from a different wallet are filtered out before ranking.

Synthetic demo documents are explicitly marked as `synthetic_demo`. They are not real investigations and are not evidence of fraud or model quality.

## Error handling

- Address without features returns `insufficient_data`.
- Retrieval or LLM errors return partial structured responses with explicit tool status.
- LLM truncation is retried once; if still truncated, the service falls back to structured tool output.
- Prompt injection text does not change routing or allow unsupported claims.
- Stale explanation JSON is returned as `stale`, not current.

## Monitoring demo

Local observability artifacts:

```powershell
python scripts\build_monitoring_dashboard.py
```

Outputs:

- `results/monitoring_snapshot.json`
- `docs/local_monitoring_dashboard.html`
- `monitoring/alerts/test_alert_<date>.json`

The test alert is local-file-only. No external messages are sent.

## Limitations to say out loud

- This is a portfolio production-like system, not evidence of corporate production operation.
- Metrics from the original LightGBM test split are row-level model metrics, not RAG/Agent quality metrics.
- Synthetic labels and synthetic demo documents are not real labels or real investigations.
- The current GitHub Actions workflows are prepared but have not been run in GitHub here.
- Remote deployment and rollback are documented but not executed or verified without a test server.
