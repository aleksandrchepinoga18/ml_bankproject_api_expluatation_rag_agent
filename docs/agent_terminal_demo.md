# Agent Terminal Demo

`scripts/demo_agent.py` is a readable Git Bash client for the already running FastAPI Agent API.
It does not change the API contract; it sends the existing `/analyze` and `/chat` JSON shapes.

The demo feature file is `docs/demo_features_valid.json`. It is a synthetic/demo row from the
existing project acceptance example: all model features are present, with `risky_tx_count=30` and
`wallet_age=10`. It is not real wallet evidence.

When `--features-file docs/demo_features_valid.json` is used, the client prints this origin note in
the terminal output so the demo row is not presented as real wallet data.

Existing user interfaces checked:

- FastAPI Swagger UI: `http://127.0.0.1:8080/docs`
- FastAPI OpenAPI schema: `http://127.0.0.1:8080/openapi.json`
- Legacy Flask operator CLI: `python operator_mode.py`, for `/explain` after `/predict`
- Monitoring dashboard: `docs/local_monitoring_dashboard.html`; this is a metrics viewer, not a chat UI

No existing web chat UI for `/chat` was found. Creating one is outside this step.

## Git Bash Demo Commands

```bash
WALLET=0x01da6f3b20d0540f24d390e28195ad7311516739
```

Document request by wallet:

```bash
python scripts/demo_agent.py \
  --endpoint analyze \
  --wallet-address "$WALLET" \
  --question "Show document sources for wallet $WALLET"
```

Methodology question with methodology sources and no scoring:

```bash
python scripts/demo_agent.py \
  --endpoint chat \
  --question "Explain the wallet risk scoring methodology and cite relevant document sources."
```

Scoring with valid demo features:

```bash
python scripts/demo_agent.py \
  --endpoint analyze \
  --wallet-address "$WALLET" \
  --features-file docs/demo_features_valid.json \
  --question "What is the risk score for wallet $WALLET?"
```

Combined scoring and documents:

```bash
python scripts/demo_agent.py \
  --endpoint analyze \
  --wallet-address "$WALLET" \
  --features-file docs/demo_features_valid.json \
  --question "Explain risk score, factors, and document sources for wallet $WALLET"
```

Useful options:

```bash
python scripts/demo_agent.py --help
python scripts/demo_agent.py --endpoint analyze --question "..." --trace
python scripts/demo_agent.py --endpoint analyze --question "..." --save-json results/demo_agent_raw.json
```
