# Command 11 Acceptance Report

Status: `partially_verified`.

Synthetic labels and synthetic_demo documents are not treated as real evidence.

## Real Docker/Ollama Checks

Checked on 2026-10-05 against the running container `ml-project-wallet-risk-api-1`
from image `wallet-risk-api:local`, published as `127.0.0.1:8080->8000`.

| Check | Evidence | Status |
| --- | --- | --- |
| Docker readiness | `/health/ready` returned `ready` with `model=true`, `scoring_adapter=true`, `retrieval_index=true`, `corpus=true`. | passed |
| Runtime version | `/version` returned `llm_provider=ollama`, `ollama_model=qwen2.5:7b-instruct`, `model_version=lightgbm_89ef90618da4df9d`, `index_version=idx_422d98707f81`. | passed |
| Metrics endpoint | Refreshed dashboard snapshot from current logs: 117 requests, availability `1.0`, 5xx error rate `0.0`, 33 Agent rows, source correctness `15/15`. | passed |
| Structured API logs | `monitoring/logs/api_requests_2026-10-05.jsonl` contains Docker HTTP calls for `/analyze`, `/chat`, `/metrics`, readiness/live checks with status code, latency, method, path, and sanitized headers only. | passed |
| Structured Agent logs | `monitoring/logs/agent_requests_2026-10-05.jsonl` contains Agent responses with route, intent, risk status, source correctness, tool statuses, LLM provider/model/status/done_reason, and limitations. | passed |
| `/analyze` document request | Agent log: route `/analyze`, intent `documents`, selected retrieval, `llm.provider=ollama`, `llm.status=ok`, `done_reason=stop`, source correctness `correct=true`, 3 sources. | passed |
| `/chat` combined request without features | Agent log: route `/chat`, data_status `insufficient_data`, risk_status `insufficient_data`, score `null`, limitations include `features_required` and `llm_answer_replaced_for_insufficient_data`, `llm.status=ok`, `done_reason=stop`. | passed |
| General methodology request without wallet scoring | Live `/chat` returned intent `general`, selected `retrieval`, `llm.status=ok`, `done_reason=stop`, and methodology sources `method_feature_dictionary::chunk_001`, `method_scoring_limits::chunk_001`, `method_retrieval_rules::chunk_001`. | passed |
| Methodology request with wallet and cite sources | Operator-confirmed follow-up Docker/Ollama check: methodology routing now uses methodology retrieval and no longer substitutes wallet observation/synthetic sources. | passed |
| Address without features | Live `/analyze` returned `data_status=insufficient_data`, `risk.status=insufficient_data`, `score=null`, `reason=features_required`; the answer explicitly rejected conclusions about low risk or absence of fraud. | passed |
| `not_requested` document-only answer | Operator-confirmed follow-up Docker/Ollama check: when `risk.status=not_requested`, the answer says risk assessment was not requested and does not say the data is insufficient. | passed |
| Answer replacement metric | Operator-confirmed follow-up Docker/Ollama check: `/metrics` exposes answer replacement count/rate/definition for `llm_answer_replaced_*` fallback events. | passed |

## Issues Found And Fixed Locally

| Issue | Fix | Verification | Docker status |
| --- | --- | --- | --- |
| The insufficient-data deterministic answer could append "LLM unavailable" even when `llm.status=ok` and the LLM answer was merely replaced for safety. | `src/agent.py` now adds the LLM-unavailable sentence only when final LLM status is `unavailable` or `error`. | `python -m pytest tests/test_agent.py::test_langgraph_replaces_llm_low_risk_claim_when_data_is_insufficient -q` -> `1 passed`; target suite -> `34 passed`. | passed in operator-confirmed Docker/Ollama follow-up. |
| A methodology request containing wallet address, `risk`, and `cite sources` was classified as `combined` and returned wallet/synthetic docs instead of methodology docs. | `classify_intent` now treats explicit methodology/rules/limits/feature-dictionary questions as `general` unless the user asks to compute score/probability. Added regression test for the exact wording shape. | `python -m pytest tests/test_agent.py::test_methodology_with_risk_and_sources_stays_general -q` -> `passed`; target suite -> `34 passed`. | passed in operator-confirmed Docker/Ollama follow-up. |
| A document-only `/analyze` response with `risk.status=not_requested` could use "insufficient data" language. | Unsafe LLM text is replaced with a deterministic answer when it contradicts `risk.status=not_requested`. | `python -m pytest tests/test_agent.py::test_langgraph_replaces_not_requested_answer_that_claims_insufficient_data tests/test_observability.py::test_collect_metrics_counts_replaced_llm_answers -q` -> `2 passed`. | passed in operator-confirmed Docker/Ollama follow-up. |
| LLM answer replacement was visible only indirectly through limitations. | Metrics now expose answer replacement count/rate/definition for `llm_answer_replaced_*` events. | `python -m pytest tests/test_observability.py::test_collect_metrics_counts_replaced_llm_answers -q` -> `passed`. | passed in operator-confirmed Docker/Ollama follow-up. |
| Runtime logs, alerts, MLflow/cache files, old review dump, and local DBs were easy to add accidentally. | `.gitignore` now excludes `.matplotlib/`, `mlruns/`, `project_review/`, `project_review.zip`, `monitoring/{alerts,drift_logs,explanations,logs}/`, `*.db`, and `*.sqlite3`. | `git check-ignore -v` confirms `.env`, monitoring logs/alerts, mlruns, `.matplotlib`, `project_review.zip`, and `mlflow/mlflow.db` are ignored. | not applicable |

## Local Mock/TestClient Acceptance

These scenarios were executed locally with TestClient, disabled/fake LLM providers, or Flask test client. They are not real Docker/Ollama checks.

| Scenario | Command | Expected | Actual | Status |
| --- | --- | --- | --- | --- |
| 7.1 valid features and linked document | TestClient POST `/analyze` with full feature row | risk scored, SHAP runs, retrieval returns linked source | scoring/shap/retrieval `ok`, 3 sources, no validation errors | passed |
| 7.2 address without features | TestClient POST `/analyze` with wallet only | `insufficient_data`, score `null` | `data_status=insufficient_data`, `score=null` | passed |
| 7.3 foreign wallet document excluded | TestClient document request for one wallet | no foreign wallet sources | foreign source count `0` | passed |
| 7.4 stale legacy JSON | Flask POST `/explain` against stale explanation JSON | `risk.status=stale` | `risk.status=stale` | passed |
| 7.5 unavailable retrieval/LLM partial response | WalletRiskAgent with BrokenRetriever and DisabledLLMProvider | explicit retrieval error and no crash | `data_status=partial`, retrieval error `qdrant_unavailable` | passed |
| 7.6 prompt injection in user/document text | TestClient POST `/analyze` with injection text | no proven-fraud claim | intent remained `documents`, no proven-fraud phrase | passed |
| 7.7 multiple rows aggregation | Flask POST `/predict` with two rows | `max_row_probability`, records `2` | aggregation rule `max_row_probability`, records `2` | passed |
| 7.8 rollback version | rollback runbook | compatible image/artifacts restored and smoke tested | not executed by instruction | not_run |

## Mock LLM Checks

- `passed`: fake-provider tests for tool-call augmentation, timeout reporting, truncation retry, and truncation fallback.
- Latest targeted local command: `python -m pytest tests/test_agent.py tests/test_fastapi_app.py tests/test_observability.py -q` -> `34 passed`, 922 warnings.

## Dashboard Snapshot

- Updated with `python scripts/build_monitoring_dashboard.py`.
- Snapshot: `results/monitoring_snapshot.json`.
- Dashboard: `docs/local_monitoring_dashboard.html`.
- Snapshot generated at `2026-10-05T16:56:03Z`.
- Versions in snapshot: `llm_provider=ollama`, `ollama_model=qwen2.5:7b-instruct`, `model_version=lightgbm_89ef90618da4df9d`.
- Current latency aggregates all HTTP requests, including health/version/metrics and non-Agent endpoints. It is not an Agent-only latency estimate.
- Source correctness checks link/support structure for sources and chunks; it is not a guarantee that every sentence in the generated answer is factually true.
- Open locally from repository root:

```powershell
Start-Process .\docs\local_monitoring_dashboard.html
```

or open this file in a browser:

```text
ml-project/docs/local_monitoring_dashboard.html
```

## Not Run

- GitHub Actions CI: not run locally; requires push/PR.
- Protected test image publication: not run by instruction.
- Remote test deployment: not run by instruction and no remote target/secrets are configured.
- Rollback: not run by instruction.

## Label Provenance

```json
{
  "status": "not_trusted",
  "reason": "label_source column missing",
  "trusted_quality_evidence": false
}
```
