# Agent terminal demo report - 2026-10-12

Проверка выполнена против уже запущенного пересобранного Docker API без повторной пересборки, commit или push.

## Runtime readiness

- API base URL: `http://127.0.0.1:8080`.
- Docker container: `ml-project-wallet-risk-api-1`, image `wallet-risk-api:local`, port `0.0.0.0:8080->8000/tcp`, status `healthy`.
- `/health/live`: `live`.
- `/health/ready`: `ready`; `model=true`, `scoring_adapter=true`, `retrieval_index=true`, `corpus=true`.
- `/version`: `agent_version=wallet_risk_agent_v1`, `prompt_version=wallet_risk_agent_prompt_v1`, `model_version=lightgbm_89ef90618da4df9d`, `schema_version=scoring_explanation_v1`, `corpus_version=demo_corpus_v1`, `index_version=idx_422d98707f81`, `llm_provider=ollama`, `ollama_model=qwen2.5:7b-instruct`.

## Saved responses

| Scenario | Endpoint | Saved JSON |
| --- | --- | --- |
| documents | `/analyze` | `results/demo_agent_documents_router_2026-10-12.json` |
| methodology | `/chat` | `results/demo_agent_methodology_router_2026-10-12.json` |
| scoring | `/analyze` | `results/demo_agent_scoring_router_2026-10-12.json` |
| combined | `/analyze` | `results/demo_agent_combined_router_2026-10-12.json` |

## Scenario summary

| Scenario | Intent | Risk status | Selected tools | Tool results | LLM | Sources | Limitations | Validation | Result |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| documents | `documents` | `not_requested` | `retrieval` | `retrieval:ok` | `ok/stop` | 3 | `synthetic_demo_documents_are_not_independent_evidence` | none | passed |
| methodology | `general` | `not_requested` | `retrieval` | `retrieval:ok` | `ok/stop` | 2 | `tool_selection_intent_guard` | none | passed |
| scoring | `score` | `scored` | `scoring` | `scoring:ok` | `ok/stop` | 0 | none | none | metadata/tooling passed; answer text consistency failed |
| combined | `combined` | `scored` | `scoring, shap, retrieval` | `scoring:ok, shap:ok, retrieval:ok` | `ok/stop` | 3 | `llm_answer_retry_after_truncation`, `synthetic_demo_documents_are_not_independent_evidence` | none | passed after truncation retry |

## Trace evidence

- documents: `parse_request[node]:ok -> llm_tool_decision[llm_tool_decision]:ok -> retrieval[tool_call]:ok -> retrieval[tool_result]:ok -> build_answer[node]:ok -> validate_answer[node]:ok`
- methodology: `parse_request[node]:ok -> llm_tool_decision[llm_tool_decision]:ok -> retrieval[tool_call]:ok -> retrieval[tool_result]:ok -> build_answer[node]:ok -> validate_answer[node]:ok`
- scoring: `parse_request[node]:ok -> llm_tool_decision[llm_tool_decision]:ok -> scoring[tool_call]:ok -> scoring[tool_result]:ok -> build_answer[node]:ok -> validate_answer[node]:ok`
- combined: `parse_request[node]:ok -> llm_tool_decision[llm_tool_decision]:ok -> scoring[tool_call]:ok -> scoring[tool_result]:ok -> shap[tool_call]:ok -> shap[tool_result]:ok -> retrieval[tool_call]:ok -> retrieval[tool_result]:ok -> build_answer[node]:ok -> validate_answer[node]:ok`

The router node is an internal LangGraph state-update node and does not append public trace entries. The public trace still confirms real tool order and completion through `validate_answer`.

## Checks

- Real transitions: all four scenarios reached `validate_answer`.
- No repeated tools: all scenarios had unique `tool_statuses` names; no `duplicate_blocked` status appeared.
- Documents/general guard: documents and methodology selected and executed only `retrieval`; no scoring or SHAP tool call appeared.
- SHAP dependency: combined executed `shap` only after `scoring:ok` and `risk.status=scored`.
- Combined recursion limit: combined completed through `validate_answer` at trace step 10, so the configured graph recursion limit was sufficient.
- Sources: documents and combined returned wallet-linked sources only; methodology returned general method documents only.
- Synthetic demo boundary: synthetic source was marked with `synthetic_demo_documents_are_not_independent_evidence` where present.

## Answer and fallback notes

- documents: answer was consistent with `risk.status=not_requested` and did not include fallback markers.
- methodology: answer described retrieved methodology documents and did not run scoring; `tool_selection_intent_guard` shows an attempted scoring selection was removed.
- scoring: structured metadata is coherent (`score=0.358947`, `threshold=0.344386`, `prediction=1`), but the real LLM answer text incorrectly said the score was below the threshold and included insufficient-data wording. This is an answer consistency failure in the source LLM text; no deterministic fallback marker was emitted.
- combined: first LLM answer was truncated and the API retried safely, shown by `llm_answer_retry_after_truncation`. The saved final response reached validation with `validation_errors=[]`. This verifies safe completion of the retry path, not the quality of the truncated original LLM text.

## Scored-answer guard rerun

After the scored-answer consistency guard was added, only the scoring and combined terminal demo scenarios were repeated against the already rebuilt Docker API with real Ollama. Docker was not rebuilt again.

| Scenario | Endpoint | Saved JSON | score | threshold | prediction | relation | LLM handling | Limitations | Validation | Trace end |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| scoring | `/analyze` | `results/demo_agent_scoring_scored_guard_2026-10-12.json` | 0.358947 | 0.344386 | 1 | score is above threshold | deterministic fallback used; real `llm.status=ok` preserved | `llm_answer_fallback`, `llm_answer_replaced_for_scored_risk_contradiction` | none | `validate_answer[node]:ok` |
| combined | `/analyze` | `results/demo_agent_combined_scored_guard_2026-10-12.json` | 0.358947 | 0.344386 | 1 | score is above threshold | LLM retry text accepted after deterministic scored header | `llm_answer_retry_after_truncation`, `synthetic_demo_documents_are_not_independent_evidence` | none | `validate_answer[node]:ok` |

Rerun trace evidence:

- scoring: `parse_request[node]:ok -> llm_tool_decision[llm_tool_decision]:ok -> scoring[tool_call]:ok -> scoring[tool_result]:ok -> build_answer[node]:ok -> validate_answer[node]:ok`
- combined: `parse_request[node]:ok -> llm_tool_decision[llm_tool_decision]:ok -> scoring[tool_call]:ok -> scoring[tool_result]:ok -> shap[tool_call]:ok -> shap[tool_result]:ok -> retrieval[tool_call]:ok -> retrieval[tool_result]:ok -> build_answer[node]:ok -> validate_answer[node]:ok`

Rerun checks:

- In both rerun responses, deterministic scored output states `score=0.358947`, `threshold=0.344386`, `prediction=1`, and `relation=score is above threshold`, matching structured `risk`.
- The scoring response used fallback because the real LLM text contradicted scored metadata; both generic and specific fallback markers were present.
- The combined response accepted the retry LLM explanation after adding deterministic scored facts; no scored contradiction fallback marker was needed.
- Both responses reached `validate_answer` with `validation_errors=[]`.
- The regression test no longer depends on `results/demo_agent_scoring_router_2026-10-12.json`; the saved defect text is embedded in the test fixture.

