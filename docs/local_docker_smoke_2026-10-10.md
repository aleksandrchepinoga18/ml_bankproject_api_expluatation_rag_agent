# Local Docker Smoke Report - 2026-10-10

Status: `passed after VS Code restart check`; the latest running Docker build has confirmations for documents, methodology, scoring, and combined scenarios. The two previously unfinished checks were completed sequentially after VS Code restart with no rebuild.

Scope: checked the already rebuilt and running local Docker API at `http://127.0.0.1:8080`. No models, corpus, index, training, image publication, remote deployment, rollback, commit, or push were changed or run.

## Versions

- agent_version: `wallet_risk_agent_v1`
- prompt_version: `wallet_risk_agent_prompt_v1`
- model_version: `lightgbm_89ef90618da4df9d`
- schema_version: `scoring_explanation_v1`
- corpus_version: `demo_corpus_v1`
- index_version: `idx_422d98707f81`
- llm_provider: `ollama`
- ollama_model: `qwen2.5:7b-instruct`

## Commands

- `Invoke-RestMethod http://127.0.0.1:8080/health/ready`
- `Invoke-RestMethod http://127.0.0.1:8080/version`
- `Invoke-RestMethod http://127.0.0.1:8080/metrics`
- `Invoke-RestMethod -Method Post http://127.0.0.1:8080/analyze -ContentType application/json -Body <doc payload>`
- `Invoke-RestMethod -Method Post http://127.0.0.1:8080/chat -ContentType application/json -Body <methodology payload>`
- `Invoke-RestMethod -Method Post http://127.0.0.1:8080/analyze -ContentType application/json -Body <scoring payload>`

## Results

| Check | Key result | Status |
| --- | --- | --- |
| /health/ready | status=ready; model=True; scoring_adapter=True; retrieval_index=True; corpus=True | True |
| /version | llm_provider=ollama; ollama_model=qwen2.5:7b-instruct; model=lightgbm_89ef90618da4df9d; index=idx_422d98707f81 | True |
| /metrics | availability=1.0; 5xx=0.0; agent_total=51; source_correctness=1.0 | True |
| /analyze documents | intent=documents; risk=not_requested; retrieval=ok; sources=3; llm=ok/stop; limitations=llm_answer_replaced_for_not_requested_risk,synthetic_demo_documents_are_not_independent_evidence | True |
| /chat methodology | intent=general; risk=not_requested; retrieval=ok; scoring_present=False; sources=method_scoring_limits::chunk_001,method_retrieval_rules::chunk_001; llm=ok/stop; limitations=tool_selection_intent_guard | True |
| /analyze scoring | intent=combined; risk=scored; score=0.358947; scoring=ok; shap=ok; retrieval=ok; llm=ok/stop | True |

## Notes

- Document-only request returned `risk.status=not_requested`, linked wallet sources, real Ollama metadata `llm.status=ok`, and an explicitly marked deterministic replacement: `llm_answer_replaced_for_not_requested_risk`.
- Methodology request returned `intent=general`, methodology source chunks, no scoring tool, and `risk.status=not_requested`.
- Methodology answer text was conservative and mostly stated that risk assessment was not requested, while metadata correctly showed `intent=general`, methodology sources, no scoring, `llm.status=ok`, and `done_reason=stop`.
- Scoring request with a full feature row returned `risk.status=scored`, scoring/shap/retrieval all `ok`, and real Ollama `done_reason=stop`.
- Full JSON response summaries are saved in `results/local_docker_smoke_2026-10-10.json`.

## Latest Rebuilt Docker Demo Rerun

The container was rebuilt again and the four terminal demo scenarios were repeated through `scripts/demo_agent.py`.

- Documents: failed content consistency. `intent=documents`, `risk.status=not_requested`, retrieval `ok`, and sources were correct, but the answer said there was "no sufficient data to assess the risk status."
- Methodology: passed. `intent=general`, method sources, no scoring, `risk.status=not_requested`; answer explains the `method_*` documents and limitations include `llm_answer_replaced_for_methodology_answer`.
- Scoring: passed scored consistency. `risk.status=scored`, structured score `0.35894730197831315`, answer score `0.3589`, `llm.status=ok`, `done_reason=stop`.
- Combined: failed fallback-marker check. Scoring/shap/retrieval succeeded and the structured answer was coherent, but `llm.status=unavailable` from Ollama timeout and the fallback was not explicitly marked in `limitations`.

No automatic guard was added immediately after this rerun. That rerun remained failed until the later local fix and after-VS-Code no-rebuild confirmations summarized below.

## Local Fix Pending Docker Rebuild

After the failed rerun, two local fixes were added:

- broader not-requested answer validation for insufficient-data wording such as "no sufficient data" and "cannot determine risk";
- unified `llm_answer_fallback` marking for deterministic replacements and LLM unavailable/error fallbacks, preserving the original LLM status and error.

Local checks now pass: Ruff, mypy, and full pytest (`70 passed, 929 warnings`).

## Final Four-Scenario Docker Summary

This table uses the latest saved confirmation for each scenario on the latest running Docker build. The after-VS-Code rows were rerun against the same healthy container with no rebuild and no parallel Ollama requests. The methodology scenario is a `/chat` response, so its Agent fields are under `analysis` in the JSON.

| Scenario | Result file | Confirmation | `llm.status` | Fallback markers |
| --- | --- | --- | --- | --- |
| documents | `results/demo_agent_documents_after_vscode_2026-10-10.json` | confirmed after VS Code restart, no rebuild | `ok` | `llm_answer_fallback`, `llm_answer_replaced_for_not_requested_risk` |
| methodology | `results/demo_agent_methodology_after_vscode_2026-10-10.json` | confirmed after VS Code restart, no rebuild | `ok` | `llm_answer_fallback`, `llm_answer_replaced_for_methodology_answer` |
| scoring | `results/demo_agent_scoring_2026-10-10.json` | confirmed before VS Code restart on the same running build | `ok` | none |
| combined | `results/demo_agent_combined_after_vscode_2026-10-10.json` | confirmed after VS Code restart, no rebuild | `ok` | `llm_answer_fallback`, `llm_answer_replaced_for_scored_risk_contradiction` |

No scenario is missing confirmation for the latest running Docker build.

Methodology note: the earlier `results/demo_agent_methodology_2026-10-10.json` had `llm_answer_replaced_for_methodology_answer` but not the generic `llm_answer_fallback`. The current-API rerun above confirms both markers without changing code or Docker.
