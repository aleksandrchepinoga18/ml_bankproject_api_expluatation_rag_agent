# Agent Terminal Demo Report - 2026-10-10

Status: `passed after VS Code restart check`; all four scenarios on the latest running Docker build are now accounted for. The two previously unfinished checks were rerun sequentially with no rebuild.

Scope: local terminal demonstration against the already running Docker API at `http://127.0.0.1:8080` with real Ollama metadata in responses. No model, corpus, index, training, deployment, rollback, commit, or push was performed.

Existing user interface check:

- `http://127.0.0.1:8080/docs` returned the FastAPI Swagger UI for `Wallet Risk Scoring RAG Agent`.
- `http://127.0.0.1:8080/openapi.json` is the machine-readable API schema exposed by FastAPI.
- `python operator_mode.py` is a legacy Flask `/explain` CLI, not the Agent chat interface.
- `docs/local_monitoring_dashboard.html` is a metrics dashboard and was not counted as a chat UI.
- No existing web chat UI for `/chat` was found.

Demo features:

- File: `docs/demo_features_valid.json`.
- Source: existing project acceptance example with all 65 model features present, `risky_tx_count=30`, and `wallet_age=10`.
- Boundary: synthetic/demo row only; not real wallet evidence.

## Commands And Results

| Scenario | Command | Actual result |
| --- | --- | --- |
| Documents by wallet | `python scripts/demo_agent.py --endpoint analyze --wallet-address 0x01da6f3b20d0540f24d390e28195ad7311516739 --question "Show document sources for wallet 0x01da6f3b20d0540f24d390e28195ad7311516739" --save-json results/demo_agent_documents_2026-10-10.json` | Failed content consistency: `intent=documents`, `risk.status=not_requested`, retrieval and sources were correct, but answer text said there was "no sufficient data to assess the risk status." That incorrectly mixes not-requested risk with insufficient-data wording. |
| Methodology sources, no scoring | `python scripts/demo_agent.py --endpoint chat --question "Explain the wallet risk scoring methodology and cite relevant document sources." --trace --save-json results/demo_agent_methodology_2026-10-10.json` | Passed: `intent=general`; `risk.status=not_requested`; selected `retrieval`; no scoring; sources `method_scoring_limits::chunk_001` and `method_retrieval_rules::chunk_001`; answer explains LightGBM score limits and wallet_address retrieval rules by method sources; limitations include `llm_answer_replaced_for_methodology_answer` and `tool_selection_intent_guard`. |
| Scoring with valid features | `python scripts/demo_agent.py --endpoint analyze --wallet-address 0x01da6f3b20d0540f24d390e28195ad7311516739 --features-file docs/demo_features_valid.json --question "What is the risk score for wallet 0x01da6f3b20d0540f24d390e28195ad7311516739?" --save-json results/demo_agent_scoring_2026-10-10.json` | Passed for scored consistency: `intent=score`; `risk.status=scored`; selected `scoring`; structured score `0.35894730197831315`; answer reports score `0.3589`; `llm.status=ok`; `done_reason=stop`; no fallback marker. |
| Combined scoring and documents | `python scripts/demo_agent.py --endpoint analyze --wallet-address 0x01da6f3b20d0540f24d390e28195ad7311516739 --features-file docs/demo_features_valid.json --question "Explain risk score, factors, and document sources for wallet 0x01da6f3b20d0540f24d390e28195ad7311516739" --save-json results/demo_agent_combined_2026-10-10.json` | Failed fallback marker check: scoring/shap/retrieval succeeded, score and sources were correct, synthetic source was marked not evidence, but `llm.status=unavailable` due to Ollama read timeout and the deterministic structured answer was not explicitly marked as fallback/replacement in `limitations`. |

## Issues Found And Fixed Locally

- Methodology answer quality: metadata and sources were correct, but the answer text did not explain methodology. Fixed in `src/agent.py` by making deterministic replacement for `intent=general` summarize the retrieved `method_*` documents.
- Scored-answer contradiction: a live scoring demo produced `risk.status=scored` metadata while the LLM text ended with "Risk status: Insufficient data." Fixed in `src/agent.py` by rejecting scored answers that contradict scored metadata and returning a deterministic scored summary.
- Rebuilt Docker still exposed a vague methodology answer without `method_*` explanation. Fixed locally by adding a methodology-answer quality guard that returns `llm_answer_replaced_for_methodology_answer` and deterministic method-source explanation when the LLM omits method references/content.
- Client provenance: `scripts/demo_agent.py` now prints the `docs/demo_features_valid.json` origin note when a feature file is used.

No new code guard was added immediately after this rerun. At that point, the rebuilt Docker result was failed because of the document-answer wording and missing combined fallback marker. The later after-VS-Code no-rebuild checks below supersede that failed state for the latest running container.

## Local Fix After Latest Rerun

The two latest Docker defects were fixed locally without changing the API contract:

- `risk.status=not_requested` answer validation now rejects broader insufficient-data wording, including "no sufficient data", "cannot determine risk", and similar risk-status language. This covers the saved documents response defect without matching only one exact sentence.
- Answer replacement/fallback marking now uses a shared helper. Every deterministic replacement/fallback adds the specific reason plus a generic `llm_answer_fallback` marker. LLM unavailable/error fallbacks add `llm_answer_fallback_after_unavailable` or `llm_answer_fallback_after_error` while preserving the original `llm.status` and `llm.error`.

Regression checks added:

- not-requested document response with "no sufficient data" wording is replaced.
- unavailable LLM answer fallback is marked while preserving `llm.status=unavailable` and the timeout/error text.

Local verification after the fix:

- `.venv\Scripts\python.exe -m pytest <targeted not_requested/insufficient_data/scored/methodology/truncation/timeout tests> -q` -> `9 passed`.
- `.venv\Scripts\python.exe -m ruff check app src scripts monitoring tests` -> `All checks passed!`
- `.venv\Scripts\python.exe -m mypy app src scripts monitoring tests` -> `Success: no issues found in 51 source files`
- `.venv\Scripts\python.exe -m pytest -q` -> `70 passed, 929 warnings`

Docker was not rebuilt after this local fix.

## After VS Code Restart Check

The running API was checked at `http://127.0.0.1:8080` after VS Code had been closed and reopened. No rebuild, model download, corpus/index change, training, deployment, rollback, commit, or push was performed. The checks that previously remained unfinished were rerun one at a time so there were no parallel Ollama requests.

- Process state: no active `python.exe` demo client was visible via ordinary `Get-Process`; Docker Desktop/backend and Ollama processes were running. Full Docker API/process command-line inspection was blocked by Windows access permissions in the current shell.
- `/health/ready` -> `ready`; model, scoring adapter, retrieval index, and corpus checks all `true`.
- `/version` -> `llm_provider=ollama`, `ollama_model=qwen2.5:7b-instruct`, model `lightgbm_89ef90618da4df9d`, corpus `demo_corpus_v1`, index `idx_422d98707f81`.
- Documents rerun saved to `results/demo_agent_documents_after_vscode_2026-10-10.json`: `intent=documents`, `risk.status=not_requested`, retrieval only, 3 sources, `llm.status=ok`, `done_reason=stop`, limitations include `llm_answer_fallback`, `llm_answer_replaced_for_not_requested_risk`, and `synthetic_demo_documents_are_not_independent_evidence`.
- Methodology rerun saved to `results/demo_agent_methodology_after_vscode_2026-10-10.json`: `intent=general`, `risk.status=not_requested`, retrieval only, 2 method sources, `llm.status=ok`, `done_reason=stop`, limitations include `llm_answer_fallback`, `llm_answer_replaced_for_methodology_answer`, and `tool_selection_intent_guard`.
- Combined rerun saved to `results/demo_agent_combined_after_vscode_2026-10-10.json`: `intent=combined`, `risk.status=scored`, score `0.35894730197831315`, scoring/shap/retrieval all selected, 3 sources, `llm.status=ok`, `done_reason=stop`, limitations include `llm_answer_fallback`, `llm_answer_replaced_for_scored_risk_contradiction`, and `synthetic_demo_documents_are_not_independent_evidence`.

Result: the previously unfinished checks passed on the running container without rebuild; methodology was also rerun to confirm the generic fallback marker on the current API.

## Final Four-Scenario Summary

This summary uses the latest saved confirmation for each scenario on the latest running Docker build. The methodology scenario is a `/chat` response, so its Agent fields are stored under `analysis` in the saved JSON.

| Scenario | Confirmation file | Confirmation status | `llm.status` | Fallback markers |
| --- | --- | --- | --- | --- |
| documents | `results/demo_agent_documents_after_vscode_2026-10-10.json` | confirmed after VS Code restart, no rebuild | `ok` | `llm_answer_fallback`, `llm_answer_replaced_for_not_requested_risk` |
| methodology | `results/demo_agent_methodology_after_vscode_2026-10-10.json` | confirmed after VS Code restart, no rebuild | `ok` | `llm_answer_fallback`, `llm_answer_replaced_for_methodology_answer` |
| scoring | `results/demo_agent_scoring_2026-10-10.json` | confirmed before VS Code restart on the same running build | `ok` | none |
| combined | `results/demo_agent_combined_after_vscode_2026-10-10.json` | confirmed after VS Code restart, no rebuild | `ok` | `llm_answer_fallback`, `llm_answer_replaced_for_scored_risk_contradiction` |

No scenario is missing confirmation for this latest running Docker build.

## Code Checks

- `python -m py_compile scripts/demo_agent.py` -> passed.
- `python scripts/demo_agent.py --help` -> passed.
- `python -c "import json; d=json.load(open('docs/demo_features_valid.json', encoding='utf-8')); print(len(d['features']))"` -> `65`.
- `Invoke-RestMethod http://127.0.0.1:8080/docs` -> returned Swagger UI HTML.
- `Invoke-RestMethod http://127.0.0.1:8080/openapi.json` -> title `Wallet Risk Scoring RAG Agent`; paths `/health/live`, `/health/ready`, `/version`, `/metrics`, `/analyze`, `/chat`.
- `python scripts/demo_agent.py --base-url http://127.0.0.1:65534 --endpoint analyze --question "Ping unavailable API"` -> clear connection error explaining that the API could not be reached.
- `.venv\Scripts\python.exe -m ruff check app src scripts monitoring tests` -> `All checks passed!`
- `.venv\Scripts\python.exe -m mypy app src scripts monitoring tests` -> `Success: no issues found in 51 source files`
- `.venv\Scripts\python.exe -m pytest -q` -> `68 passed, 929 warnings`

Raw JSON responses are saved in `results/demo_agent_*_2026-10-10.json`.
