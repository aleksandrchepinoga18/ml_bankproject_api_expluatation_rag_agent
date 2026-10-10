# Project Progress

Last updated: 2026-10-06.

## Current Status

Commands completed through Command 9.

The project now has:

- Scoring boundary and versioned scoring contract.
- Wallet-level aggregation and source-feature audit.
- MLflow tracking rehearsal with real runs and registry rollback report.
- Document corpus, chunking audit, local retrieval index, Qdrant export/status artifacts, and retrieval comparison reports.
- LangGraph Agent with typed state, local Ollama tool calling, allow-listed tools, deterministic fallback, step limits, tool timeouts, duplicate-call protection, and answer validation.
- Composite-question audit for Agent answers, including per-part coverage and chunk-level grounding checks.
- FastAPI facade over the Agent with `/analyze`, `/chat`, `/health/live`, `/health/ready`, and `/version`.
- Legacy Flask `/predict` and `/explain` preserved and covered by contract tests.
- Dockerfile, `docker-compose.yml`, `.env.example`, README commands, and local FastAPI smoke script.
- Local observability for Command 11: structured JSONL logs, `/metrics`, fixed-bin PSI report, local dashboard, local file-only test alert, Section 7 acceptance report, and interview demo script.

Recent fix:

- `/chat` with `Что ты умеешь?` now returns a clear capabilities answer without tool calls, without LLM call, and without unsupported-message text.

## Key Files Added Or Updated

- `src/agent.py`
- `app/fastapi_app.py`
- `app/api.py`
- `src/scoring_adapter.py`
- `src/retrieval.py`
- `src/document_corpus.py`
- `src/mlflow_tracking.py`
- `scripts/run_agent_rehearsal.py`
- `scripts/audit_composite_agent_answers.py`
- `scripts/smoke_fastapi.py`
- `tests/test_agent.py`
- `tests/test_fastapi_app.py`
- `tests/test_scoring_contract.py`
- `Dockerfile`
- `docker-compose.yml`
- `.env.example`
- `requirements.txt`
- `README.md`
- `monitoring/observability.py`
- `monitoring/fixed_bins_psi.py`
- `scripts/check_fixed_bins_psi.py`
- `scripts/build_monitoring_dashboard.py`
- `scripts/run_command11_acceptance.py`

Main reports:

- `docs/agent_report.md`
- `docs/agent_composite_answer_audit.md`
- `docs/fastapi_local_run.md`
- `docs/mlflow_tracking.md`
- `docs/retrieval_report.md`
- `docs/retrieval_search_comparison.md`
- `docs/scoring_contract.md`
- `docs/command11_acceptance_report.md`
- `docs/interview_demo.md`
- `docs/local_monitoring_dashboard.html`
- `results/command11_acceptance_report.json`
- `results/monitoring_snapshot.json`
- `results/psi_fixed_bins_report.json`
- `results/agent_rehearsal.json`
- `results/agent_composite_answer_audit.json`
- `results/mlflow_rehearsal.json`
- `results/retrieval_eval.json`
- `results/retrieval_search_comparison.json`

## Checks Passed

Latest full test run:

```bash
python -m pytest -q
```

Result:

```text
52 passed, 927 warnings
```

Targeted checks passed:

- `python -m pytest tests\test_fastapi_app.py tests\test_agent.py -q` -> `24 passed`
- `python scripts\smoke_fastapi.py` -> `status=ok`
- Real local HTTP FastAPI smoke on `127.0.0.1:8000`:
  - `/health/live` -> `200`
  - `/health/ready` -> `200`, status `ready`
  - `/version` -> `200`
  - `/analyze` document request -> `intent=documents`, sources `3`
  - `/chat` no-tool request -> no tool calls
  - `/chat` `Что ты умеешь?` -> capabilities answer, no tool calls
  - `/analyze` address without features -> `insufficient_data`
- `python scripts\audit_composite_agent_answers.py` -> `status=ok`
- `python scripts\run_agent_rehearsal.py` -> requested checks `ok`

Important current `/chat` fact:

```text
Question: Что ты умеешь?
HTTP: 200
data_status: ok
tool_selection_source: none
tool_statuses: []
llm.status: skipped
```

Answer:

```text
Я могу помочь с анализом риска кошелька: рассчитать модельный risk score при наличии признаков, объяснить основные факторы модели, найти и процитировать связанные фрагменты документов, отдельно пометить synthetic_demo материалы и явно сказать, если для оценки не хватает данных. Я не подтверждаю мошенничество как факт и не принимаю кредитные решения.
```

## Docker / Compose

Status: `verified on 2026-09-30`.

Docker image was rebuilt and the container was started by the operator after adding the system package `libgomp1` to the `python:3.11-slim` Dockerfile.

Observed HTTP results:

- `/health/ready` -> `200`, status `ready`, checks `model=true`, `scoring_adapter=true`, `retrieval_index=true`, `corpus=true`.
- `/version` -> `200`, status `ok`, `llm_provider=disabled`, model `lightgbm_4fc20542d0f4c2cb`, schema `scoring_explanation_v1`, corpus `demo_corpus_v1`, index `idx_422d98707f81`.

Current Ollama-on-Windows Compose settings:

```text
AGENT_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
AGENT_LLM_TIMEOUT_SECONDS=120
AGENT_LLM_NUM_PREDICT=512
```

`docker-compose.yml` reads `.env` via `env_file` and maps Windows host port `8080` to container port `8000`.

```text
API_PORT=8000
Compose ports: "8080:8000"
Windows API URL: http://127.0.0.1:8080
```

Apply changed `.env` without rebuilding or downloading:

```powershell
docker compose up -d --no-build --force-recreate wallet-risk-api
```

Then verify:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health/ready
Invoke-RestMethod http://127.0.0.1:8080/version
```

Real Docker `/analyze` with Ollama was verified by the operator:

- `execution_mode=langgraph`
- `tool_selection_source=llm_tool_calling`
- `selected_tools=[retrieval]`
- `llm.status=ok`

The first real LLM answer was truncated at `AGENT_LLM_NUM_PREDICT=192`. The Agent now records Ollama `done_reason`, retries once with a concise prompt, and falls back to structured tool output with `validation_errors=["llm_answer_truncated"]` if the retry is still truncated.

Smoke in a second Git Bash terminal:

```bash
curl -s http://127.0.0.1:8080/health/live
curl -s http://127.0.0.1:8080/health/ready
curl -s http://127.0.0.1:8080/version
curl -s -X POST http://127.0.0.1:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Что ты умеешь?"}'
curl -s -X POST http://127.0.0.1:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"question":"Show document sources for wallet 0x01da6f3b20d0540f24d390e28195ad7311516739","wallet_address":"0x01da6f3b20d0540f24d390e28195ad7311516739"}'
```

Stop containers:

```powershell
docker compose down
```

Commands that download images or dependencies:

```bash
docker compose build
docker compose up --build
docker compose pull
```

Inside Docker build, Python dependencies are downloaded by:

```dockerfile
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
```

## Docker Checks and Local Follow-up on 2026-10-01

Already completed real Docker checks, before the current source-code fixes:

- `/health/ready` returned `ready`.
- `/analyze` and `/chat` worked with Ollama through LangGraph.
- Retry after LLM truncation worked; final `done_reason=stop`.
- In `/chat`, scoring was added by `coverage_guard`, not selected by the LLM.
- Without input features, scoring returned `insufficient_data`, `reason=features_required`, `score=null`.

Current source-code fixes added after those Docker checks:

- `insufficient_data` answers now explicitly say the data is insufficient and reject LLM text that claims low risk, no fraud, or absence of fraud.
- Pure methodology questions are routed to methodology retrieval and do not automatically require scoring.
- General methodology retrieval searches methodology documents without substituting wallet observation cards.
- Trace details keep the distinction between LLM-selected tools, tools removed by intent guard, and tools added by `coverage_guard`.
- FastAPI LLM defaults now match `.env`: timeout `120` seconds and `num_predict=512`.

New local tests only; fixed code has not been verified inside the running container yet:

```powershell
python -m pytest tests\test_agent.py tests\test_fastapi_app.py tests\test_retrieval.py
```

Result: `36 passed`.

## What Remains

Immediate remaining item:

- Rebuild/recreate the existing container only when ready to verify the latest source fix for insufficient-data fallback wording inside Docker. Current container was not stopped or rebuilt by Codex.

## Docker / Ollama Follow-up On 2026-10-05

Real Docker/Ollama checks were run against `wallet-risk-api:local` on `127.0.0.1:8080`.

- `/health/ready` -> `ready`, all readiness checks `true`.
- `/version` -> `llm_provider=ollama`, `ollama_model=qwen2.5:7b-instruct`, `model_version=lightgbm_89ef90618da4df9d`.
- `/metrics` -> 79 requests, availability `1.0`, 5xx error rate `0.0`, Agent total `24`, source correctness `12/12`.
- Structured API and Agent logs for 2026-10-05 contain `/analyze`, `/chat`, `/metrics`, readiness/live checks, tool statuses, LLM provider/model/status/done_reason, limitations, and source correctness.
- Real `/analyze` document request completed with Ollama: intent `documents`, retrieval `ok`, `llm.status=ok`, `done_reason=stop`, 3 sources.
- Real `/chat` combined request without features completed with Ollama: `data_status=insufficient_data`, `risk_status=insufficient_data`, `score=null`, limitations include `features_required` and `llm_answer_replaced_for_insufficient_data`.
- Real methodology `/chat` completed with Ollama: intent `general`, retrieval selected, sources `method_feature_dictionary`, `method_scoring_limits`, and `method_retrieval_rules`.
- Real `/analyze` address-without-features response did not treat missing features as no risk; it returned `insufficient_data`, `score=null`, `reason=features_required`.

Issues found after the Docker checks and the operator rebuild:

- The deterministic insufficient-data fallback could still append a stale "LLM unavailable" sentence when the LLM metadata was `status=ok` and the LLM answer was only replaced for safety.
- A methodology request shaped as "wallet risk methodology ... cite sources ... <wallet>" was routed as `combined`, added scoring through `coverage_guard`, and returned wallet observation/synthetic sources instead of methodology documents.
- Fixed locally in `src/agent.py`: deterministic fallback now only adds the LLM-unavailable sentence when the final LLM status is `unavailable` or `error`; methodology/rules/limits/feature-dictionary questions route to `general` unless score/probability computation is explicitly requested.
- Added regression coverage in `tests/test_agent.py`.
- Target tests passed: `python -m pytest tests/test_agent.py tests/test_fastapi_app.py tests/test_observability.py -q` -> `34 passed`.

Follow-up Docker/Ollama verification supplied by the operator:

- Methodology routing is confirmed in Docker/Ollama: methodology/rules/limits/source questions route to methodology retrieval instead of wallet observation replacement.
- `risk.status=not_requested` is confirmed in Docker/Ollama: the answer states that risk assessment was not requested and no longer says the data is insufficient.
- Answer replacement monitoring is confirmed in Docker/Ollama: `/metrics` exposes the replacement counter/rate and definition for `llm_answer_replaced_*` fallback events.
- Current HTTP latency metrics aggregate all API requests, including health/version/metrics and non-Agent endpoints. They are operational API latency metrics, not a separate Agent-only latency estimate.
- Source correctness remains a link/support-structure check: sources have expected IDs and findings are tied to chunks. It does not guarantee that the full natural-language answer is factually true.

Dashboard:

- Refreshed via `python scripts/build_monitoring_dashboard.py`.
- Snapshot: `results/monitoring_snapshot.json`, generated at `2026-10-05T16:56:03Z`.
- Dashboard: `docs/local_monitoring_dashboard.html`.
- Open from `ml-project` with `Start-Process .\docs\local_monitoring_dashboard.html`.

Next stage:

- Check files before commit and open a GitHub PR to run CI.
- Remote deployment remains `not_run`.
- Rollback remains `not_run`.

## GitHub CI On 2026-10-06

GitHub Actions run:
[`37521638488`](https://github.com/aleksandrchepinoga18/ml_bankproject_api_expluatation_rag_agent/actions/runs/37521638488).

- Title: `Make CI tests independent of unpublished dataset`.
- Commit SHA: `6421dfed1ddd1e5243286f95d7ae2bac6f7cc3d7`.
- Overall status: `success`.
- `Unit, Contract, and Corpus Checks`: `success`.
  - `Compile Python modules`: `success`.
  - `Lint with Ruff`: `success`.
  - `Type check with mypy`: `success`.
  - `Check runtime artifacts for full tests`: `success`.
  - `Check corpus and local retrieval index`: `success`.
  - `Run tests with mock or disabled LLM`: `success`.
- `Docker Build Dry Run`: `success`.
- GitHub annotations: 2 warnings about Node.js 20 deprecation for GitHub Actions, plus 1 Ubuntu runner-image notice; none failed the run.
- Remote deployment: `not_run`.
- Rollback verification: `not_run`.

Command 10 status:

- Added GitHub Actions CI workflow for compile syntax check, Ruff lint, mypy type check, runtime artifact gate, corpus/index check, full pytest with disabled/mock LLM, and Docker build dry run.
- Added manual protected `test` environment workflow for dry-run or confirmed GHCR test image publication.
- Server deployment and server-side smoke test are not prepared yet because no test server, access method, secrets, or smoke URL are configured.
- Docker images are tagged by commit SHA; `test-latest` is only a moving test pointer.
- Added rollback runbook for a compatible image/model/corpus/index/manifest set in `docs/cicd_test_deployment.md`; rollback is prepared but not executed or verified.
- Local validation completed without Docker rebuild/pull/push: compile syntax check (`compileall`), runtime artifact check, corpus check, and `python -m pytest -q`.
- Local Ruff and mypy checks were not executed because those tools are not installed locally; CI configuration was prepared in `pyproject.toml` and `requirements-dev.txt`.

Command 11 status:

- Section 7 of `TZ_Scoring_RAG_Agent_Production_v2.md` was found and used as the acceptance source.
- Added structured API and Agent logs without request bodies, secrets, or full document text.
- Added `/metrics` with latency p50/p95, availability, 5xx error rate, Agent error/status rates, versions, and source correctness.
- Source correctness denominator: Agent responses where intent is `documents`, `combined`, or `general`, or responses with sources/findings. Numerator: eligible responses with supported findings, every source carrying `document_id`/`chunk_id`, and no foreign/missing/unsupported source validation errors.
- Source correctness checks source structure and link support, but it does not guarantee factual correctness of the full LLM-generated text.
- Added fixed-reference-bin PSI report and reference bins; drift is diagnostic only and `auto_retrain_started=false`.
- Model quality was not evaluated: label provenance is not confirmed. The label provenance check refused current labels as quality evidence because `label_source` is missing in `monitoring/logs/predictions_with_labels.csv`.
- Added local dashboard and file-only test alert; no external alert was sent.
- Section 7 local acceptance report saved in `docs/command11_acceptance_report.md` and `results/command11_acceptance_report.json`.
- Local final checks: `python -m pytest -q` -> `61 passed`; Section 7 local acceptance -> 7 local scenarios `passed`, rollback scenario `not_run`.
- Mock LLM checks passed and are reported separately from real Ollama checks.
- Real `/analyze` and `/chat` with Ollama were checked earlier, but that was before later Command 10-11 changes and is not verification of the current code.
- New Command 11 code has not been checked in Docker. A rebuild/recreate is required before treating container results as current.
- GitHub CI, remote deployment, rollback, and MLflow snapshot are `not_run`.
- Runtime settings to preserve: internal `API_PORT=8000`, Compose publication `"8080:8000"`, `AGENT_LLM_TIMEOUT_SECONDS=120`, `AGENT_LLM_NUM_PREDICT=512`.

## Continuation Checklist

1. Rebuild/recreate the container with the current code.
2. Verify readiness, version, metrics, `/analyze`, and `/chat` on `http://127.0.0.1:8080`.
3. Verify the local dashboard, structured logs, source-correctness metrics, and file-only alert artifacts.
4. Run GitHub CI and inspect lint, mypy, pytest, corpus, runtime artifact gate, and Docker build dry-run results.
5. Perform a separate test deployment on a real test target, then verify rollback with a compatible image, model, corpus, index, and manifest set.

## Resume Point After Reboot

Start from:

```bash
cd "/c/workplace/GitHub/Projekt_bamk_scor VSCODE/ml_bankproject_api_expluatation_rag_agent/ml-project"
python -m pytest -q
```

Then do one of these:

1. Recreate the existing container with `.env` LLM settings and no rebuild.
2. Verify `/version` reports `llm_provider=ollama`.
3. Verify `/analyze` and `/chat` return `llm.provider=ollama` and `llm.status=ok`.

Do not install or update libraries, programs, models, or Docker images automatically. If a dependency is missing, check version first, provide exact Git Bash commands, and wait for confirmation before dependent actions.

## Local Docker Smoke On 2026-10-10

Current local Docker API was checked at `http://127.0.0.1:8080` after the container had already been rebuilt and started by the operator.

- Report saved: `docs/local_docker_smoke_2026-10-10.md`.
- JSON details saved: `results/local_docker_smoke_2026-10-10.json`.
- `/health/ready` -> `ready`; model, scoring adapter, retrieval index, and corpus checks all `true`.
- `/version` -> `llm_provider=ollama`, `ollama_model=qwen2.5:7b-instruct`, `model_version=lightgbm_89ef90618da4df9d`, `corpus_version=demo_corpus_v1`, `index_version=idx_422d98707f81`.
- `/metrics` -> request availability `1.0`, 5xx error rate `0.0`, source correctness `1.0`, versions present.
- `/analyze` document request -> `intent=documents`, retrieval `ok`, 3 linked sources, `risk.status=not_requested`, `llm.status=ok`, `done_reason=stop`; deterministic replacement was explicitly marked with `llm_answer_replaced_for_not_requested_risk`.
- `/chat` methodology request -> `intent=general`, methodology sources `method_scoring_limits::chunk_001` and `method_retrieval_rules::chunk_001`, no scoring tool, `risk.status=not_requested`, `llm.status=ok`, `done_reason=stop`. The answer text was conservative about risk not being requested, while metadata and sources matched the methodology/no-scoring requirement.
- `/analyze` scoring request with a full feature row -> `risk.status=scored`, score `0.35894730197831315`, scoring/shap/retrieval all `ok`, `llm.status=ok`, `done_reason=stop`.
- No model, corpus, index, training, image publication, remote deployment, rollback, commit, or push was performed.

## Agent Terminal Demo On 2026-10-10

- Added `scripts/demo_agent.py`, a readable Git Bash terminal client for the running FastAPI Agent API. It supports `--base-url`, `--endpoint analyze|chat`, `--question`, `--wallet-address`, `--features-file`, `--trace`, and `--save-json`.
- Added `docs/demo_features_valid.json`, a synthetic/demo full feature row from the existing acceptance example. It is not real wallet evidence.
- Added `docs/agent_terminal_demo.md` with Git Bash commands for document retrieval, methodology retrieval without scoring, scoring with valid features, and combined scoring plus documents.
- Added `docs/agent_terminal_demo_report_2026-10-10.md` with actual Docker/Ollama results and raw JSON locations.
- Existing UI check: FastAPI Swagger UI is available at `http://127.0.0.1:8080/docs`; OpenAPI schema is at `http://127.0.0.1:8080/openapi.json`; legacy `operator_mode.py` is only a Flask `/explain` CLI; `docs/local_monitoring_dashboard.html` is a metrics viewer, not a chat UI. No existing web chat UI for `/chat` was found.
- Live Docker/Ollama demo results: documents -> `intent=documents`, `risk.status=not_requested`, retrieval `ok`; methodology -> `intent=general`, retrieval only, methodology sources, no scoring; scoring -> `intent=score`, `risk.status=scored`, score `0.35894730197831315`; combined -> `intent=combined`, scoring/shap/retrieval all selected and `ok`.
- Code checks: `python -m py_compile scripts/demo_agent.py`, `python scripts/demo_agent.py --help`, and JSON feature-file load all passed.

Final pre-commit demo check:

- Saved methodology response was not acceptable as-is: metadata/sources were correct, but the answer text only said scoring was not requested and listed sources. Fixed locally in `src/agent.py` so methodology deterministic replacement explains retrieved `method_*` documents.
- A live scoring demo also exposed an answer/metadata contradiction: `risk.status=scored` in metadata, but LLM text said "Risk status: Insufficient data." Fixed locally in `src/agent.py` with a scored-answer contradiction guard and deterministic scored fallback.
- `scripts/demo_agent.py` now prints feature-file provenance; `docs/demo_features_valid.json` is clearly marked synthetic/demo, not real wallet data.
- Unavailable API behavior checked with bad port: the CLI prints a clear connection error.
- Final local checks from `.venv`: Ruff passed for `app src scripts monitoring tests`; mypy passed for 51 files; full pytest passed with `67 passed, 929 warnings`.
- Because server code changed after the Docker/Ollama run, Docker must be rebuilt/recreated and the terminal demo smoke repeated before treating container results as final.

Rebuilt Docker rerun:

- Repeated four `scripts/demo_agent.py` scenarios against `http://127.0.0.1:8080` with real Ollama after the operator rebuilt the container.
- Documents passed: `intent=documents`, `risk.status=not_requested`, retrieval `ok`, 3 linked sources, synthetic source marked demo-only/not evidence, limitations included `llm_answer_replaced_for_not_requested_risk`.
- Methodology failed: metadata was correct (`intent=general`, retrieval only, method sources, no scoring), but answer text did not explain methodology by `method_*` sources. This was not accepted as passed.
- Scoring passed: `intent=score`, `risk.status=scored`, structured score `0.35894730197831315`, answer score `0.358947`, no `insufficient_data` contradiction, limitations included `llm_answer_replaced_for_scored_risk_contradiction`.
- Combined passed: `intent=combined`, scoring/shap/retrieval all `ok`, structured score `0.35894730197831315`, answer score `0.3589`, synthetic source marked as not evidence, limitations included `llm_answer_retry_after_truncation` and `synthetic_demo_documents_are_not_independent_evidence`.
- Fixed locally after the failed methodology rerun: added methodology-answer quality guard in `src/agent.py`; vague methodology answers are replaced with deterministic method-source explanations and marked `llm_answer_replaced_for_methodology_answer`.
- Current local checks after the fix: Ruff passed; mypy passed for 51 files; full pytest passed with `68 passed, 929 warnings`.
- Docker must be rebuilt/recreated again and the four terminal demo scenarios repeated before treating the latest server code as verified in Docker.

Second rebuilt Docker rerun after methodology guard:

- Repeated four `scripts/demo_agent.py` scenarios against `http://127.0.0.1:8080` with real Ollama.
- Documents failed content consistency: metadata was `intent=documents`, `risk.status=not_requested`, retrieval `ok`, but answer text said there was "no sufficient data to assess the risk status."
- Methodology passed: answer explains `method_scoring_limits` and `method_retrieval_rules`; limitations include `llm_answer_replaced_for_methodology_answer` and `tool_selection_intent_guard`; no scoring.
- Scoring passed: `intent=score`, `risk.status=scored`, structured score `0.35894730197831315`, answer score `0.3589`, `llm.status=ok`, `done_reason=stop`.
- Combined failed fallback-marker check: scoring/shap/retrieval succeeded and synthetic source was marked not evidence, but `llm.status=unavailable` from Ollama timeout and limitations did not include an explicit LLM fallback/replacement marker.
- No additional guard or server code change was made after this rerun, per instruction to report new defects first.

Local fix after reported Docker defects:

- Generalized not-requested answer validation to reject insufficient-data wording beyond one exact phrase, including "no sufficient data", "cannot determine risk", and related risk-status language.
- Added a shared answer fallback marker helper. All deterministic replacements/fallbacks now include a specific marker and the generic `llm_answer_fallback`; LLM unavailable/error fallbacks keep the real `llm.status` and `llm.error` and add `llm_answer_fallback_after_unavailable` or `llm_answer_fallback_after_error`.
- Added mock-LLM regression tests for the "no sufficient data" not-requested document wording and unavailable LLM fallback marker/status preservation.
- Targeted tests for not_requested, insufficient_data, scored, methodology, truncation, unavailable fallback, and tool timeout passed: `9 passed`.
- Full local checks through `.venv` passed: Ruff clean, mypy clean for 51 files, pytest `70 passed, 929 warnings`.
- Docker was not rebuilt after these fixes.

After VS Code restart / no-rebuild continuation:

- Checked saved reports and results before running more work.
- No active `python.exe` demo client was visible via ordinary `Get-Process`; Docker Desktop/backend and Ollama were running. Docker API and full process command-line inspection were blocked by Windows access permissions in the current shell.
- The running API remained healthy: `/health/ready` returned `ready`; `/version` returned `llm_provider=ollama`, `ollama_model=qwen2.5:7b-instruct`, model `lightgbm_89ef90618da4df9d`, corpus `demo_corpus_v1`, and index `idx_422d98707f81`.
- Continued only the two unfinished terminal demo checks, sequentially, with no parallel Ollama requests and no rebuild.
- Documents rerun passed and was saved to `results/demo_agent_documents_after_vscode_2026-10-10.json`: `intent=documents`, `risk.status=not_requested`, retrieval only, 3 sources, `llm.status=ok`, `done_reason=stop`, limitations include `llm_answer_fallback` and `llm_answer_replaced_for_not_requested_risk`.
- Methodology rerun passed and was saved to `results/demo_agent_methodology_after_vscode_2026-10-10.json`: `intent=general`, `risk.status=not_requested`, retrieval only, 2 method sources, `llm.status=ok`, `done_reason=stop`, limitations include `llm_answer_fallback`, `llm_answer_replaced_for_methodology_answer`, and `tool_selection_intent_guard`. This supersedes the earlier `results/demo_agent_methodology_2026-10-10.json`, which did not include the generic fallback marker.
- Combined rerun passed and was saved to `results/demo_agent_combined_after_vscode_2026-10-10.json`: `intent=combined`, `risk.status=scored`, score `0.35894730197831315`, scoring/shap/retrieval selected, 3 sources, `llm.status=ok`, `done_reason=stop`, limitations include `llm_answer_fallback` and `llm_answer_replaced_for_scored_risk_contradiction`.
- Updated `docs/agent_terminal_demo_report_2026-10-10.md` with the after-restart status.
- Reconciled `docs/local_docker_smoke_2026-10-10.md`, `docs/agent_terminal_demo_report_2026-10-10.md`, and this progress file with the final four-scenario Docker summary:
  - documents: `results/demo_agent_documents_after_vscode_2026-10-10.json`, `llm.status=ok`, fallback markers `llm_answer_fallback`, `llm_answer_replaced_for_not_requested_risk`.
  - methodology: `results/demo_agent_methodology_after_vscode_2026-10-10.json`, `llm.status=ok`, fallback markers `llm_answer_fallback`, `llm_answer_replaced_for_methodology_answer`; this is a `/chat` response and the Agent fields are under `analysis`.
  - scoring: `results/demo_agent_scoring_2026-10-10.json`, `llm.status=ok`, no fallback markers.
  - combined: `results/demo_agent_combined_after_vscode_2026-10-10.json`, `llm.status=ok`, fallback markers `llm_answer_fallback`, `llm_answer_replaced_for_scored_risk_contradiction`.
- No scenario is missing confirmation for the latest running Docker build.
