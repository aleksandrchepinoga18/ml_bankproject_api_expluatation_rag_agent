# Project Progress

Last updated: 2026-10-12.

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

- Agent answer consistency was tightened locally after the 2026-10-12 audit:
  - `run()` now reports `execution_mode=langgraph` when the LangGraph path actually runs, while retaining a `builtin` fallback if LangGraph is unavailable.
  - Methodology/not-requested answers explain retrieved `method_*` documents and do not turn an unrequested risk assessment into an insufficient-risk-data conclusion.
  - Scored-answer validation checks all reported score/threshold/prediction mentions, English and Russian threshold-relation wording, impossible numeric comparisons, and unsupported claims that LightGBM scores are based on retrieval or synthetic notes.
  - Rejected LLM answer text is replaced by deterministic structured output, marked with the specific `llm_answer_replaced_*` reason plus `llm_answer_fallback`, and surfaced through `validation_errors`, trace detail, and `data_status=partial` for rejected/truncated LLM text.
  - Source validation remains a grounding/link check; it does not guarantee the truth of every generated sentence.
  - New Docker/Ollama smoke after these local fixes: `not_run`.
- Docker/Ollama check on the already rebuilt API was run on 2026-10-12 without another rebuild:
  - Report: `docs/docker_real_ollama_agent_check_2026-10-12.md`.
  - Raw JSON: `results/demo_agent_methodology_real_ollama_2026-10-12.json`, `results/demo_agent_scoring_real_ollama_2026-10-12.json`, `results/demo_agent_combined_real_ollama_2026-10-12.json`, plus health/version/metrics JSON files.
  - `/health/ready`, `/version`, and `/metrics` were reachable; `/version` reported `llm_provider=ollama` and `ollama_model=qwen2.5:7b-instruct`.
  - Scoring and combined responses correctly used deterministic fallback for scored-risk contradictions; fallback markers, validation errors, `data_status=partial`, and trace replacement reasons were visible.
  - Methodology metadata/tools were correct and did not claim `insufficient_data`, but a new content issue remains: the answer says "The wallet risk scoring methodology is not requested" even though the methodology was requested.
  - `llm.status=ok` and `validate_answer:ok` were treated only as execution/structural status, not proof of text correctness.
- Local methodology-only fix after the Docker/Ollama check:
  - Explicit methodology questions now skip free-form answer generation and render deterministically from retrieved `method_*` findings.
  - The answer keeps `risk.status=not_requested` without saying methodology was not requested and without treating absent scoring as insufficient data.
  - If no methodology documents are found, the answer says that directly.
  - Answer-generation metadata is truthful: `llm.status=skipped`, with `llm_answer_skipped_for_explicit_methodology`; LLM tool-selection details remain in trace.
  - Regression tests cover both found method documents and missing method documents.
  - Local checks passed: Ruff clean; mypy clean for 51 files; full pytest `83 passed, 929 warnings`.
  - Docker/Ollama rerun after this local fix: `not_run`.
- Docker/Ollama methodology rerun after the operator rebuild:
  - Report updated: `docs/docker_real_ollama_agent_check_2026-10-12.md`.
  - Raw JSON: `results/demo_agent_methodology_after_fix_real_ollama_2026-10-12.json` and `results/docker_api_health_ready_after_methodology_fix_2026-10-12.json`.
  - `/health/ready` returned `ready`.
  - Explicit methodology answer now explains `method_scoring_limits/method_scoring_limits::chunk_001` and `method_retrieval_rules/method_retrieval_rules::chunk_001`.
  - It does not say methodology was not requested and does not claim `insufficient_data`.
  - Scoring did not run; selected tools were `retrieval` only.
  - Answer generation was skipped truthfully with `llm.status=skipped`, marker `llm_answer_skipped_for_explicit_methodology`, tool-selection retained in trace, and `validation_errors=[]`.

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

## Final Daily Status On 2026-10-10

Commit tracked for today's final verification:

- SHA: `20bc8c33371d798c803d6801c426aabe5fa7a5a2`
- Subject: `Add readable agent demo and consistent answer fallbacks`
- GitHub CI: [`38066504722`](https://github.com/aleksandrchepinoga18/ml_bankproject_api_expluatation_rag_agent/actions/runs/38066504722), successful.

Terminal demo:

- Added and documented `scripts/demo_agent.py` for readable terminal demos against the FastAPI Agent API.
- Demo instructions: `docs/agent_terminal_demo.md`.
- Demo report: `docs/agent_terminal_demo_report_2026-10-10.md`.
- Synthetic feature row: `docs/demo_features_valid.json`; explicitly demo-only, not real wallet evidence.
- Saved terminal demo results:
  - documents: `results/demo_agent_documents_after_vscode_2026-10-10.json`.
  - methodology: `results/demo_agent_methodology_after_vscode_2026-10-10.json`.
  - scoring: `results/demo_agent_scoring_2026-10-10.json`.
  - combined: `results/demo_agent_combined_after_vscode_2026-10-10.json`.

Docker/Ollama smoke:

- Local Docker/Ollama smoke report: `docs/local_docker_smoke_2026-10-10.md`.
- API URL: `http://127.0.0.1:8080`.
- `/health/ready` stayed `ready`; model, scoring adapter, retrieval index, and corpus checks were `true`.
- `/version` reported `llm_provider=ollama`, `ollama_model=qwen2.5:7b-instruct`, model `lightgbm_89ef90618da4df9d`, corpus `demo_corpus_v1`, and index `idx_422d98707f81`.
- Final four-scenario Docker/Ollama summary:
  - documents: `intent=documents`, `risk.status=not_requested`, retrieval only, 3 sources, `llm.status=ok`, `done_reason=stop`.
  - methodology: `intent=general`, `risk.status=not_requested`, retrieval only, 2 method sources, `llm.status=ok`, `done_reason=stop`.
  - scoring: `intent=score`, `risk.status=scored`, score `0.35894730197831315`, selected tool `scoring`, `llm.status=ok`, `done_reason=stop`.
  - combined: `intent=combined`, `risk.status=scored`, score `0.35894730197831315`, selected tools `scoring`, `shap`, and `retrieval`, 3 sources, `llm.status=ok`, `done_reason=stop`.
- No model, corpus, index, training, image publication, deployment, rollback, commit, or push was performed during local smoke/demo verification.

Deterministic fallback replacements confirmed:

- documents answer was replaced by deterministic fallback for not-requested risk: `llm_answer_fallback`, `llm_answer_replaced_for_not_requested_risk`.
- methodology answer was replaced by deterministic fallback for methodology answer quality: `llm_answer_fallback`, `llm_answer_replaced_for_methodology_answer`.
- scoring answer had no deterministic fallback marker in the final saved scoring result.
- combined answer was replaced by deterministic fallback for scored-risk contradiction: `llm_answer_fallback`, `llm_answer_replaced_for_scored_risk_contradiction`.

Operational status:

- Server deployment: `not_run`.
- Remote deployment: `not_run`.
- Rollback verification: `not_run`.

## Targeted Agent Review Fixes On 2026-10-13

Scope:

- Fixed `src/agent.py` review defects in one targeted pass while preserving the existing LangGraph router, API response schema, successful scoring/retrieval/fallback scenarios, and public `INTENTS` / `langchain_tool_specs` surface.
- Models, training, corpus, vector index, MLflow, Docker image/container, deployment, rollback, git add/commit/push were not changed or run.

Review report:

| Defect | Fix | Verification | Remaining limitation |
| --- | --- | --- | --- |
| Labeled number extraction could bind `Threshold` to the previous `Score` value across neighboring fields. | `_extract_labeled_numbers` now scans local sentence/line segments and stops at adjacent known field labels while preserving repeated mention checks. | Added regressions for `Score=0.358947. Threshold=0.344386.`, repeated bad threshold, rounded values, prediction, English/Russian relation checks; `tests/test_agent.py` passed. | Label detection remains regex-based for the supported score/threshold/prediction wording. |
| Final validation detected forbidden text such as `proven fraud` but returned the unsafe answer. | `_validate_answer` now records the rejection, replaces the answer with a safe fallback, validates that fallback once, and traces original rejection vs fallback replacement vs final violation. Negated `not proven fraud` is not treated as a positive fraud claim. | Added direct validation and LangGraph regressions for forbidden phrase replacement and negation. | Fallback is single-pass by design to avoid validation/fallback loops. |
| Tool result conversion could fail after `_run_tool` and escape normal tool error handling. | Added explicit scoring/SHAP/retrieval result contract validation and conversion error mapping to honest tool `error` status. Bad scoring skips SHAP, keeps independent retrieval, and completes validation. | Added scoring contract regression for `aggregation=None`; existing scoring error and retrieval partial tests still pass. | Unknown programming errors inside tool execution still surface as tool errors rather than successful results. |
| Builtin and LangGraph small-step budgets behaved differently and could report `not_requested` after requested scoring was blocked. | Builtin now mirrors LangGraph safe completion after budget exhaustion: no more tool/LLM calls, safe answer still built and validated, requested-but-unrun scoring becomes `risk.status=error`. Constructor validates budget/timeout settings. | Added/updated small-budget and config validation tests for builtin and LangGraph. | Small budget responses are intentionally partial because requested scoring did not run. |
| Truncation retry fallback could misreport the retry failure reason. | Retry path now preserves retry `unavailable/error` status and marker separately from repeated truncation; empty `ok` response gets an explicit fallback reason. | Added regressions for successful retry, repeated truncation, retry unavailable, and empty `ok` response. | A truncated first answer is still recorded as `llm_answer_retry_after_truncation` before retry outcome. |
| `_run_tool` created one executor per call and timeout did not stop already-running work. | Agent now owns a bounded executor plus semaphore; saturation returns `tool_queue_saturated`, timeout is reported as an await timeout, and the slot is released only when the background task actually finishes. | Added regression that a timed-out task holds the slot, a second task is predictably rejected, and late completion does not mutate returned state. | Timed-out Python work is not force-killed; it is bounded and ignored after timeout. |
| Multi-row aggregated score could be explained with SHAP from only the first row. | SHAP is skipped for multi-row aggregate scoring with explicit `aggregated_score_shap_not_supported` status/limitation; one-row SHAP remains unchanged. | Added multi-row scoring regression; existing one-row combined/explain tests pass. | Aggregate SHAP explanation is not implemented for `max_row_probability`; unsupported explanation is explicitly skipped. |
| Minimal cleanup requested. | Removed the redundant `try/except ValidationError: raise` in `validate_agent_request` and updated the prompt wording from deterministic-only routing to selected-and-validated router wording. | Ruff, mypy, and full pytest passed. | No broader graph cleanup, re-planning, checkpointer, reducers, or web UI changes were made. |

Local checks through existing `.venv`:

- `.venv\Scripts\python.exe -m pytest tests/test_agent.py -q` -> `54 passed, 922 warnings`.
- `.venv\Scripts\python.exe -m ruff check app src scripts monitoring tests` -> `All checks passed!`.
- `.venv\Scripts\python.exe -m mypy app src scripts monitoring tests` -> `Success: no issues found in 51 source files` plus one note about unchecked untyped test function bodies.
- `.venv\Scripts\python.exe -m pytest -q` -> `93 passed, 929 warnings`.

Docker/Ollama smoke after these changes:

- Status: `not_run`.
- Required next action: run the existing Docker/Ollama smoke/demo flow against a container built from these code changes, then verify `/health/ready`, `/version`, document-only, methodology, scoring, combined, fallback markers, real `llm.status/error`, and multi-row SHAP-skip behavior.

## Docker API Review Smoke On 2026-10-13

Checked already rebuilt Docker API at `http://127.0.0.1:8080`; Docker was not rebuilt and the container was not stopped.
Code, models, corpus, index, and MLflow were not changed in this smoke pass.

Saved raw JSON:

- `results/docker_api_health_ready_review_2026-10-13.json`
- `results/docker_api_version_review_2026-10-13.json`
- `results/docker_api_metrics_review_2026-10-13.json`
- `results/demo_agent_documents_review_2026-10-13.json`
- `results/demo_agent_methodology_review_2026-10-13.json`
- `results/demo_agent_scoring_review_2026-10-13.json`
- `results/demo_agent_combined_review_2026-10-13.json`
- `results/demo_agent_multirow_review_2026-10-13.json`

Short smoke report:

- Report file: `docs/docker_api_review_smoke_2026-10-13.md`.
- `/health/ready`: `ready`; model, scoring adapter, retrieval index, corpus all `true`.
- `/version`: Ollama provider with `qwen2.5:7b-instruct`; model `lightgbm_89ef90618da4df9d`; schema `scoring_explanation_v1`; corpus `demo_corpus_v1`; index `idx_422d98707f81`.
- `/metrics`: availability `1.0`; 5xx error rate `0.0`; Agent total `140`; answer replacement rate `0.114286`; source correctness `64/64`. The answer replacement metric is a fallback/replacement count, not proof of LLM answer quality.
- Documents scenario: `intent=documents`, `risk.status=not_requested`, retrieval `ok`, 3 sources, `llm.status=ok`, `done_reason=stop`, no fallback marker.
- Methodology scenario: `intent=general`, retrieval `ok`, 2 method sources, final answer deterministic; `llm.status=skipped`, `llm.error=answer_generation_skipped_for_explicit_methodology`.
- Scoring scenario: score `0.35894730197831315`, threshold `0.34438552759270996`, prediction `1`; original LLM text rejected with `llm_answer_replaced_for_scored_risk_contradiction`; final answer is deterministic fallback with `llm_answer_fallback`.
- Combined scenario: scoring/SHAP/retrieval all `ok`; same score/threshold/prediction; 3 sources; LLM retry path ended truncated with `llm.status=ok`, `done_reason=length`, markers `llm_answer_retry_after_truncation`, `llm_answer_fallback_after_truncation`, `llm_answer_fallback`.
- Multi-row scoring: aggregate score `0.3615507526626647`, threshold `0.34438552759270996`, prediction `1`, `aggregation.records_count=2`, representative row `review-row-a`; SHAP first-row explanation was not returned as aggregate explanation because `risk.top_features=[]` and tool `shap=skipped(aggregated_score_shap_not_supported)`.

Defect preserved as concrete example:

- `results/demo_agent_multirow_review_2026-10-13.json` final answer includes an LLM explanation sentence claiming `Given the methodology documents retrieved...` while the same response has `sources=[]`, no retrieval tool status, and selected tools only `scoring, shap`.
- This is a final-answer source consistency defect in the multi-row scenario. The aggregate score and explicit SHAP skip are correct, but the LLM-generated explanatory tail refers to methodology documents that were not retrieved.

## Source-Consistency Final Validation Fix On 2026-10-13

Scope:

- Fixed the final-answer source consistency defect found in `results/demo_agent_multirow_review_2026-10-13.json`.
- The fix is in the shared final validation layer for all intents, not in the multi-row path.
- LangGraph topology, scoring, multi-row SHAP policy, models, corpus, index, MLflow, Docker, commit, and push were not changed or run.

Fix summary:

- `_validate_answer` now rejects final answers that claim documents/sources/chunks were actually retrieved, found, used, ranked, cited, returned, or linked when `sources` is empty.
- The guard distinguishes actual source claims from capability descriptions such as "the RAG layer can retrieve documents" and from explicit no-source statements such as "documents were not found".
- On violation, the original answer is replaced with deterministic fallback, `llm_answer_replaced_for_source_consistency` and generic `llm_answer_fallback` markers are preserved, real `llm.status/error` remains unchanged, and the fallback is checked once for final-answer violations.

Regression coverage added:

- Saved multi-row defect shape without depending on the local `results` file: scoring + SHAP selected, retrieval not called, sources empty, LLM claims methodology documents were retrieved.
- Retrieval not called.
- Retrieval returned an empty list.
- Retrieval ended with an error.
- Real sources are present, so source claims are allowed.
- RAG capability wording without actual-source claims is allowed.

Validation:

- `.venv\Scripts\python.exe -m pytest <source-consistency regression tests> -q` -> `5 passed, 922 warnings`.
- `.venv\Scripts\python.exe -m pytest tests/test_agent.py -q` -> `59 passed, 922 warnings`.
- `.venv\Scripts\python.exe -m ruff check app src scripts monitoring tests` -> `All checks passed!`.
- `.venv\Scripts\python.exe -m mypy app src scripts monitoring tests` -> `Success: no issues found in 51 source files` plus the existing note about unchecked untyped test function bodies.
- `.venv\Scripts\python.exe -m pytest -q` -> `98 passed, 929 warnings`.

Docker follow-up:

- Docker smoke after this code fix: `not_run`.
- Next Docker check should rerun the multi-row scenario that previously produced `Given the methodology documents retrieved...` and confirm the final answer is deterministic fallback with `llm_answer_replaced_for_source_consistency` when retrieval is absent and `sources=[]`.

## Docker Source-Consistency Rerun On 2026-10-13

Checked the already rebuilt Docker API at `http://127.0.0.1:8080`; Docker was not rebuilt and code was not changed.
Commit/push were not performed.

Saved raw JSON:

- `results/docker_api_health_ready_source_consistency_2026-10-13.json`
- `results/demo_agent_multirow_source_consistency_2026-10-13.json`
- `results/demo_agent_documents_source_consistency_2026-10-13.json`

Updated smoke report:

- `docs/docker_api_review_smoke_2026-10-13.md`

Results:

- `/health/ready`: `ready`; model, scoring adapter, retrieval index, corpus all `true`.
- Multi-row scenario: `intent=explain`, `risk.status=scored`, aggregate score `0.3615507526626647`, threshold `0.34438552759270996`, prediction `1`, `aggregation.records_count=2`, representative row `review-row-a`.
- Multi-row SHAP: explicit skip preserved: `shap=skipped(aggregated_score_shap_not_supported)`, `risk.top_features=[]`.
- Multi-row sources: `sources=[]`; final answer did not claim retrieved/found/used documents. It was accepted LLM text, not deterministic fallback, so source-consistency markers were not expected in this run.
- Documents scenario: `intent=documents`, retrieval `ok`, 3 wallet-linked sources, `llm.status=ok`, `done_reason=stop`, no validation errors. Correct references to actual sources were accepted and not blocked.
- If a future run produces a source claim with `sources=[]`, expected fallback markers are `llm_answer_replaced_for_source_consistency` and `llm_answer_fallback`.

## Intent Routing Priority Fix On 2026-10-13

Scope:

- Fixed intent routing for a concrete wallet assessment request containing the Russian word `ограничения`.
- Graph, API schema, scoring, models, data, corpus, index, MLflow, Docker, commit, and push were not changed or run.

Fix summary:

- Added a shared `asks_specific_wallet_assessment` helper and aligned `classify_intent` with `is_explicit_methodology_question`.
- A concrete wallet-risk explanation request such as `Объясни оценку риска этого кошелька простыми словами: результат модели, сравнение с порогом, основные факторы и ограничения.` now routes to `explain` and runs `scoring, shap` when features are supplied.
- Explicit methodology/rules/feature-dictionary/retrieval-rules questions still route to `general` even when `wallet_address` and `features` are supplied.
- Documents-only routing remains `documents` and retrieval-only, including when features are present.

Regression coverage added:

- Russian concrete wallet-risk explanation with `ограничения` -> `intent=explain`, tools `scoring, shap`.
- General methodology question with wallet/features -> `intent=general`, tool `retrieval` only.
- Documents-only question with wallet/features -> `intent=documents`, tool `retrieval` only.

Validation:

- `.venv\Scripts\python.exe -m pytest tests/test_agent.py::test_russian_wallet_risk_explanation_with_limits_routes_to_scoring_and_shap tests/test_agent.py::test_methodology_with_wallet_and_features_stays_general tests/test_agent.py::test_documents_only_with_features_still_uses_retrieval_only -q` -> `3 passed, 922 warnings`.
- `.venv\Scripts\python.exe -m ruff check app src scripts monitoring tests` -> `All checks passed!`.
- `.venv\Scripts\python.exe -m mypy app src scripts monitoring tests` -> `Success: no issues found in 51 source files` plus the existing unchecked untyped test note.
- `.venv\Scripts\python.exe -m pytest -q` -> `101 passed, 929 warnings`.

## Readable Scored Fallback And Demo Statuses On 2026-10-13

Scope:

- Completed the user-facing deterministic scored/explain fallback without changing graph, scoring, models, corpus, index, MLflow, Docker, public API schema, commit, or push.
- Checked the previous response shape: rejected LLM text is not present in normal public responses/raw JSON, so it was not reconstructed by guess.

Fix summary:

- Deterministic scored fallback now starts with a plain-language score-vs-threshold result, then lists SHAP factors, then states limitations.
- The fallback keeps exact structured `risk.score`, `risk.threshold`, `prediction`, `model_version`, `schema_version`, original feature names, values, and SHAP impacts in the structured response.
- The fallback does not treat score as a calibrated probability, does not present SHAP as causal proof, and does not add approve/decline decisions.
- Existing feature-dictionary suffix notes are used for `_count` and `_timestamp` fields without inventing units or causal explanations.
- `scripts/demo_agent.py` now distinguishes rejected original LLM text with safe deterministic fallback from a final answer that still failed validation, while preserving `validation_errors` and replacement reasons.
- Demo feature provenance is preserved by the client in an adjacent `*.client_context.json` sidecar when `--save-json` and a feature-file note are used, leaving the raw API JSON unchanged.
- Added opt-in local diagnostics via `AGENT_REJECTED_LLM_DIAGNOSTICS_DIR`; when enabled, the next rejected LLM answer is saved locally with the exact text, replacement reason, request id, intent, and real `llm.status/error`. It is disabled by default and does not write full text to normal production logs.

Regression coverage added:

- Readable scored fallback includes score/threshold relation, SHAP factor names, non-causal SHAP limitation, and no rejected approve/decline text.
- Opt-in rejected-LLM diagnostic writes the exact rejected text and reason.
- Demo client status separates safe fallback from real final validation failure.
- Demo client sidecar preserves synthetic/demo feature warnings.

Validation:

- `.venv\Scripts\python.exe -m pytest tests/test_demo_agent.py tests/test_agent.py::test_scored_fallback_is_readable_and_preserves_feature_names tests/test_agent.py::test_rejected_llm_text_diagnostic_is_opt_in -q` -> `6 passed, 922 warnings`.
- `.venv\Scripts\python.exe -m ruff check app src scripts monitoring tests` -> `All checks passed!`.
- `.venv\Scripts\python.exe -m mypy app src scripts monitoring tests` -> `Success: no issues found in 52 source files` plus the existing unchecked untyped test note.
- `.venv\Scripts\python.exe -m pytest -q` -> `107 passed, 929 warnings`.

## Final Review Defects In Agent Output Validation On 2026-10-13

Scope:

- Fixed only the three final-review defects in `src/agent.py` and `scripts/demo_agent.py`.
- Graph, models, data, MLflow, API schema, Docker, commit, and push were not changed or run.

Fix summary:

- `_extract_labeled_numbers` no longer binds one number to two neighboring fields. Correct forms such as `Score=0.358947, threshold=0.344386, prediction=1.` are accepted, while repeated genuinely wrong score/threshold mentions are still rejected.
- Mojibake Russian field labels in `_extract_labeled_numbers` were replaced with valid Russian labels for `скор`, `оценка`, `порог`, `предсказание`, and `класс`; Russian relation checks also preserve explicit above/below/equal phrase handling.
- Source-consistency validation now checks source claims per sentence/segment. A valid negation such as `No documents were found` no longer suppresses a separate unsupported claim such as `According to retrieved documents...`; plain no-source statements and RAG capability descriptions remain allowed.
- After text fallback, validation now rechecks structured `sources` and `document_findings`; remaining structural errors produce `final_answer_violation` rather than `safe_fallback_validated`. Duplicate validation errors are not re-added.
- `scripts/demo_agent.py` gives final validation errors priority in `summarize_state` and does not hide remaining structural source/finding errors just because fallback replacement occurred.

Regression coverage added:

- Correct comma-separated score/threshold/prediction fields.
- Correct Russian score/threshold/class field labels.
- Repeated wrong score after correct fields.
- Mixed no-source negation plus a separate positive retrieved-documents claim.
- Plain `No documents were found` and RAG capability text stay allowed.
- Structured `document_findings` errors after fallback remain final validation failures.
- Demo client does not hide structural errors after fallback.

Validation:

- `.venv\Scripts\python.exe -m pytest tests/test_agent.py::test_scored_answer_labeled_numbers_do_not_cross_neighbor_fields tests/test_agent.py::test_scored_answer_labeled_numbers_accept_comma_separated_fields tests/test_agent.py::test_scored_answer_labeled_numbers_accept_russian_labels tests/test_agent.py::test_scored_answer_replacement_checks_repeated_threshold_mentions tests/test_agent.py::test_scored_answer_replacement_when_repeated_score_is_wrong_after_correct_fields tests/test_agent.py::test_source_consistency_checks_positive_claims_separately_from_negation tests/test_agent.py::test_source_consistency_allows_plain_no_documents_found tests/test_agent.py::test_source_consistency_allows_rag_capability_description_without_sources tests/test_agent.py::test_fallback_validation_keeps_structural_source_errors_failed tests/test_demo_agent.py -q` -> `14 passed, 922 warnings`.
- `.venv\Scripts\python.exe -m ruff check app src scripts monitoring tests` -> `All checks passed!`.
- `.venv\Scripts\python.exe -m mypy app src scripts monitoring tests` -> `Success: no issues found in 52 source files` plus the existing unchecked untyped test note.
- `.venv\Scripts\python.exe -m pytest -q` -> `114 passed, 929 warnings`.

## Docker Final Review Rerun On 2026-10-13

Scope:

- Checked the already rebuilt Docker API at `http://127.0.0.1:8080`.
- Docker was not rebuilt, code was not changed, and commit/push were not performed.

Raw JSON saved:

- `results/docker_api_health_ready_final_review_2026-10-13.json`
- `results/demo_agent_russian_explain_final_review_2026-10-13.json`
- `results/demo_agent_russian_explain_final_review_2026-10-13.json.client_context.json`
- `results/demo_agent_documents_final_review_2026-10-13.json`

Results:

- `/health/ready`: `ready`; model, scoring adapter, retrieval index, and corpus checks all `true`.
- Russian explain request: `intent=explain`, tools `scoring, shap`, `risk.status=scored`, score `0.35894730197831315`, threshold `0.34438552759270996`, prediction `1`, relation above threshold, `llm.status=ok`, `done_reason=stop`, `validation_errors=[]`, `sources=[]`.
- Russian explain answer: deterministic prefix is readable and does not include `score=...`, `threshold=...`, `prediction=...`, `model_version`, `schema_version`, or `dictionary_note` key-value noise. It was accepted LLM text with deterministic prefix, not safe fallback.
- Demo feature warning was preserved in the client sidecar JSON.
- Documents request: `intent=documents`, tool `retrieval`, `risk.status=not_requested`, retrieval `ok`, 3 wallet-linked sources, `llm.status=ok`, `done_reason=stop`, `validation_errors=[]`.
- Documents answer: correct references to actual returned sources were allowed; no source-consistency fallback was triggered.
- Safe fallback vs final errors: neither response had `llm_answer_*` fallback markers, `final_answer_replaced_after_validation`, or `final_answer_violation`; these were successful accepted LLM responses with no remaining final validation errors.

Defect preserved:

- `results/demo_agent_russian_explain_final_review_2026-10-13.json` contains an accepted LLM explanation tail that describes SHAP impacts as percentages (`45.86%`, `25.13%`, `19.99%`).
- The deterministic prefix is readable and numerically consistent, but the accepted LLM tail violates the desired SHAP presentation rule. This is not a safe-fallback result and should not be counted as evidence of fallback quality.

Smoke report updated:

- `docs/docker_api_review_smoke_2026-10-13.md`

## SHAP Unit Interpretation Guard On 2026-10-13

Scope:

- Fixed the SHAP interpretation defect reproduced by `results/demo_agent_russian_explain_final_review_2026-10-13.json`.
- Graph, model, data, API schema, Docker, commit, and push were not changed or run.

Fix summary:

- Confirmed the configured explainer reports `model_output=raw`; SHAP impacts from `shap.TreeExplainer(model)` are raw LightGBM output/margin contributions, not percentages, probabilities, or percentage points.
- Deterministic scored fallback now prints each SHAP impact with direction and unit: raw output/margin LightGBM.
- Prompt instructions now tell the LLM not to convert SHAP impacts to percentages/probabilities/percentage points unless a verified transform is provided.
- Added a scored-answer guard that rejects LLM explanations presenting SHAP impacts as percentages, probabilities, or percentage points, while still allowing `%` for ordinary percentage-valued features.
- On violation, the answer is replaced with deterministic fallback and existing fallback markers/reasons are preserved via `llm_answer_replaced_for_shap_unit_misinterpretation` plus `llm_answer_fallback`.

Regression coverage added:

- Saved Russian explain defect shape: SHAP impacts described as `45.86%`, `25.13%`, `19.99%` is rejected and replaced with fallback.
- Allowed percentage feature value: `utilization_percent feature value is 45%` is accepted when it is not presented as a SHAP impact.

Validation:

- `.venv\Scripts\python.exe -c "from src.inference import load_explainer_and_metadata; e,*_=load_explainer_and_metadata(); print(getattr(getattr(e,'model',None),'model_output',None)); print(getattr(e,'model_output',None))"` -> `raw`, `raw`.
- `.venv\Scripts\python.exe -m pytest tests/test_agent.py::test_saved_russian_explain_shap_percentages_are_replaced tests/test_agent.py::test_shap_guard_allows_percentage_feature_value_not_shap_percent tests/test_agent.py::test_scored_fallback_is_readable_and_preserves_feature_names -q` -> `3 passed, 922 warnings`.
- `.venv\Scripts\python.exe -m ruff check app src scripts monitoring tests` -> `All checks passed!`.
- `.venv\Scripts\python.exe -m mypy app src scripts monitoring tests` -> `Success: no issues found in 52 source files` plus the existing unchecked untyped test note.
- `.venv\Scripts\python.exe -m pytest -q` -> `116 passed, 929 warnings`.

## Current Handoff State On 2026-10-13

Confirmed status:

- Local validation remains green: full pytest `116 passed`; Ruff and mypy passed.
- The latest Docker Russian explain rerun completed as a safe deterministic fallback, not as accepted LLM text.
- Raw result: `results/demo_agent_explain_shap_units_2026-10-13.json`.
- Client warning sidecar: `results/demo_agent_explain_shap_units_2026-10-13.json.client_context.json`.
- Fallback reason is `llm_answer_replaced_for_scored_risk_contradiction`; limitations include `llm_answer_fallback`, `llm_answer_replaced_for_scored_risk_contradiction`, and `tool_selection_coverage_guard`.
- `llm.status=ok`, `llm.error=null`, `done_reason=stop`; this confirms the source LLM generation completed but was rejected.
- Structured numbers are consistent: score `0.35894730197831315`, threshold `0.34438552759270996`, prediction `1`, relation above threshold.
- The returned user answer reports SHAP in `raw output/margin LightGBM`, not as percentages; `%` is absent from the final answer.
- The user-facing explanation remains too technical because it exposes raw feature names and raw SHAP units in the main answer.

Important distinction:

- Do not describe this Docker explain rerun as successful acceptance of LLM text.
- Do not treat fallback as evidence of LLM output quality.
- This run is a safety success for fallback validation, but the user-facing copy still needs a separate readability pass.

Next task:

- Check the feature dictionary and the actual meaning of features before replacing raw names with understandable labels.
- Separate the plain-language user explanation from technical diagnostics.
- Investigate the exact reason the LLM text was rejected.
- Do not assume `rejected_text` exists in the saved raw API response. Check first. If local diagnostics were not enabled for that run, an explicit diagnostic reproduction will be required.

Operational status:

- Server deployment: `not_run`.
- Rollback verification: `not_run`.

## Readable Fallback Copy Polish On 2026-10-13

Scope:

- Polished only the deterministic scored/explain fallback copy and demo-client diagnostics.
- Docker, graph, scoring, models, corpus, index, MLflow, public API schema, commit, and push were not changed or run.

Fix summary:

- Main fallback answer no longer exposes `relation=...`, `prediction=...`, `model_version=...`, `schema_version=...`, `dictionary_note=...`, or `score=.../threshold=...` key-value formatting.
- The first paragraph is now human-readable: score and threshold are shown as rounded display numbers with comma decimal separator, followed by plain-language above/below/equal threshold text.
- Exact score, threshold, prediction, model version, and schema version remain in the structured `risk` response.
- SHAP factors keep original feature names unless a verified feature label exists; no unverified human expansion is invented.
- SHAP impact is described only as model contribution, not as percent, probability, or causal proof.
- `scripts/demo_agent.py` now prints diagnostic `relation` plus feature dictionary notes in the LightGBM/top-features diagnostic sections, while keeping demo-warning sidecar behavior.
- Fallback markers and validation/replacement reporting are preserved.

Validation:

- `.venv\Scripts\python.exe -m pytest tests/test_agent.py::test_scored_answer_replacement_does_not_contradict_metadata tests/test_agent.py::test_saved_scoring_demo_contradiction_is_replaced tests/test_agent.py::test_scored_fallback_is_readable_and_preserves_feature_names tests/test_agent.py::test_scored_answer_replacement_when_above_threshold_text_says_below tests/test_agent.py::test_scored_answer_replacement_checks_all_score_mentions tests/test_agent.py::test_scored_answer_replacement_when_prediction_contradicts_structured_risk tests/test_agent.py::test_unavailable_llm_answer_fallback_is_marked_and_status_preserved tests/test_demo_agent.py -q` -> `11 passed, 922 warnings`.
- `.venv\Scripts\python.exe -m ruff check app src scripts monitoring tests` -> `All checks passed!`.
- `.venv\Scripts\python.exe -m mypy app src scripts monitoring tests` -> `Success: no issues found in 52 source files` plus the existing unchecked untyped test note.
- `.venv\Scripts\python.exe -m pytest -q` -> `107 passed, 929 warnings`.

## User Output And Rejection Diagnostics On 2026-10-15

Scope:

- Changed only user answer presentation, demo CLI display, local rejection diagnostics, targeted tests and documentation.
- Graph, scoring, models, data, corpus, index and MLflow were not changed. Public API schema and raw JSON saving remain unchanged.
- Nothing was installed; commit/push were not performed.

Changes:

- Default demo output contains question, declared feature-origin note when supplied, concise answer and request time. A brief fallback/final-validation notice remains visible.
- Added --details for full structured risk, selected SHAP values and raw output/margin units, sources, versions, LLM and validation diagnostics. --trace remains independent.
- Scored answer reports score/threshold and the class from structured prediction; equality to threshold maps to class 1 under the existing binary scoring contract.
- Positive SHAP contributions are explicitly only part of the explanation. Unknown feature meanings, units and zero semantics are not expanded. No unconditional demo provenance is added to arbitrary inputs.
- Accepted LLM answers contain one explanation without a deterministic prefix or technical LLM tail. A presentation guard and aligned prompt enforce the concise Russian format while preserving numeric, source, SHAP-unit and final safety checks. Nonconforming text uses the explicitly marked safe fallback; LLM generation is still enabled.
- Opt-in local rejection files now contain rejection_detail with rule, field, expected and detected value/fragment. No API fields were added. Write failure logs request ID, reason and exception type without rejected text or exception message.
- Added target coverage for display flag combinations, unchanged raw JSON, real-input provenance, accepted concise LLM output, prediction at threshold, unknown semantics, numeric subreasons, diagnostics disabled by default and safe write-error logging. Existing copy assertions were updated for the new presentation contract.
- Expected default and --details examples: docs/user_output_examples.md. These are illustrative layouts based on the saved result, not a new API run.

Validation:

- Targeted tests: not_run (user instruction).
- Full tests: not_run (user instruction).
- Ruff: not_run (user instruction).
- mypy: not_run (user instruction).
- Docker/API requests: not_run (user instruction).
- Manual source/diff review only; no inference or training was run.

## Free Scored Explanations Without Phrase Matching On 2026-10-15

Scope:

- Removed _scored_presentation_issue and the scored_presentation fallback branch entirely, including mandatory wording, technical/name/demo word bans and formatting-based rejection.
- Graph, scoring, model, data, corpus, index, API schema and MLflow were not changed. No install, commit or push.

Responsibility split:

- Code composes the authoritative score, threshold, predicted class and comparison, then adds its own limitation. Numeric display preserves the supplied float representation without rounding it to three decimals.
- LLM supplies a free explanation. Missing numbers, missing warnings, synonyms, Markdown and technical terms are not rejection conditions. Prompt ownership instructions replace mandatory quoted phrases.
- Code removes echoes of its own summary/limitation for display only. Final safety/source evidence is preserved before display cleanup; formatting cleanup cannot hide a known violation.
- Demo source warning is added from explicit synthetic_demo source metadata. Feature-input provenance remains a client-owned note from the declared feature file; arbitrary inputs are not labeled demo.
- Unknown feature definitions, feature units and zero semantics are provided as LLM context and shown in --details. No deposit/zero word bans remain.

Bounded factual checks:

- Kept rejection/fallback for explicitly wrong score, threshold, numeric or named predicted class, threshold comparison, false source claims, score attributed to documents and wrong SHAP units.
- Removed generic missing-data triggers from scored checks when they do not explicitly deny scoring. Tightened labeled-number extraction to avoid treating an unrelated nearby number as a claimed score/class.
- Added bounded handling of negated numeric claims, comparisons, document attribution, SHAP units and final forbidden phrases. Mentioning SHAP class 1 as an explanation target is not by itself a predicted-class claim.
- These are limited regex-based checks of explicit forms, not a complete truth verifier. No claim of complete semantic validation or measured acceptance rate is made.
- Opt-in local rejection diagnostics and write-failure operator warning remain available. Historical saved scored_presentation responses are unchanged.

Tests and review:

- Added tests/test_scored_explanations.py with varied scores, thresholds and feature names: free wording and omitted boilerplate, correct negations, unknown zero meaning, actual contradictions, code-owned summary/limitation deduplication, conditional demo-source warning and safety before rendering cleanup.
- Updated existing acceptance assertions that previously expected stylistic fallback. Added a feature-semantics details assertion.
- Updated expected-output documentation; examples are illustrative, not new API results.
- Targeted/full tests: not_run (user instruction).
- Ruff: not_run (user instruction).
- mypy: not_run (user instruction).
- Docker/API requests: not_run (user instruction).
- Manual source/diff review only. Turn-specific diff: docs/scored_explanation_changes_2026-10-15.patch.

## Feature Presence, Binary Classes And Rule Evidence On 2026-10-15

Diagnosis of results/demo_agent_explain_without_style_guard_2026-10-15.json:

- Request 9c7df72638784ffeab647fea0e5dcb8a accepted an LLM explanation with absent-value claims despite three numeric top_features.value zeros. Unknown definitions/units had not been separated clearly enough from data availability or SHAP attribution.
- It also called the result medium risk despite prediction=1 and the binary scoring contract. Existing checks covered numbers and selected binary labels but not an explicitly invented middle category.
- risky_tx_count=30 and wallet_age=10 come from docs/demo_features_valid.json, whose note explicitly declares synthetic/demo input. The note previously stayed in the client display/sidecar and never reached the LLM.
- rule_triggers were generated by get_rule_triggers in src/inference.py, called from the agent SHAP tool and the prediction API, not invented by the LLM. The count predicate is >20. The age predicate is <604800 and its previous message divided by 86400, assuming seconds without a verified source-unit contract. Data preparation reads precomputed parquet fields; the dictionary does not establish wallet_age units. Legacy /explain duplicated the same age conversion.

Changes:

- Added src/explanation_contract.py for internal feature-presence/attribution evidence and rule definitions. Numeric zero remains present_numeric, independently of unknown field meaning. Original input availability is separate from the post-preprocessing model value, including an imputed model zero.
- LLM context now separates feature_evidence, classification_contract, rule_evidence and input_evidence. It explicitly describes binary classes, no additional configured risk levels, unknown feature units/zero meanings and rule predicates as caller-input checks rather than independently verified wallet history.
- Added bounded checks of explicit named-feature absence claims and risk-category assignments, with negation/uncertainty handling. They apply to arbitrary feature names and numeric values; no mandatory wording, technical-word bans or per-demo-feature rejection rules were introduced. Existing numeric, source and SHAP-unit checks remain.
- Kept numeric rule predicates and identifiers unchanged. Reworded count/age rule messages to report input checks, removed unverified age-to-days claims, and shared that interpretation with legacy /explain without rewriting saved explanation files. Public rule keys and response schemas remain unchanged.
- Demo client forwards a declared provenance note through the existing question/message field, while displaying the original user question. No new API request field was added. Inputs without a note are not labeled demo. Details explain that numeric zero is a value and rule names are not proof of time units/events.
- Updated the old age-diagnostic expectation: an age-in-days claim is now unverified, not a verified conversion yielding zero days.

Regression coverage added/updated:

- Numeric zero/nonzero vs null/NaN, original missing input vs model/imputed zero, and SHAP vs feature units.
- Absent-value contradictions for different feature names; acceptable unknown-zero-meaning warnings and negations.
- Unsupported medium/moderate category assertions across different scores/predictions; acceptable negations and contribution targets.
- Demo input origin, numeric rule boundaries, legacy rule-message rendering, rule-context provenance, client provenance forwarding and no invented demo origin.
- These checks cover explicit forms and remain heuristics, not complete natural-language truth validation.

Validation:

- Targeted and full tests: not_run (user instruction).
- Ruff: not_run (user instruction).
- mypy: not_run (user instruction).
- Docker, API requests and Ollama requests: not_run (user instruction).
- Manual code and diff review only. Graph, model, training, dataset, corpus, index, MLflow and API schema were not changed. No commit/push.

Commands for the user, from ml-project (not executed):

- .\.venv\Scripts\python.exe -m pytest tests/test_explanation_contract.py tests/test_scored_explanations.py tests/test_agent.py tests/test_demo_agent.py tests/test_scoring_contract.py -q
- .\.venv\Scripts\python.exe -m ruff check app src scripts monitoring tests
- .\.venv\Scripts\python.exe -m mypy app src scripts monitoring tests

## Evidence Scope Clarification On 2026-10-15

- Latest saved live result: results/demo_agent_explanation_contract_2026-10-15.json, request 18c26b5163e84c3fbc1e2dc54372e549. Technically successful (tools and LLM ok, validation_errors empty), but the accepted explanation contains unsupported interpretations: zeros imply inactivity, demo input implies a synthetic wallet, and a separate risky_tx_count rule implies a model-score contribution. This is not a semantically validated result.
- The saved client sidecar identifies docs/demo_features_valid.json and explicitly limits its synthetic/demo note to an input example, not real wallet evidence. The saved result does not contain an exact generation prompt; context construction was inspected in the current code, not reconstructed as a historical prompt capture.
- Existing context separated feature values, SHAP, rules and provenance into fields, but did not state the evidentiary boundaries strongly enough: not missing does not establish activity meaning; not SHAP causes does not clearly exclude score attribution; demo example provenance does not establish wallet origin. Raw structured risk also includes both top_features and rule_triggers, so their co-location must not imply common attribution.
- Added a shared internal evidence contract in src/explanation_contract.py, passed to the LLM: SHAP supports local partial model attribution; rules support input predicates only; provenance describes the supplied example only; numeric zero alone does not establish inactivity. Clarified those boundaries in the prompt and the input-provenance note. No new wording requirements or content-rejection rules; existing fact checks and free-form explanation remain.
- Added context-construction regression cases with different feature names, zero/nonzero values, positive/negative SHAP impacts, unknown rule definitions, a rule sharing a SHAP feature, and declared/unknown input origins. Rule evidence does not gain SHAP attribution even when names overlap.
- Context clarification does not guarantee truthful generated prose. Existing bounded fact checks do not validate all semantic inferences; acceptance does not prove correctness. No new live result or measured improvement is claimed.
- Tests, Ruff, mypy, Docker and Ollama: not_run (user instruction). No model, graph, data or API-schema changes; no commit/push.

## Stage Handoff And Evidence Inventory On 2026-10-15

This section supersedes earlier current-status summaries. Earlier validation entries are historical observations, not certification of the current working tree. This handoff updated documentation only and performed no new checks or service requests.

Available local-check evidence:

- The last local validation block recorded before the 2026-10-15 changes (Readable Fallback Copy Polish On 2026-10-13) reports targeted pytest 11 passed / 922 warnings, full pytest 107 passed / 929 warnings, Ruff All checks passed, and mypy no issues in 52 source files with the unchecked-untyped-test note. An earlier block reports 116 passed; neither count is evidence for the current files. The recorded counts are retained without inventing an execution order beyond the report sections.
- The subsequently user-reported mypy error at src/explanation_contract.py:42 was addressed with an explicit value is not None guard before float(value). Existing None/zero parameter coverage was inspected; a successful mypy rerun after that fix is not evidenced here.
- Current-stage targeted/full pytest, Ruff and mypy: not_run by this agent; no saved passing report tied to the current working tree was found. Do not carry historical green status forward.

Available Docker/Ollama evidence:

- docs/docker_api_review_smoke_2026-10-13.md and results/docker_api_health_ready_final_review_2026-10-13.json document the historical local Docker API readiness result: model, scoring adapter, retrieval index and corpus true. The historical SHAP-units explain rerun ended in deterministic fallback; it was not accepted LLM text. These observations do not certify the current container/image or working tree.
- The latest available explain artifact is results/demo_agent_evidence_boundaries_2026-10-15.json, request 1644fe29c63a4ebe92661a45941dd81e. Scoring and SHAP are ok; risk.status=scored, data_status=ok, prediction=1, score=0.35894730197831315 above threshold=0.34438552759270996. Ollama qwen2.5:7b-instruct reports ok, done_reason=stop, error=null; validation_errors is empty and no fallback/replacement markers are present. The LLM explanation was accepted.
- The latest client sidecar explicitly identifies a synthetic/demo input example, not real wallet evidence. The JSON does not identify an image digest/source revision or capture the exact generation prompt, so it does not prove which complete working-tree revision ran, nor independently establish a fresh Docker rebuild or health check.
- This latest explain is technically successful but still has interpretation and language defects: unsupported natural-language feature definitions (including first deposit rendered as first record), repeated score/class summary and limitations, awkward wording and an unexplained shortened SHAP-number presentation. These are not evidence of production-quality explanations. The prior inactivity/synthetic-wallet/rule-attribution assertions are not repeated explicitly in this sample; one sample cannot establish their general resolution.
- Quality on real data: not confirmed. Acceptance by bounded factual checks is not proof of semantic correctness, readability or calibrated fraud probability.
- Docker build/health checks and Ollama requests during this handoff: not_run. No new runtime or local-check success is claimed.

Operational status:

- Remote deployment: not_run.
- Rollback verification: not_run.
- Commit/push: not_run.

Exact intended commit allowlist, relative to ml-project (nothing staged or committed):

```text
app/api.py
docs/progress.md
docs/user_output_examples.md
scripts/demo_agent.py
src/agent.py
src/explanation_contract.py
src/inference.py
tests/test_agent.py
tests/test_demo_agent.py
tests/test_explanation_contract.py
tests/test_scored_explanations.py
tests/test_scoring_contract.py
```

Exclude results/ and client_context sidecars, opt-in rejected-text diagnostics, logs, secrets/.env/credentials, local environments, model/data/index/MLflow artifacts (including mlflow_verification/), and unrelated untracked files. Also exclude docs/scored_explanation_changes_2026-10-15.patch: it is an intermediate historical diff, not the current patch. The allowlist includes only the stage implementation, regression tests and maintained documentation; use explicit paths rather than git add . or git add -A.
