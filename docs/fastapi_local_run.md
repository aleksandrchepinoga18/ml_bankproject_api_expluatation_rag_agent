# FastAPI local run

Command 9 result for FastAPI and local startup.

## Dependency Check

Checked in the active Python:

```text
python 3.11
fastapi 0.111.0
uvicorn 0.30.0
pydantic 2.7.4
pytest 9.1.1
flask 3.0.0
```

No packages, programs, models, or Docker images were installed or updated.

## Added API

- `GET /health/live`
- `GET /health/ready`
- `GET /version`
- `POST /analyze`
- `POST /chat`

Legacy Flask endpoints are preserved:

- `POST /predict`
- `POST /explain`

## Local HTTP Smoke

Server command used:

```bash
AGENT_LLM_PROVIDER=disabled python -m uvicorn app.fastapi_app:app --host 127.0.0.1 --port 8000
```

Results:

| Check | HTTP | Result |
| --- | ---: | --- |
| `/health/live` | 200 | `live` |
| `/health/ready` | 200 | `ready`, model/scoring_adapter/retrieval_index/corpus all true |
| `/version` | 200 | model `lightgbm_4fc20542d0f4c2cb`, schema `scoring_explanation_v1`, corpus `demo_corpus_v1`, index `idx_422d98707f81` |
| `/analyze` document request | 200 | `intent=documents`, `data_status=ok`, sources `3` |
| `/chat` no-document request | 200 | `intent=unsupported`, no tool calls required |
| `/analyze` address without features | 200 | `data_status=insufficient_data`, `intent=score` |

## Git Bash Commands

Check versions:

```bash
python -c "import importlib.metadata as m; print('fastapi', m.version('fastapi')); print('uvicorn', m.version('uvicorn')); print('pydantic', m.version('pydantic')); print('flask', m.version('flask'))"
```

If these packages are missing in a clean environment, install the project-pinned versions:

```bash
python -m pip install -r requirements.txt
```

Run deterministic local smoke:

```bash
AGENT_LLM_PROVIDER=disabled python scripts/smoke_fastapi.py
```

Run FastAPI locally:

```bash
AGENT_LLM_PROVIDER=disabled python -m uvicorn app.fastapi_app:app --host 127.0.0.1 --port 8000
```

Run with Ollama after confirming the model exists locally:

```bash
AGENT_LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:7b-instruct python -m uvicorn app.fastapi_app:app --host 127.0.0.1 --port 8000
```

Docker Compose files are prepared, but image build/run was not executed in this step:

```bash
cp .env.example .env
docker compose up --build
```

## Docker Compose Verification

Status: `verified on 2026-09-30`.

Docker image was rebuilt and the container was started by the operator after adding `libgomp1` to the `python:3.11-slim` Dockerfile.

Observed HTTP results:

| Check | HTTP | Result |
| --- | ---: | --- |
| `/health/ready` | 200 | `ready`; model, scoring adapter, retrieval index, and corpus checks are all true |
| `/version` | 200 | `status=ok`, `llm_provider=disabled`, model `lightgbm_4fc20542d0f4c2cb`, schema `scoring_explanation_v1`, corpus `demo_corpus_v1`, index `idx_422d98707f81` |

For connecting the existing container to Ollama running on Windows, `.env` should contain:

```text
AGENT_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
AGENT_LLM_TIMEOUT_SECONDS=120
AGENT_LLM_NUM_PREDICT=512
```

`docker-compose.yml` reads `.env` through `env_file` and maps Windows host port `8080` to container port `8000`.

Docker API inspection from this Codex session was blocked by local Docker permissions:

```text
permission denied while trying to connect to the docker API at npipe:////./pipe/docker_engine
```

Apply changed `.env` without rebuilding:

```powershell
docker compose up -d --no-build --force-recreate wallet-risk-api
```

Then verify:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health/ready
Invoke-RestMethod http://127.0.0.1:8080/version
```

Real Docker `/analyze` with Ollama was verified by the operator:

```text
execution_mode=langgraph
tool_selection_source=llm_tool_calling
selected_tools=[retrieval]
llm.status=ok
```

The initial answer was truncated when `AGENT_LLM_NUM_PREDICT=192`. The Agent now records Ollama `done_reason`, retries once with a concise answer prompt, and returns structured fallback output with `validation_errors=["llm_answer_truncated"]` if the retry is still truncated.

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

Inside the Docker build, this line downloads Python dependencies from `requirements.txt`:

```dockerfile
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
```
