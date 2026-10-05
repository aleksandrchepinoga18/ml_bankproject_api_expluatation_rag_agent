# CI/CD and Test Deployment

Command 10 result.

## CI Workflow

Workflow: `.github/workflows/ci.yml`.

Triggers:

- `pull_request`
- `push` to `main` or `master`

Checks:

- Install pinned project dependencies from `requirements.txt`.
- Install CI-only development tools from `requirements-dev.txt`.
- Compile Python modules in `app`, `src`, `scripts`, `monitoring`, and `tests`. This is a syntax check, not a type check and not a dependency import check.
- Run Ruff lint checks with `python -m ruff check app src scripts monitoring tests`.
- Run mypy checks with `python -m mypy app src scripts monitoring tests`.
- Run `scripts/ci_runtime_artifacts_check.py` to fail fast when model/runtime artifacts are missing.
- Run `scripts/ci_corpus_check.py` to validate the checked-in corpus and local retrieval index.
- Run `python -m pytest -q` with `AGENT_LLM_PROVIDER=disabled`.
- Build the Docker image without pushing it, tagged by `${{ github.sha }}`.

The Docker build job is a dry run. It verifies the Dockerfile and dependency installation path in CI, but it does not publish or deploy an image.

Full tests require these runtime artifacts to be restored before CI runs:

```text
models/lightgbm_model.pkl
models/lightgbm_best_threshold.pkl
models/lightgbm_feature_names.pkl
models/lightgbm_preprocessing.pkl
data/corpus/manifest.json
data/corpus/chunks.jsonl
data/vector_index/local_dense_index.json
```

## Image Versioning

Every CI/test-deployment image is identified by the Git commit SHA:

```text
wallet-risk-api:<commit-sha>
ghcr.io/<owner>/wallet-risk-api:<commit-sha>
```

The manual test-deployment workflow also uses a moving test tag:

```text
ghcr.io/<owner>/wallet-risk-api:test-latest
```

Use the SHA tag for reproducible rollback. Treat `test-latest` only as a convenience pointer.

## Protected Manual Test Deployment

Workflow: `.github/workflows/test-deploy.yml`.

Trigger: `workflow_dispatch`.

Protection:

- The job targets the GitHub Environment named `test`.
- Configure required reviewers and deployment branch restrictions in GitHub repository settings for that environment.
- `dry_run` defaults to `true`.
- Publishing a test image requires both `dry_run=false` and `confirm_deploy=true`.

This repository does not include credentials or commands for a remote host, Kubernetes cluster, or cloud runtime. The manual workflow therefore prepares or publishes a test image only. It does not deploy to a server and does not run a server-side smoke test. Server deployment is not prepared until a real test target, access method, secrets, and smoke URL are provided.

## GitHub PR Runbook

Use these steps to open a PR and trigger CI. Do not run remote deployment or rollback from this checklist.

### 1. Verify The Existing Remote

```powershell
git remote -v
git branch --show-current
```

Expected remote observed locally on 2026-10-05:

```text
origin  https://github.com/aleksandrchepinoga18/ml_bankproject_api_expluatation.git (fetch)
origin  https://github.com/aleksandrchepinoga18/ml_bankproject_api_expluatation.git (push)
```

### 2. Check Files Before Commit

Review the changed and untracked files:

```powershell
git status --short
git diff -- .gitignore src/agent.py tests/test_agent.py docs/progress.md docs/command11_acceptance_report.md docs/cicd_test_deployment.md docs/local_monitoring_dashboard.html results/monitoring_snapshot.json results/command11_acceptance_report.json
```

Verify local secrets and unnecessary runtime data are excluded:

```powershell
git check-ignore -v .env monitoring/logs/api_requests_2026-10-05.jsonl monitoring/alerts/test_alert_2026-10-05.json mlruns/1 .matplotlib/fontlist-v330.json project_review.zip mlflow/mlflow.db
git ls-files .env monitoring/logs mlruns mlflow/mlflow.db project_review.zip project_review data/dataset.parquet monitoring/reference/reference_features.parquet monitoring/reference/reference_scores.parquet
```

The second command should print nothing for those sensitive/local paths. `.env.example` is safe to commit; `.env` is not.

### 3. Create A Working Branch

```powershell
git switch -c command11-docker-observability-report
```

If the branch already exists:

```powershell
git switch command11-docker-observability-report
```

### 4. Stage Only Intended Files

Adjust the list if `git status --short` shows additional intentional project files from earlier commands.

```powershell
git add .gitignore src/agent.py tests/test_agent.py docs/progress.md docs/command11_acceptance_report.md docs/cicd_test_deployment.md docs/local_monitoring_dashboard.html results/monitoring_snapshot.json results/command11_acceptance_report.json
```

Before committing, re-check the staged set:

```powershell
git diff --cached --name-status
git diff --cached --check
```

### 5. Commit

```powershell
git commit -m "Verify Docker observability and update command 11 report"
```

### 6. Push The Branch

This starts the GitHub-side path toward CI, but does not run deployment or rollback.

```powershell
git push -u origin command11-docker-observability-report
```

### 7. Open A Pull Request

With GitHub CLI:

```powershell
gh pr create --base main --head command11-docker-observability-report --title "Verify Docker observability and update Command 11 report" --body "Separates real Docker/Ollama checks from mock/TestClient checks, updates monitoring dashboard snapshot, and protects local runtime artifacts from accidental commits."
```

Or open GitHub in a browser after the push and create the PR from the suggested branch banner.

### 8. Inspect CI

On the PR, verify:

- CI workflow `.github/workflows/ci.yml` starts automatically.
- Ruff, mypy, runtime artifact check, corpus/index check, pytest, and Docker build dry run are green.
- No protected `test-deploy` workflow is manually dispatched.
- No remote deployment or rollback is started.

## Commands Shown, Not Executed Here

Docker build dry run:

```bash
docker build -t wallet-risk-api:<commit-sha> .
```

GHCR image publication through the protected workflow:

```text
workflow_dispatch:
  dry_run: false
  confirm_deploy: true
  image_name: wallet-risk-api
```

These commands were documented for operators. They were not executed during Command 10 completion in Codex.

## Test Host Commands

After a protected workflow publishes a SHA-tagged image, a test host can run:

```bash
export IMAGE=ghcr.io/<owner>/wallet-risk-api:<commit-sha>
export API_PORT=8000
export AGENT_LLM_PROVIDER=ollama
export OLLAMA_BASE_URL=http://host.docker.internal:11434
export OLLAMA_MODEL=qwen2.5:7b-instruct
export AGENT_LLM_TIMEOUT_SECONDS=120
export AGENT_LLM_NUM_PREDICT=512

docker pull "$IMAGE"
docker run --rm -d --name wallet-risk-api-test \
  -p 8080:8000 \
  -e API_PORT="$API_PORT" \
  -e AGENT_LLM_PROVIDER="$AGENT_LLM_PROVIDER" \
  -e OLLAMA_BASE_URL="$OLLAMA_BASE_URL" \
  -e OLLAMA_MODEL="$OLLAMA_MODEL" \
  -e AGENT_LLM_TIMEOUT_SECONDS="$AGENT_LLM_TIMEOUT_SECONDS" \
  -e AGENT_LLM_NUM_PREDICT="$AGENT_LLM_NUM_PREDICT" \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/data/corpus:/app/data/corpus:ro" \
  -v "$PWD/data/vector_index:/app/data/vector_index:ro" \
  -v "$PWD/mlruns:/app/mlruns:ro" \
  "$IMAGE"
```

Smoke checks from Windows:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health/ready
Invoke-RestMethod http://127.0.0.1:8080/version
```

## Rollback Runbook

Status: prepared, not executed, and not verified against a real test host.

Rollback must restore one compatible release set. Do not mix a previous image with current model, corpus, index, or manifests.

### 1. Preflight Before Stopping Anything

Verify that the previous image and the full matching artifact set are available:

```bash
export PREVIOUS_SHA=<previous-good-sha>
export PREVIOUS_IMAGE=ghcr.io/<owner>/wallet-risk-api:$PREVIOUS_SHA
export ARTIFACT_SET=artifacts/$PREVIOUS_SHA

docker manifest inspect "$PREVIOUS_IMAGE" >/dev/null

test -f "$ARTIFACT_SET/models/lightgbm_model.pkl"
test -f "$ARTIFACT_SET/models/lightgbm_best_threshold.pkl"
test -f "$ARTIFACT_SET/models/lightgbm_feature_names.pkl"
test -f "$ARTIFACT_SET/models/lightgbm_preprocessing.pkl"

test -f "$ARTIFACT_SET/data/corpus/manifest.json"
test -f "$ARTIFACT_SET/data/corpus/chunks.jsonl"
test -d "$ARTIFACT_SET/data/corpus/documents"

test -f "$ARTIFACT_SET/data/vector_index/local_dense_index.json"
test -f "$ARTIFACT_SET/data/vector_index/qdrant_points.jsonl"
test -f "$ARTIFACT_SET/results/qdrant_index_status.json"
```

Pull the image while the current service is still running:

```bash
docker pull "$PREVIOUS_IMAGE"
```

### 2. Stop Only For The Real Rollback Window

Stop the service only after preflight succeeds:

```bash
docker compose stop wallet-risk-api
```

### 3. Restore The Complete Artifact Set Atomically

Move the current runtime artifacts aside and copy the previous compatible set as a whole:

```bash
export ROLLBACK_BACKUP_DIR=rollback-backups/$(date +%Y%m%d%H%M%S)
mkdir -p "$ROLLBACK_BACKUP_DIR"

mv models "$ROLLBACK_BACKUP_DIR/models"
mv data/corpus "$ROLLBACK_BACKUP_DIR/corpus"
mv data/vector_index "$ROLLBACK_BACKUP_DIR/vector_index"
mv results/qdrant_index_status.json "$ROLLBACK_BACKUP_DIR/qdrant_index_status.json"

mkdir -p models data/corpus data/vector_index results
cp -a "$ARTIFACT_SET/models/." models/
cp -a "$ARTIFACT_SET/data/corpus/." data/corpus/
cp -a "$ARTIFACT_SET/data/vector_index/." data/vector_index/
cp -a "$ARTIFACT_SET/results/qdrant_index_status.json" results/qdrant_index_status.json
```

If an MLflow registry alias is used for runtime model selection, roll it back to the matching version for the same artifact set:

```bash
python scripts/mlflow_registry_rehearsal.py
```

### 4. Restore Qdrant Separately If It Is Used

Copying `qdrant_points.jsonl` only restores the export file. It does not restore or recreate a Qdrant collection.

For local file-backed Qdrant, recreate the collection from the restored local index/export:

```bash
python scripts/index_corpus_qdrant.py --recreate
```

For a remote Qdrant service, use the service's snapshot restore procedure or re-index points from the restored `qdrant_points.jsonl` with the expected collection name and vector size. Confirm the collection status before starting traffic.

### 5. Start The Pinned Previous Image

```bash
export WALLET_RISK_IMAGE="$PREVIOUS_IMAGE"
docker compose up -d --no-build --force-recreate wallet-risk-api
```

### 6. Verify Health, Version, And Smoke

Run all checks before declaring rollback complete:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health/ready
Invoke-RestMethod http://127.0.0.1:8080/version

Invoke-RestMethod http://127.0.0.1:8080/chat `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"message":"What can this assistant do?"}'

Invoke-RestMethod http://127.0.0.1:8080/analyze `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"Show document sources for wallet 0x01da6f3b20d0540f24d390e28195ad7311516739","wallet_address":"0x01da6f3b20d0540f24d390e28195ad7311516739"}'
```
