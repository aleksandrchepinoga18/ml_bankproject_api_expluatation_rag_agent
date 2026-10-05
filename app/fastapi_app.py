import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from monitoring.observability import collect_metrics, log_agent_response, log_api_request
from src.agent import (
    AGENT_VERSION,
    PROMPT_VERSION,
    AgentRequest,
    AgentResponse,
    DisabledLLMProvider,
    OllamaProvider,
    create_default_agent,
)
from src.retrieval import DEFAULT_INDEX_PATH, build_local_index, load_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    wallet_address: str | None = None
    features: dict[str, Any] | list[dict[str, Any]] | None = None
    row_ids: list[str] | None = None
    conversation_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=10)


class ChatResponse(BaseModel):
    conversation_id: str | None = None
    answer: str
    analysis: AgentResponse


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def ensure_retrieval_index():
    if not DEFAULT_INDEX_PATH.exists():
        build_local_index(output_path=DEFAULT_INDEX_PATH)


def llm_provider_from_env():
    provider = os.getenv("AGENT_LLM_PROVIDER", "ollama").strip().lower()
    if provider in {"disabled", "none", "off"}:
        return DisabledLLMProvider(model="disabled")
    return OllamaProvider(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        model=os.getenv("OLLAMA_MODEL"),
        timeout_seconds=_float_env("AGENT_LLM_TIMEOUT_SECONDS", 120.0),
        num_predict=_int_env("AGENT_LLM_NUM_PREDICT", 512),
    )


def version_payload() -> dict[str, Any]:
    errors = {}
    model_version = None
    schema_version = None
    corpus_version = None
    index_version = None

    try:
        from app import api as flask_api

        model_version = flask_api.MODEL_VERSION
        schema_version = flask_api.SCHEMA_VERSION
    except Exception as exc:
        errors["model"] = str(exc)

    try:
        manifest = load_manifest()
        corpus_version = manifest.get("corpus_version")
    except Exception as exc:
        errors["corpus"] = str(exc)

    try:
        if DEFAULT_INDEX_PATH.exists():
            import json

            with DEFAULT_INDEX_PATH.open("r", encoding="utf-8") as f:
                index_version = json.load(f).get("index_version")
    except Exception as exc:
        errors["index"] = str(exc)

    response = {
        "status": "degraded" if errors else "ok",
        "agent_version": AGENT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model_version": model_version,
        "schema_version": schema_version,
        "corpus_version": corpus_version,
        "index_version": index_version,
        "llm_provider": os.getenv("AGENT_LLM_PROVIDER", "ollama"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.1"),
    }
    if errors:
        response["errors"] = errors
    return response


@lru_cache(maxsize=1)
def get_agent():
    ensure_retrieval_index()
    return create_default_agent(
        llm_provider=llm_provider_from_env(),
        max_steps=_int_env("AGENT_MAX_STEPS", 8),
        tool_timeout_seconds=_float_env("AGENT_TOOL_TIMEOUT_SECONDS", 3.0),
    )


def create_app() -> FastAPI:
    api = FastAPI(
        title="Wallet Risk Scoring RAG Agent",
        version=AGENT_VERSION,
        description="FastAPI facade for the wallet risk scoring agent and legacy scoring adapter.",
    )

    @api.middleware("http")
    async def structured_request_logging(request: Request, call_next):
        started = time.perf_counter()
        request_id = request.headers.get("x-request-id") or request.headers.get("x-correlation-id") or "-"
        status_code = 500
        error = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            error = exc.__class__.__name__
            raise
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            log_api_request(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                latency_ms=latency_ms,
                headers=dict(request.headers),
                error=error,
            )

    @api.get("/health/live")
    def live():
        return {"status": "live"}

    @api.get("/health/ready")
    def ready():
        checks = {
            "model": False,
            "scoring_adapter": False,
            "retrieval_index": DEFAULT_INDEX_PATH.exists(),
            "corpus": False,
        }
        try:
            from app import api as flask_api

            checks["model"] = flask_api.model is not None
            checks["scoring_adapter"] = flask_api.scoring_adapter is not None
        except Exception as exc:
            return {"status": "not_ready", "checks": checks, "error": str(exc)}

        try:
            manifest = load_manifest()
            checks["corpus"] = manifest.get("chunk_count", 0) > 0
        except Exception:
            checks["corpus"] = False

        status = "ready" if all(checks.values()) else "not_ready"
        return {"status": status, "checks": checks}

    @api.get("/version")
    def version():
        return version_payload()

    @api.get("/metrics")
    def metrics():
        return collect_metrics(versions=version_payload())

    @api.post("/analyze", response_model=AgentResponse)
    def analyze(request: AgentRequest):
        started = time.perf_counter()
        analysis = get_agent().run_langgraph(request)
        log_agent_response(analysis, route="/analyze", total_latency_ms=(time.perf_counter() - started) * 1000)
        return analysis

    @api.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest):
        started = time.perf_counter()
        analysis = get_agent().run_langgraph(
            AgentRequest(
                question=request.message,
                wallet_address=request.wallet_address,
                features=request.features,
                row_ids=request.row_ids,
                request_id=request.conversation_id,
                top_k=request.top_k,
            )
        )
        log_agent_response(analysis, route="/chat", total_latency_ms=(time.perf_counter() - started) * 1000)
        return ChatResponse(
            conversation_id=request.conversation_id,
            answer=analysis.answer,
            analysis=analysis,
        )

    return api


app = create_app()
