from __future__ import annotations

import os
import re
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any, Callable, Literal, Protocol, TypedDict, cast

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.inference import build_model_input, get_rule_triggers, get_top_risk_factors
from src.explanation_contract import EXPLANATION_EVIDENCE_CONTRACT, RULE_DEFINITIONS, feature_evidence
from src.retrieval import DEFAULT_INDEX_PATH, LocalDenseRetriever, RetrievalHit


PROMPT_VERSION = "wallet_risk_agent_prompt_v1"
AGENT_VERSION = "wallet_risk_agent_v1"
ADDRESS_RE = re.compile(r"\b0x[a-fA-F0-9]{40}\b")
INTENTS = {"score", "explain", "documents", "combined", "general", "unsupported"}
ALLOWED_TOOL_NAMES = ("scoring", "shap", "retrieval")
TOOL_EXECUTION_ORDER = {"scoring": 0, "shap": 1, "retrieval": 2}
FORBIDDEN_ANSWER_PHRASES = ["proved fraud", "proven fraud", "доказано мошенничество", "credit approved", "credit declined"]


class AgentState(TypedDict, total=False):
    request: AgentRequest
    request_id: str
    intent: Literal["score", "explain", "documents", "combined", "general", "unsupported"]
    wallet_address: str | None
    data_status: Literal["ok", "insufficient_data", "invalid_address", "unsupported", "partial"]
    risk: RiskBlock
    sources: list[SourceRef]
    document_findings: list[DocumentFinding]
    document_hits: list[RetrievalHit]
    selected_tool_calls: list[ToolCallRequest]
    tool_queue: list[ToolCallRequest]
    tool_index: int
    next_route: Literal["scoring", "shap", "retrieval", "build_answer"]
    tool_decision: ToolDecision
    tool_selection_llm: LLMStatus
    tool_selection_source: Literal["llm_tool_calling", "deterministic_fallback", "none"]
    tool_statuses: list[ToolStatus]
    called_tools: set[str]
    trace: list[TraceStep]
    limitations: list[str]
    validation_errors: list[str]
    step_count: int
    llm: LLMStatus
    answer: str
    response: dict[str, Any]
    execution_mode: Literal["builtin", "langgraph"]
    no_tool_answer: str
    scoring_result: Any
    llm_answer_was_truncated: bool
    llm_fallback_reason: str
    _trace_detail: str
    _trace_action: str


class RetrieverBackend(Protocol):
    def search_bm25(
        self,
        query: str,
        wallet_address: str | None = None,
        include_general: bool = False,
        top_k: int = 3,
    ) -> list[RetrievalHit]:
        ...


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    wallet_address: str | None = None
    features: dict[str, Any] | list[dict[str, Any]] | None = None
    row_ids: list[str] | None = None
    request_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=10)

    @field_validator("wallet_address")
    @classmethod
    def normalize_wallet(cls, value):
        return value.lower() if isinstance(value, str) else value


class ToolStatus(BaseModel):
    name: str
    status: Literal["ok", "skipped", "error", "timeout", "duplicate_blocked"]
    elapsed_ms: float = 0.0
    error: str | None = None


class TraceStep(BaseModel):
    step: int
    node: str
    action: str
    status: str
    detail: str | None = None


class ToolCallRequest(BaseModel):
    name: Literal["scoring", "shap", "retrieval"]
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolDecision(BaseModel):
    status: Literal["ok", "fallback", "unavailable", "invalid"]
    source: Literal["llm_tool_calling", "deterministic_fallback"]
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    raw: str | None = None
    error: str | None = None


class LLMStatus(BaseModel):
    provider: str
    model: str
    status: Literal["ok", "unavailable", "skipped", "error"]
    error: str | None = None
    done_reason: str | None = None
    eval_count: int | None = None
    prompt_eval_count: int | None = None


class RiskBlock(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["not_requested", "scored", "insufficient_data", "stale", "error"]
    nature: str = "model_score_not_blockchain_fact"
    model_version: str | None = None
    schema_version: str | None = None
    score: float | None = None
    threshold: float | None = None
    prediction: int | None = None
    aggregation: dict[str, Any] | None = None
    top_features: list[dict[str, Any]] = Field(default_factory=list)
    rule_triggers: list[dict[str, Any]] = Field(default_factory=list)
    reason: str | None = None


class SourceRef(BaseModel):
    document_id: str
    chunk_id: str
    source_kind: str
    wallet_address: str | None
    rank: int
    score: float


class DocumentFinding(BaseModel):
    claim: str
    document_id: str
    chunk_id: str
    source_kind: str
    supported: bool
    synthetic_demo: bool


class AgentResponse(BaseModel):
    request_id: str
    agent_version: str = AGENT_VERSION
    prompt_version: str = PROMPT_VERSION
    intent: Literal["score", "explain", "documents", "combined", "general", "unsupported"]
    wallet_address: str | None = None
    data_status: Literal["ok", "insufficient_data", "invalid_address", "unsupported", "partial"]
    risk: RiskBlock
    document_findings: list[DocumentFinding] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    answer: str
    limitations: list[str] = Field(default_factory=list)
    tool_statuses: list[ToolStatus] = Field(default_factory=list)
    llm: LLMStatus
    trace: list[TraceStep] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    execution_mode: Literal["builtin", "langgraph"] = "builtin"
    tool_selection_source: Literal["llm_tool_calling", "deterministic_fallback", "none"] = "none"
    selected_tools: list[str] = Field(default_factory=list)


class ScoringToolInput(BaseModel):
    features: list[dict[str, Any]] | None = None
    wallet_address: str | None = None
    row_ids: list[str] | None = None


class RetrievalToolInput(BaseModel):
    query: str
    wallet_address: str | None = None
    intent: Literal["documents", "combined", "general", "explain"]
    top_k: int = Field(default=3, ge=1, le=10)


class SHAPToolInput(BaseModel):
    features: list[dict[str, Any]] | None = None
    top_k: int = Field(default=3, ge=1, le=10)


@dataclass
class ToolResult:
    value: Any = None
    status: str = "ok"
    error: str | None = None


class OllamaProvider:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 2.0,
        num_predict: int = 512,
    ):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or "llama3.1"
        self.timeout_seconds = timeout_seconds
        self.num_predict = num_predict

    def generate(self, prompt: str, num_predict: int | None = None) -> tuple[LLMStatus, str | None]:
        effective_num_predict = num_predict or self.num_predict
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": effective_num_predict},
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            return (
                LLMStatus(
                    provider="ollama",
                    model=self.model,
                    status="ok",
                    done_reason=payload.get("done_reason"),
                    eval_count=payload.get("eval_count"),
                    prompt_eval_count=payload.get("prompt_eval_count"),
                ),
                payload.get("response"),
            )
        except Exception as exc:
            return (
                LLMStatus(provider="ollama", model=self.model, status="unavailable", error=str(exc)),
                None,
            )

    def decide_tools(
        self,
        *,
        question: str,
        wallet_address: str | None,
        has_features: bool,
        top_k: int,
    ) -> tuple[LLMStatus, ToolDecision]:
        tools = ollama_tool_specs()
        messages = [
            {
                "role": "system",
                "content": (
                    "You choose tools for a wallet-risk LangGraph agent. "
                    "Allowed tools are exactly scoring, shap, retrieval. "
                    "Call scoring for risk score or insufficient-data checks. "
                    "Call shap when the user asks why/explain/factors and input features exist. "
                    "Call retrieval for document/source/methodology questions. "
                    "If the user asks for document sources, chunks, citations, or evidence, retrieval is mandatory, "
                    "including when scoring features are missing. "
                    "For combined risk plus documents, call scoring and retrieval; also call shap only when "
                    "features exist and the user asks for factors or explanations. "
                    "If features are missing and the user asks for factors, scoring is enough to report "
                    "insufficient data for the model explanation. "
                    "Do not answer the user. Select tools only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "wallet_address": wallet_address,
                        "has_features": has_features,
                        "top_k": top_k,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": tools,
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": self.num_predict},
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            message = payload.get("message", {})
            calls = parse_ollama_tool_calls(message)
            raw = json.dumps(message, ensure_ascii=False, sort_keys=True)
            return (
                LLMStatus(
                    provider="ollama",
                    model=self.model,
                    status="ok",
                    done_reason=payload.get("done_reason"),
                    eval_count=payload.get("eval_count"),
                    prompt_eval_count=payload.get("prompt_eval_count"),
                ),
                ToolDecision(status="ok", source="llm_tool_calling", tool_calls=calls, raw=raw),
            )
        except Exception as exc:
            return (
                LLMStatus(provider="ollama", model=self.model, status="unavailable", error=str(exc)),
                ToolDecision(
                    status="unavailable",
                    source="deterministic_fallback",
                    error=str(exc),
                ),
            )


class DisabledLLMProvider:
    def __init__(self, model: str = "disabled"):
        self.model = model

    def generate(self, prompt: str, **_kwargs) -> tuple[LLMStatus, str | None]:
        return LLMStatus(provider="disabled", model=self.model, status="unavailable", error="llm_disabled"), None

    def decide_tools(self, **_kwargs) -> tuple[LLMStatus, ToolDecision]:
        return (
            LLMStatus(provider="disabled", model=self.model, status="unavailable", error="llm_disabled"),
            ToolDecision(status="unavailable", source="deterministic_fallback", error="llm_disabled"),
        )


def extract_wallet_address(text: str) -> str | None:
    match = ADDRESS_RE.search(text or "")
    return match.group(0).lower() if match else None


def is_valid_wallet_address(value: str | None) -> bool:
    return value is None or bool(ADDRESS_RE.fullmatch(value))


def is_capability_question(question: str) -> bool:
    q = question.lower().strip()
    capability_terms = [
        "что ты умеешь",
        "что умеешь",
        "что можешь",
        "какие вопросы",
        "возможности",
        "помочь",
        "what can this assistant do",
        "what can you do",
        "what do you do",
        "capabilities",
        "help me with",
    ]
    return any(term in q for term in capability_terms)


def is_explicit_methodology_question(question: str) -> bool:
    q = question.lower()
    strong_methodology_terms = [
        "methodology",
        "scoring methodology",
        "feature dictionary",
        "retrieval rules",
        "interpretation limits",
        "методология",
        "правила",
    ]
    if any(term in q for term in strong_methodology_terms):
        return True
    methodology_terms = [
        "method",
        "limits",
        "метод",
        "ограничения",
    ]
    return any(term in q for term in methodology_terms) and not asks_specific_wallet_assessment(q)


def asks_specific_wallet_assessment(q: str) -> bool:
    wallet_terms = ["wallet", "кошел"]
    assessment_terms = [
        "risk score",
        "score",
        "risk",
        "model result",
        "model output",
        "threshold",
        "factor",
        "shap",
        "оцен",
        "риск",
        "результат модели",
        "порог",
        "фактор",
    ]
    direct_phrases = [
        "risk score for",
        "score for",
        "score wallet",
        "probability for",
        "оценку риска",
        "оценка риска",
        "результат модели",
        "сравнение с порогом",
        "основные факторы",
    ]
    return any(phrase in q for phrase in direct_phrases) or (
        any(term in q for term in wallet_terms) and any(term in q for term in assessment_terms)
    )


def classify_intent(question: str, wallet_address: str | None) -> str:
    q = question.lower()
    if any(term in q for term in ["execute", "run code", "powershell", "cmd.exe", "drop table"]):
        return "unsupported"

    doc_terms = ["document", "source", "источник", "документ", "фрагмент", "retrieval", "case", "note"]
    score_terms = ["score", "risk", "скор", "риск", "probability", "оцен"]
    explain_terms = ["why", "explain", "factor", "shap", "почему", "объяс", "фактор"]
    general_terms = ["method", "метод", "rule", "правил", "limit", "огранич", "feature dictionary"]

    asks_docs = any(term in q for term in doc_terms)
    asks_score = any(term in q for term in score_terms)
    asks_explain = any(term in q for term in explain_terms)
    asks_general = any(term in q for term in general_terms)
    asks_wallet_assessment = asks_specific_wallet_assessment(q) or any(
        term in q for term in ["calculate", "compute", "скор для", "риск для", "рассч", "посч"]
    )

    if asks_general and not asks_wallet_assessment:
        return "general"
    if is_explicit_methodology_question(question):
        return "general"
    if wallet_address is None and asks_general:
        return "general"
    if asks_docs and (asks_score or asks_explain):
        return "combined"
    if asks_docs:
        return "documents" if wallet_address else "general"
    if asks_explain:
        return "explain"
    if asks_score:
        return "score"
    return "unsupported"


def normalize_feature_rows(features: dict[str, Any] | list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if features is None:
        return None
    if isinstance(features, dict):
        return [features]
    return features


def ollama_tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "scoring",
                "description": "Compute the wallet risk model score or return insufficient_data if features are missing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Why scoring is needed for this request.",
                        }
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "shap",
                "description": "Explain the current model score with SHAP-style risk factors when input features exist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Why model explanation factors are needed.",
                        }
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "retrieval",
                "description": "Retrieve wallet-linked or general methodology document chunks with source IDs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Why document retrieval is needed.",
                        }
                    },
                    "additionalProperties": False,
                },
            },
        },
    ]


def parse_ollama_tool_calls(message: dict[str, Any]) -> list[ToolCallRequest]:
    raw_calls = message.get("tool_calls") or []
    parsed = []
    for call in raw_calls:
        function = call.get("function", {})
        name = function.get("name")
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        if name in ALLOWED_TOOL_NAMES:
            parsed.append(ToolCallRequest(name=name, arguments=dict(arguments)))

    if parsed:
        return dedupe_tool_calls(parsed)

    content = message.get("content") or ""
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return []

    names = payload.get("tools") or payload.get("tool_calls") or []
    if isinstance(names, str):
        names = [names]
    for item in names:
        if isinstance(item, str):
            name, arguments = item, {}
        else:
            name = item.get("name")
            arguments = item.get("arguments") or {}
        if name in ALLOWED_TOOL_NAMES:
            parsed.append(ToolCallRequest(name=name, arguments=dict(arguments)))
    return dedupe_tool_calls(parsed)


def dedupe_tool_calls(calls: list[ToolCallRequest]) -> list[ToolCallRequest]:
    seen = set()
    result = []
    for call in calls:
        if call.name in seen:
            continue
        seen.add(call.name)
        result.append(call)
    return result


def summarize_hit(hit: RetrievalHit, max_chars: int = 220) -> str:
    text = " ".join(hit.text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


class WalletRiskAgent:
    def __init__(
        self,
        *,
        scoring_adapter=None,
        explainer=None,
        feature_names: list[str] | None = None,
        preprocessing: dict[str, Any] | None = None,
        retriever: RetrieverBackend | None = None,
        llm_provider=None,
        max_steps: int = 8,
        tool_timeout_seconds: float = 3.0,
        max_pending_tool_tasks: int = 4,
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if tool_timeout_seconds <= 0:
            raise ValueError("tool_timeout_seconds must be positive")
        if max_pending_tool_tasks < 1:
            raise ValueError("max_pending_tool_tasks must be at least 1")
        self.scoring_adapter = scoring_adapter
        self.explainer = explainer
        self.feature_names = feature_names or []
        self.preprocessing = preprocessing or {}
        self.retriever = retriever or LocalDenseRetriever(DEFAULT_INDEX_PATH)
        self.llm_provider = llm_provider or OllamaProvider()
        self.max_steps = max_steps
        self.tool_timeout_seconds = tool_timeout_seconds
        self.max_pending_tool_tasks = max_pending_tool_tasks
        self._tool_executor = ThreadPoolExecutor(max_workers=max_pending_tool_tasks)
        self._tool_slots = BoundedSemaphore(max_pending_tool_tasks)

    def run(self, request: AgentRequest | dict[str, Any]) -> AgentResponse:
        req = request if isinstance(request, AgentRequest) else AgentRequest(**request)
        try:
            response = self._invoke_graph(req)
            response.execution_mode = "langgraph"
        except RuntimeError as exc:
            if str(exc) != "langgraph_not_installed":
                raise
            response = self._invoke_builtin(req)
            response.execution_mode = "builtin"
        return response

    def run_langgraph(self, request: AgentRequest | dict[str, Any]) -> AgentResponse:
        req = request if isinstance(request, AgentRequest) else AgentRequest(**request)
        response = self._invoke_graph(req)
        response.execution_mode = "langgraph"
        return response

    def _invoke_graph(self, req: AgentRequest) -> AgentResponse:
        graph = build_langgraph_app(self)
        state = graph.invoke(
            {"request": req.model_dump()},
            config={"recursion_limit": self._graph_recursion_limit()},
        )
        response = AgentResponse(**state["response"])
        return response

    def _invoke_builtin(self, req: AgentRequest) -> AgentResponse:
        state: dict[str, Any] = {
            "request": req,
            "request_id": req.request_id or uuid.uuid4().hex,
            "trace": [],
            "step_count": 0,
            "tool_statuses": [],
            "called_tools": set(),
            "risk": RiskBlock(status="not_requested"),
            "document_findings": [],
            "sources": [],
            "document_hits": [],
            "limitations": ["langgraph_unavailable_builtin_fallback"],
            "validation_errors": [],
            "selected_tool_calls": [],
            "tool_queue": [],
            "tool_index": 0,
        }
        self._step(state, "parse_request", self._parse_request)
        if state.get("intent") != "unsupported" and state.get("data_status") != "invalid_address":
            self._step(state, "llm_tool_decision", self._decide_tools)
            for call in self._ordered_tool_calls(state.get("selected_tool_calls", [])):
                self._execute_tool_node(state, call.name)
        if state.get("step_count", 0) >= self.max_steps:
            self._mark_max_steps_reached(state)
            state["llm"] = LLMStatus(provider="not_called", model="none", status="skipped")
            state["answer"] = self._deterministic_answer(state)
            self._append_trace(state, "build_answer", "safe_fallback", "ok", "max_steps_reached")
        else:
            self._step(state, "build_answer", self._build_answer)
        try:
            self._validate_answer(state)
            validate_status = "ok"
            validate_detail = None
        except Exception as exc:
            validate_status = "error"
            validate_detail = str(exc)
            state["limitations"].append("validate_answer_error")
        self._append_trace(state, "validate_answer", "node", validate_status, validate_detail)
        return self._response(state)

    def _graph_recursion_limit(self) -> int:
        # parse + decision + one router pass before each possible tool and before
        # answer + up to 3 tool nodes + answer + validate, with a small margin.
        max_tool_nodes = len(ALLOWED_TOOL_NAMES)
        return max(8, (2 * max_tool_nodes) + 8)

    def _step(self, state: dict[str, Any], node: str, func: Callable[[dict[str, Any]], None]):
        if state.get("step_count", 0) >= self.max_steps:
            state["limitations"].append("max_steps_reached")
            state["trace"].append(
                TraceStep(
                    step=len(state["trace"]) + 1,
                    node=node,
                    action="blocked",
                    status="error",
                    detail="max_steps_reached",
                )
            )
            return
        state["step_count"] = state.get("step_count", 0) + 1
        try:
            state.pop("_trace_detail", None)
            state.pop("_trace_action", None)
            func(state)
            status = "ok"
            detail = state.pop("_trace_detail", None)
            action = state.pop("_trace_action", "node")
        except Exception as exc:
            status = "error"
            detail = str(exc)
            action = "node"
            state["limitations"].append(f"{node}_error")
        state["trace"].append(
            TraceStep(
                step=len(state["trace"]) + 1,
                node=node,
                action=action,
                status=status,
                detail=detail,
            )
        )

    def _append_trace(self, state: dict[str, Any], node: str, action: str, status: str, detail: str | None = None):
        state["trace"].append(
            TraceStep(
                step=len(state["trace"]) + 1,
                node=node,
                action=action,
                status=status,
                detail=detail,
            )
        )

    def _run_tool(self, state: dict[str, Any], name: str, func: Callable[[], Any]) -> ToolResult:
        if name in state["called_tools"]:
            status = ToolStatus(name=name, status="duplicate_blocked", error="repeated_tool_call")
            state["tool_statuses"].append(status)
            return ToolResult(status="duplicate_blocked", error=status.error)
        state["called_tools"].add(name)
        if not self._tool_slots.acquire(blocking=False):
            status = ToolStatus(name=name, status="error", error="tool_queue_saturated")
            state["tool_statuses"].append(status)
            state["limitations"].append(f"{name}_queue_saturated")
            return ToolResult(status="error", error=status.error)

        started = time.perf_counter()
        slot_released = False
        future = self._tool_executor.submit(func)

        def release_slot(_future):
            nonlocal slot_released
            if not slot_released:
                slot_released = True
                self._tool_slots.release()

        future.add_done_callback(release_slot)
        try:
            value = future.result(timeout=self.tool_timeout_seconds)
            elapsed = (time.perf_counter() - started) * 1000
            state["tool_statuses"].append(ToolStatus(name=name, status="ok", elapsed_ms=elapsed))
            return ToolResult(value=value)
        except TimeoutError:
            elapsed = (time.perf_counter() - started) * 1000
            future.cancel()
            state["tool_statuses"].append(
                ToolStatus(name=name, status="timeout", elapsed_ms=elapsed, error="tool_timeout")
            )
            state["limitations"].append(f"{name}_timeout")
            return ToolResult(status="timeout", error="tool_timeout")
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            state["tool_statuses"].append(
                ToolStatus(name=name, status="error", elapsed_ms=elapsed, error=str(exc))
            )
            state["limitations"].append(f"{name}_error")
            return ToolResult(status="error", error=str(exc))
        finally:
            if future.done():
                release_slot(future)

    def _parse_request(self, state: dict[str, Any]):
        req: AgentRequest = state["request"]
        question_address = extract_wallet_address(req.question)
        wallet_address = req.wallet_address or question_address
        if req.wallet_address and question_address and req.wallet_address != question_address:
            state["intent"] = "unsupported"
            state["wallet_address"] = req.wallet_address
            state["data_status"] = "invalid_address"
            state["limitations"].append("wallet_address_mismatch")
            return
        if not is_valid_wallet_address(wallet_address):
            state["intent"] = "unsupported"
            state["wallet_address"] = wallet_address
            state["data_status"] = "invalid_address"
            state["limitations"].append("invalid_wallet_address")
            return

        intent = classify_intent(req.question, wallet_address)
        state["intent"] = intent
        state["wallet_address"] = wallet_address
        if intent == "unsupported" and is_capability_question(req.question):
            state["data_status"] = "ok"
            state["no_tool_answer"] = "capabilities"
        else:
            state["data_status"] = "unsupported" if intent == "unsupported" else "ok"
        if intent == "unsupported" and state.get("no_tool_answer") != "capabilities":
            state["limitations"].append("unsupported_intent")
        no_tool = f"; no_tool_answer={state['no_tool_answer']}" if state.get("no_tool_answer") else ""
        state["_trace_detail"] = (
            f"question={req.question[:120]}; wallet_address={wallet_address}; fallback_intent={intent}{no_tool}"
        )

    def _deterministic_tool_calls(self, state: dict[str, Any]) -> list[ToolCallRequest]:
        intent = state.get("intent")
        req = state.get("request")
        features = normalize_feature_rows(req.features) if isinstance(req, AgentRequest) else None
        calls = []
        if intent in {"score", "explain", "combined"}:
            calls.append(ToolCallRequest(name="scoring"))
        if intent in {"explain", "combined"} and features:
            calls.append(ToolCallRequest(name="shap"))
        if intent in {"documents", "combined", "general"}:
            calls.append(ToolCallRequest(name="retrieval"))
        return calls

    def _decide_tools(self, state: dict[str, Any]):
        req: AgentRequest = state["request"]
        if state.get("intent") == "unsupported" or state.get("data_status") == "invalid_address":
            decision = ToolDecision(status="ok", source="deterministic_fallback", tool_calls=[])
            state["tool_decision"] = decision
            state["selected_tool_calls"] = []
            state["tool_selection_source"] = decision.source
            state["_trace_detail"] = "selected=none; reason=request_not_tool_eligible"
            return

        features = normalize_feature_rows(req.features)
        llm_status, decision = self.llm_provider.decide_tools(
            question=req.question,
            wallet_address=state.get("wallet_address"),
            has_features=bool(features),
            top_k=req.top_k,
        )
        state["tool_selection_llm"] = llm_status

        valid_calls = []
        invalid_names = []
        intent_removed = []
        dependency_removed = []
        for call in decision.tool_calls:
            if call.name not in ALLOWED_TOOL_NAMES:
                invalid_names.append(call.name)
                continue
            if not self._tool_allowed_for_intent(state, call.name):
                intent_removed.append(call.name)
                continue
            ready, _reason = self._tool_ready(state, call.name)
            if not ready:
                dependency_removed.append(call.name)
                continue
            valid_calls.append(call)

        coverage_added = []
        if decision.status == "ok":
            selected_calls = dedupe_tool_calls(valid_calls)
            selected_names = {call.name for call in selected_calls}
            for required_call in self._deterministic_tool_calls(state):
                if required_call.name not in selected_names:
                    selected_calls.append(required_call)
                    selected_names.add(required_call.name)
                    coverage_added.append(required_call.name)
            state["selected_tool_calls"] = self._ordered_tool_calls(selected_calls)
            state["tool_selection_source"] = "llm_tool_calling"
        else:
            state["selected_tool_calls"] = self._ordered_tool_calls(self._deterministic_tool_calls(state))
            state["tool_selection_source"] = "deterministic_fallback"
            state["limitations"].append("tool_selection_fallback")

        if invalid_names:
            state["limitations"].append("invalid_llm_tool_name")
        if intent_removed:
            state["limitations"].append("tool_selection_intent_guard")
        if dependency_removed:
            state["limitations"].append("tool_selection_dependency_guard")
        if coverage_added:
            state["limitations"].append("tool_selection_coverage_guard")

        selected = ",".join(call.name for call in state["selected_tool_calls"]) or "none"
        state["tool_decision"] = ToolDecision(
            status=decision.status if state["tool_selection_source"] == "llm_tool_calling" else "fallback",
            source=state["tool_selection_source"],
            tool_calls=state["selected_tool_calls"],
            raw=decision.raw,
            error=decision.error,
        )
        state["_trace_action"] = "llm_tool_decision"
        state["_trace_detail"] = (
            f"source={state['tool_selection_source']}; selected={selected}; "
            f"llm_status={llm_status.status}; invalid={','.join(invalid_names) or 'none'}; "
            f"intent_removed={','.join(intent_removed) or 'none'}; "
            f"dependency_removed={','.join(dependency_removed) or 'none'}; "
            f"coverage_added={','.join(coverage_added) or 'none'}"
        )

    def _tool_allowed_for_intent(self, state: dict[str, Any], tool_name: str) -> bool:
        intent = state.get("intent")
        if intent == "general":
            return tool_name == "retrieval"
        if intent == "documents":
            return tool_name == "retrieval"
        if intent == "score":
            return tool_name == "scoring"
        if intent == "explain":
            return tool_name in {"scoring", "shap"}
        if intent == "combined":
            return True
        return False

    def _tool_ready(self, state: dict[str, Any], tool_name: str) -> tuple[bool, str | None]:
        req: AgentRequest = state["request"]
        features = normalize_feature_rows(req.features)
        if tool_name == "shap" and not features:
            return False, "features_required_for_shap"
        return True, None

    def _ordered_tool_calls(self, calls: list[ToolCallRequest]) -> list[ToolCallRequest]:
        return sorted(dedupe_tool_calls(calls), key=lambda call: TOOL_EXECUTION_ORDER[call.name])

    def _mark_max_steps_reached(self, state: dict[str, Any]):
        if "max_steps_reached" not in state["limitations"]:
            state["limitations"].append("max_steps_reached")
        if (
            state.get("intent") in {"score", "explain", "combined"}
            and "scoring" not in state.get("called_tools", set())
            and state.get("risk", RiskBlock(status="not_requested")).status == "not_requested"
        ):
            state["risk"] = RiskBlock(status="error", reason="scoring_not_executed_max_steps")
            state["data_status"] = "partial"
            state["limitations"].append("scoring_not_executed_max_steps")

    def _execute_tool_node(self, state: dict[str, Any], tool_name: Literal["scoring", "shap", "retrieval"]):
        if state.get("step_count", 0) >= self.max_steps:
            self._mark_max_steps_reached(state)
            self._append_trace(state, tool_name, "tool_call_blocked", "error", "max_steps_reached")
            state["tool_index"] = state.get("tool_index", 0) + 1
            return

        state["step_count"] = state.get("step_count", 0) + 1
        validation_error = self._validate_selected_tool_arguments(state, tool_name)
        if validation_error:
            state["limitations"].append(f"{tool_name}_argument_error")
            self._append_trace(state, tool_name, "tool_call", "error", validation_error)
            state["tool_index"] = state.get("tool_index", 0) + 1
            return

        if tool_name == "shap" and state.get("risk", RiskBlock(status="not_requested")).status != "scored":
            status = ToolStatus(name="shap", status="skipped", error="requires_scored_risk")
            state["tool_statuses"].append(status)
            state["limitations"].append("shap_skipped_requires_scored_risk")
            self._append_trace(state, "shap", "tool_call", "skipped", status.error)
            state["tool_index"] = state.get("tool_index", 0) + 1
            return

        ready, reason = self._tool_ready(state, tool_name)
        if not ready:
            status = ToolStatus(name=tool_name, status="skipped", error=reason)
            state["tool_statuses"].append(status)
            state["limitations"].append(reason or f"{tool_name}_skipped")
            self._append_trace(state, tool_name, "tool_call", "skipped", reason)
            state["tool_index"] = state.get("tool_index", 0) + 1
            return

        self._append_trace(state, tool_name, "tool_call", "ok", "arguments=validated")
        before_status_count = len(state["tool_statuses"])
        if tool_name == "scoring":
            self._call_scoring(state)
        elif tool_name == "shap":
            self._call_shap(state)
        elif tool_name == "retrieval":
            self._call_retrieval(state)

        status = state["tool_statuses"][-1] if len(state["tool_statuses"]) > before_status_count else None
        if status is not None:
            self._append_trace(
                state,
                tool_name,
                "tool_result",
                status.status,
                status.error or f"elapsed_ms={status.elapsed_ms:.3f}",
            )
        state["tool_index"] = state.get("tool_index", 0) + 1

    def _validate_selected_tool_arguments(self, state: dict[str, Any], tool_name: str) -> str | None:
        req: AgentRequest = state["request"]
        try:
            if tool_name == "scoring":
                ScoringToolInput(
                    features=normalize_feature_rows(req.features),
                    wallet_address=state.get("wallet_address"),
                    row_ids=req.row_ids,
                )
            elif tool_name == "shap":
                SHAPToolInput(features=normalize_feature_rows(req.features), top_k=3)
            elif tool_name == "retrieval":
                RetrievalToolInput(
                    query=req.question,
                    wallet_address=state.get("wallet_address"),
                    intent=(
                        state["intent"]
                        if state["intent"] in {"documents", "combined", "general", "explain"}
                        else "documents"
                    ),
                    top_k=req.top_k,
                )
        except ValidationError as exc:
            return str(exc)
        return None

    def _call_scoring(self, state: dict[str, Any]):
        req: AgentRequest = state["request"]
        tool_input = ScoringToolInput(
            features=normalize_feature_rows(req.features),
            wallet_address=state.get("wallet_address"),
            row_ids=req.row_ids,
        )

        def call():
            if self.scoring_adapter is None:
                raise RuntimeError("scoring_adapter_not_configured")
            return self.scoring_adapter.score_rows(
                tool_input.features,
                wallet_address=tool_input.wallet_address,
                row_ids=tool_input.row_ids,
            )

        result = self._run_tool(state, "scoring", call)
        if result.status != "ok":
            state["risk"] = RiskBlock(status="error", reason=result.error)
            state["data_status"] = "partial"
            return

        try:
            scoring_result = self._validate_scoring_result(result.value)
        except ValueError as exc:
            self._mark_tool_contract_error(state, "scoring", str(exc))
            state["risk"] = RiskBlock(status="error", reason=str(exc))
            state["data_status"] = "partial"
            return
        if scoring_result.status == "insufficient_data":
            state["risk"] = RiskBlock(
                status="insufficient_data",
                model_version=scoring_result.model_version,
                schema_version=scoring_result.schema_version,
                reason=scoring_result.reason,
            )
            state["data_status"] = "insufficient_data"
            state["limitations"].append(scoring_result.reason or "insufficient_data")
            return

        aggregation = scoring_result.aggregation
        state["risk"] = RiskBlock(
            status="scored",
            model_version=scoring_result.model_version,
            schema_version=scoring_result.schema_version,
            score=aggregation.score,
            threshold=scoring_result.rows[0].threshold if scoring_result.rows else None,
            prediction=aggregation.prediction,
            aggregation=None if aggregation is None else aggregation.__dict__,
        )
        state["scoring_result"] = scoring_result

    def _call_shap(self, state: dict[str, Any]):
        req: AgentRequest = state["request"]
        tool_input = SHAPToolInput(features=normalize_feature_rows(req.features), top_k=3)
        if tool_input.features and len(tool_input.features) > 1:
            state["tool_statuses"].append(
                ToolStatus(name="shap", status="skipped", error="aggregated_score_shap_not_supported")
            )
            state["limitations"].append("shap_skipped_aggregated_score_not_supported")
            return

        def call():
            if self.explainer is None or not self.feature_names:
                raise RuntimeError("shap_not_configured")
            if not tool_input.features:
                raise RuntimeError("features_required_for_shap")
            X = build_model_input(tool_input.features, self.feature_names, self.preprocessing)
            shap_values = self.explainer.shap_values(X.iloc[0:1])
            return {
                "top_features": get_top_risk_factors(
                    shap_values,
                    self.feature_names,
                    X.iloc[0],
                    top_k=tool_input.top_k,
                ),
                "rule_triggers": get_rule_triggers(X.iloc[0]),
            }

        result = self._run_tool(state, "shap", call)
        if result.status == "ok":
            try:
                shap_result = self._validate_shap_result(result.value)
            except ValueError as exc:
                self._mark_tool_contract_error(state, "shap", str(exc))
                return
            state["risk"].top_features = shap_result["top_features"]
            state["risk"].rule_triggers = shap_result["rule_triggers"]
        else:
            state["limitations"].append(result.error or "shap_unavailable")

    def _call_retrieval(self, state: dict[str, Any]):
        req: AgentRequest = state["request"]
        tool_input = RetrievalToolInput(
            query=req.question,
            wallet_address=state.get("wallet_address"),
            intent=state["intent"] if state["intent"] in {"documents", "combined", "general", "explain"} else "documents",
            top_k=req.top_k,
        )

        def call():
            include_general = tool_input.intent == "general"
            search_wallet_address = None if include_general else tool_input.wallet_address
            if tool_input.intent != "general" and search_wallet_address is None:
                return []
            hits = self.retriever.search_bm25(
                tool_input.query,
                wallet_address=search_wallet_address,
                include_general=include_general,
                top_k=tool_input.top_k,
            )
            if include_general:
                hits = [hit for hit in hits if hit.document_type == "methodology"]
            return hits

        result = self._run_tool(state, "retrieval", call)
        if result.status != "ok":
            state["data_status"] = "partial"
            return

        try:
            hits = self._validate_retrieval_result(result.value)
        except ValueError as exc:
            self._mark_tool_contract_error(state, "retrieval", str(exc))
            state["data_status"] = "partial"
            return
        state["sources"] = [
            SourceRef(
                document_id=hit.document_id,
                chunk_id=hit.chunk_id,
                source_kind=hit.source_kind,
                wallet_address=hit.wallet_address,
                rank=hit.rank,
                score=hit.score,
            )
            for hit in hits
        ]
        state["document_hits"] = hits
        state["document_findings"] = [
            DocumentFinding(
                claim=summarize_hit(hit),
                document_id=hit.document_id,
                chunk_id=hit.chunk_id,
                source_kind=hit.source_kind,
                supported=True,
                synthetic_demo=hit.source_kind == "synthetic_demo",
            )
            for hit in hits
        ]
        if not hits:
            state["limitations"].append("no_documents_found")
            if tool_input.intent == "general":
                state["limitations"].append("no_methodology_documents_found")
        if any(hit.source_kind == "synthetic_demo" for hit in hits):
            state["limitations"].append("synthetic_demo_documents_are_not_independent_evidence")

    def _mark_tool_contract_error(self, state: dict[str, Any], name: str, error: str):
        if state.get("tool_statuses") and state["tool_statuses"][-1].name == name:
            state["tool_statuses"][-1] = ToolStatus(name=name, status="error", error=error)
        else:
            state["tool_statuses"].append(ToolStatus(name=name, status="error", error=error))
        state["limitations"].append(f"{name}_contract_error")

    def _validate_scoring_result(self, value: Any) -> Any:
        status = getattr(value, "status", None)
        if status not in {"scored", "insufficient_data"}:
            raise ValueError("scoring_contract_invalid_status")
        for field in ("model_version", "schema_version"):
            if not isinstance(getattr(value, field, None), str) or not getattr(value, field):
                raise ValueError(f"scoring_contract_missing_{field}")
        rows = getattr(value, "rows", None)
        if rows is None:
            raise ValueError("scoring_contract_missing_rows")
        if status == "insufficient_data":
            return value
        aggregation = getattr(value, "aggregation", None)
        if aggregation is None:
            raise ValueError("scoring_contract_missing_aggregation")
        if not rows:
            raise ValueError("scoring_contract_missing_scored_rows")
        for field in ("score", "threshold", "prediction"):
            if getattr(rows[0], field, None) is None:
                raise ValueError(f"scoring_contract_missing_row_{field}")
        for field in ("rule", "score", "prediction", "records_count", "representative_row_id"):
            if getattr(aggregation, field, None) is None:
                raise ValueError(f"scoring_contract_missing_aggregation_{field}")
        return value

    def _validate_shap_result(self, value: Any) -> dict[str, list[dict[str, Any]]]:
        if not isinstance(value, dict):
            raise ValueError("shap_contract_not_dict")
        top_features = value.get("top_features")
        rule_triggers = value.get("rule_triggers")
        if not isinstance(top_features, list) or not isinstance(rule_triggers, list):
            raise ValueError("shap_contract_invalid_fields")
        return {"top_features": top_features, "rule_triggers": rule_triggers}

    def _validate_retrieval_result(self, value: Any) -> list[RetrievalHit]:
        if not isinstance(value, list):
            raise ValueError("retrieval_contract_not_list")
        for hit in value:
            for field in ("chunk_id", "document_id", "source_kind", "rank", "score", "text"):
                if getattr(hit, field, None) is None:
                    raise ValueError(f"retrieval_contract_missing_{field}")
        return cast(list[RetrievalHit], value)

    def _build_answer(self, state: dict[str, Any]):
        if state.get("no_tool_answer") == "capabilities":
            state["llm"] = LLMStatus(provider="not_called", model="none", status="skipped")
            state["answer"] = self._deterministic_answer(state)
            return
        req: AgentRequest = state["request"]
        if (
            state.get("intent") == "general"
            and state.get("risk", RiskBlock(status="not_requested")).status == "not_requested"
            and is_explicit_methodology_question(req.question)
        ):
            state["llm"] = LLMStatus(
                provider="not_called",
                model="none",
                status="skipped",
                error="answer_generation_skipped_for_explicit_methodology",
            )
            state["limitations"].append("llm_answer_skipped_for_explicit_methodology")
            state["answer"] = self._deterministic_methodology_answer(state) or self._deterministic_answer(state)
            state["_trace_action"] = "deterministic_methodology_answer"
            state["_trace_detail"] = "llm_generation_skipped_for_explicit_methodology"
            return
        llm_status, llm_text = self._generate_answer(self._prompt(state))
        state["llm"] = llm_status
        if llm_text and llm_status.status == "ok":
            if not self._is_truncated_llm_answer(llm_status, llm_text):
                replacement_reason = self._answer_replacement_reason(state, llm_text)
                if replacement_reason:
                    self._save_rejected_llm_diagnostic(state, replacement_reason, llm_text)
                    self._mark_answer_fallback(state, replacement_reason)
                    state["answer"] = self._deterministic_answer(state)
                    return
                state["answer"] = self._accepted_llm_answer(state, llm_text)
                return

            state["limitations"].append("llm_answer_retry_after_truncation")
            retry_status, retry_text = self._generate_answer(
                self._retry_prompt(state, llm_text),
                num_predict=max(getattr(self.llm_provider, "num_predict", 512), 512),
            )
            state["llm"] = retry_status
            if retry_text and retry_status.status == "ok" and not self._is_truncated_llm_answer(retry_status, retry_text):
                replacement_reason = self._answer_replacement_reason(state, retry_text)
                if replacement_reason:
                    self._save_rejected_llm_diagnostic(state, replacement_reason, retry_text)
                    self._mark_answer_fallback(state, replacement_reason)
                    state["answer"] = self._deterministic_answer(state)
                    return
                state["answer"] = self._accepted_llm_answer(state, retry_text)
                return

            if retry_status.status in {"unavailable", "error"}:
                state["llm_fallback_reason"] = retry_status.status
                self._mark_answer_fallback(state, f"llm_answer_fallback_after_{retry_status.status}")
            else:
                state["llm_answer_was_truncated"] = True
                state["llm_fallback_reason"] = "truncated"
                self._mark_answer_fallback(state, "llm_answer_fallback_after_truncation")
        elif llm_status.status in {"unavailable", "error"}:
            state["llm_fallback_reason"] = llm_status.status
            self._mark_answer_fallback(state, f"llm_answer_fallback_after_{llm_status.status}")
        elif llm_status.status == "ok" and not llm_text:
            state["llm_fallback_reason"] = "empty_response"
            self._mark_answer_fallback(state, "llm_answer_fallback_after_empty_response")
        state["answer"] = self._deterministic_answer(state)
        if state.get("llm_fallback_reason") == "truncated" and state["risk"].status != "scored":
            state["answer"] = (
                f"{state['answer']} LLM answer was truncated by generation limits; "
                "returning structured tool output instead."
            )

    def _mark_answer_fallback(self, state: dict[str, Any], reason: str):
        limitations = state.setdefault("limitations", [])
        if reason not in limitations:
            limitations.append(reason)
        marker = "llm_answer_fallback"
        if marker not in limitations:
            limitations.append(marker)
        state.setdefault("validation_errors", []).append(reason)
        if reason.startswith("llm_answer_replaced_") or reason == "llm_answer_fallback_after_truncation":
            state["data_status"] = "partial"
        state["_trace_detail"] = f"llm_answer_replaced:{reason}"

    def _accepted_llm_answer(self, state: dict[str, Any], text: str) -> str:
        # Preserve original evidence for the existing final validation/fallback path.
        lowered = text.lower()
        if self._source_consistency_error(state, lowered) or self._final_answer_phrase_errors(lowered, FORBIDDEN_ANSWER_PHRASES):
            return text.strip()
        if state.get("risk", RiskBlock(status="not_requested")).status == "scored":
            return self._compose_scored_answer(state, text)
        return text.strip()

    def _save_rejected_llm_diagnostic(self, state: dict[str, Any], reason: str, rejected_text: str | None) -> None:
        diagnostics_dir = os.getenv("AGENT_REJECTED_LLM_DIAGNOSTICS_DIR")
        if not diagnostics_dir or not rejected_text:
            return
        llm = state.get("llm")
        if isinstance(llm, LLMStatus):
            llm_payload: dict[str, Any] = llm.model_dump()
        else:
            llm_payload = {}
        payload = {
            "agent_version": AGENT_VERSION,
            "prompt_version": PROMPT_VERSION,
            "request_id": state.get("request_id"),
            "created_at_unix": time.time(),
            "intent": state.get("intent"),
            "replacement_reason": reason,
            "rejection_detail": self._rejection_detail(state, reason, rejected_text),
            "llm": llm_payload,
            "rejected_text": rejected_text,
        }
        try:
            path = Path(diagnostics_dir)
            path.mkdir(parents=True, exist_ok=True)
            request_id = str(state.get("request_id") or "unknown")
            file_path = path / f"rejected_llm_{request_id}_{uuid.uuid4().hex[:8]}.json"
            file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            logging.getLogger(__name__).warning(
                "Rejected LLM diagnostic write failed: request_id=%s reason=%s error_type=%s",
                state.get("request_id"), reason, type(exc).__name__,
            )

    def _answer_replacement_reason(self, state: dict[str, Any], text: str) -> str | None:
        if self._violates_methodology_answer(state, text):
            return "llm_answer_replaced_for_methodology_answer"
        if self._violates_insufficient_data_boundary(state, text):
            return "llm_answer_replaced_for_insufficient_data"
        if self._violates_not_requested_boundary(state, text):
            return "llm_answer_replaced_for_not_requested_risk"
        if self._violates_scored_boundary(state, text):
            return "llm_answer_replaced_for_scored_risk_contradiction"
        if self._violates_shap_interpretation(state, text):
            return "llm_answer_replaced_for_shap_unit_misinterpretation"
        return None

    def _generate_answer(self, prompt: str, *, num_predict: int | None = None) -> tuple[LLMStatus, str | None]:
        try:
            return self.llm_provider.generate(prompt, num_predict=num_predict)
        except TypeError:
            return self.llm_provider.generate(prompt)

    def _is_truncated_llm_answer(self, status: LLMStatus, text: str) -> bool:
        reason = (status.done_reason or "").lower()
        if reason in {"length", "num_predict"} or "length" in reason or "num_predict" in reason:
            return True
        stripped = text.strip()
        if not stripped:
            return True
        if stripped.count("**") % 2 == 1 or stripped.count("```") % 2 == 1:
            return True
        last_line = stripped.splitlines()[-1].strip()
        if last_line.endswith((",", ":", ";", "and", "or")):
            return True
        return bool(re.fullmatch(r"\d+\.\s+\*\*Document ID:\*\*\s+\S+", last_line))

    def _violates_insufficient_data_boundary(self, state: dict[str, Any], text: str) -> bool:
        risk: RiskBlock = state.get("risk", RiskBlock(status="not_requested"))
        if risk.status != "insufficient_data":
            return False
        lowered = text.lower()
        phrases = [
            "low risk",
            "lower risk",
            "minimal risk",
            "no fraud",
            "not fraud",
            "does not appear fraudulent",
            "absence of fraud",
            "нет мошенничества",
            "низкий риск",
            "риск низкий",
            "мошенничество отсутствует",
            "не является мошенничеством",
        ]
        return any(phrase in lowered for phrase in phrases)

    def _violates_not_requested_boundary(self, state: dict[str, Any], text: str) -> bool:
        risk: RiskBlock = state.get("risk", RiskBlock(status="not_requested"))
        if risk.status != "not_requested":
            return False
        lowered = text.lower()
        phrases = [
            "not enough data",
            "no sufficient data",
            "insufficient data",
            "not enough information",
            "cannot assess risk",
            "can't assess risk",
            "cannot determine risk",
            "can't determine risk",
            "cannot determine the risk",
            "no risk status",
            "risk status cannot be determined",
            "no risk status or findings can be determined",
            "no risk assessment can be made",
            "unable to assess risk",
            "missing data",
            "missing features",
            "данных недостаточно",
            "недостаточно данных",
        ]
        return any(phrase in lowered for phrase in phrases)

    def _violates_methodology_answer(self, state: dict[str, Any], text: str) -> bool:
        if state.get("intent") != "general":
            return False
        method_sources = [
            source for source in state.get("sources", []) if str(source.document_id).startswith("method_")
        ]
        if not method_sources:
            return False
        lowered = text.lower()
        mentions_method_ref = any(
            source.document_id.lower() in lowered or source.chunk_id.lower() in lowered
            for source in method_sources
        )
        methodology_terms = [
            "lightgbm",
            "model score",
            "calibrated",
            "wallet_address",
            "semantic",
            "retrieval",
            "feature_names",
            "методолог",
            "модель",
            "признак",
        ]
        mentions_method_content = any(term in lowered for term in methodology_terms)
        return not (mentions_method_ref and mentions_method_content)

    def _violates_scored_boundary(self, state: dict[str, Any], text: str) -> bool:
        return self._scored_boundary_detail(state, text) is not None

    def _scored_boundary_detail(self, state: dict[str, Any], text: str) -> dict[str, Any] | None:
        risk: RiskBlock = state.get("risk", RiskBlock(status="not_requested"))
        if risk.status != "scored":
            return None
        lowered = re.sub(r"(?<=\d),(?=\d)", ".", text.lower())
        for detail in (self._feature_presence_detail(state, lowered), self._risk_label_detail(risk, lowered)):
            if detail:
                return detail
        # Check explicit claims that scoring did not happen, not missing semantic documentation.
        insufficient_patterns = [
            r"\b(?:insufficient|not\s+enough)\s+(?:data|information)\s+to\s+(?:calculate|compute)\s+(?:the\s+)?(?:risk\s+)?score\b",
            r"\bcannot\s+(assess|determine)\s+(the\s+)?risk\b",
            r"\bcan't\s+(assess|determine)\s+(the\s+)?risk\b",
            r"\bunable\s+to\s+(assess|determine)\s+(the\s+)?risk\b",
            r"\brisk\s+status\s*(is|:)?\s*insufficient\s+data\b",
            r"\bscore\s+is\s+unavailable\b",
        ]
        for pattern in insufficient_patterns:
            match = re.search(pattern, lowered)
            if match:
                return {"rule": "scored_status", "field": "risk.status", "expected": "scored", "detected": match.group(0)}

        relation = self._score_threshold_relation(risk)
        if relation and self._text_contradicts_score_relation(lowered, relation):
            return {"rule": "score_threshold_relation", "field": "risk.score/risk.threshold", "expected": relation, "detected": text}

        for field, actual, labels in (
            ("risk.score", risk.score, ("risk score", "model score", "score", "скор", "оценка модели", "оценка")),
            ("risk.threshold", risk.threshold, ("threshold", "порог", "пороге")),
            ("risk.prediction", risk.prediction, ("prediction", "predicted class", "предсказание", "итоговый класс")),
        ):
            detail = self._reported_number_detail(lowered, field, actual, labels)
            if detail:
                return detail
        for match in re.finditer(r"\b(?:class|класс)\s*[:=]\s*(\d+(?:\.\d+)?)", lowered):
            if risk.prediction is not None and float(match.group(1)) != risk.prediction:
                return {"rule": "reported_class", "field": "risk.prediction", "expected": risk.prediction, "detected": match.group(0)}
        for segment in re.split(r"(?<=[.!?])\s+|[\n;]+", lowered):
            match = re.match(
                r"\s*(?:результат классификации\s*[-—:=]?\s*)?класс\s+(повышенного|пониженного)\s+риска\b",
                segment,
            )
            if match and risk.prediction in (0, 1):
                detected = 1 if match.group(1) == "повышенного" else 0
                if detected != risk.prediction:
                    return {"rule": "reported_class_label", "field": "risk.prediction", "expected": risk.prediction, "detected": match.group(0)}
            english = re.search(
                r"\b(?:predicted class|prediction|model predicts)\s*(?:is\s+|[:=]\s*)?"
                r"(?:the\s+)?(?:class\s+)?(high(?:er)?|low(?:er)?)\s+risk\b", segment,
            )
            russian = re.search(
                r"\b(?:предсказанный класс\s*(?:[-—:=]\s*)?|"
                r"модель отнесла.{0,30}к классу\s+)(повышенного|пониженного)\s+риска\b", segment,
            )
            reported = english or russian
            if reported and risk.prediction in (0, 1):
                detected = int(reported.group(1) in {"high", "higher", "повышенного"})
                if detected != risk.prediction:
                    return {"rule": "reported_class_label", "field": "risk.prediction", "expected": risk.prediction, "detected": reported.group(0)}

        if self._text_claims_model_score_from_retrieval(lowered):
            return {"rule": "score_provenance", "field": "risk.score", "expected": "model output, not retrieval", "detected": text}

        return self._feature_or_rule_number_detail(state, lowered)

    def _feature_facts(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        request = state.get("request")
        rows = normalize_feature_rows(request.features) if isinstance(request, AgentRequest) else None
        row = rows[0] if rows and len(rows) == 1 else None
        return feature_evidence(state["risk"].top_features, row)

    def _feature_presence_detail(self, state: dict[str, Any], text: str) -> dict[str, Any] | None:
        facts = {item["feature"].lower(): item for item in self._feature_facts(state)}
        if not facts:
            return None
        names = "|".join(re.escape(name) for name in sorted(facts, key=len, reverse=True))
        mentions = list(re.finditer(rf"(?<!\w)(?:{names})(?!\w)", text))
        absence = (
            r"не было\s+(?:никаких\s+)?(?:значений|данных)|"
            r"(?:значение|значения|данные)\s+(?:признака\s+)?(?:отсутств\w*|не\s+(?:передан\w*|задан\w*))|"
            r"данных\s+нет|no\s+(?:value|values|data)\s+(?:was|were|is|are)\s+(?:provided|available)|"
            r"(?:value|data)\s+(?:is|was)\s+missing"
        )
        for index, mention in enumerate(mentions):
            end = mentions[index + 1].start() if index + 1 < len(mentions) else len(text)
            fragment = text[mention.end():min(end, mention.end() + 280)]
            fact = facts[mention.group(0)]
            for match in re.finditer(absence, fragment):
                preceding = text[max(0, mention.start() - 90):mention.end()] + fragment[:match.start()]
                prefix = re.split(r"(?<=[.!?])\s+|;", preceding)[-1][-130:]
                if re.search(r"(?:смысл|определение|интерпретация|meaning|definition)(?:\s+(?:этого|его|the|of))*\s*$", prefix):
                    continue
                if re.search(r"(?:нельзя|не следует)\s+(?:считать|говорить|утверждать)|cannot\s+(?:say|claim)|неизвестно.{0,40}(?:ли|означает)|not known whether", prefix):
                    continue
                raw_claim = bool(re.search(r"\b(?:исходн\w*|входн\w*|original|input)\b", prefix.replace(mention.group(0), "")))
                status = fact["input_value_status"] if raw_claim else fact["model_value_status"]
                if status == "present_numeric":
                    return {
                        "rule": "feature_value_presence", "field": f"risk.top_features[{fact['feature']}].value",
                        "expected": {"status": status, "value": fact["input_value"] if raw_claim else fact["model_value"]},
                        "detected": match.group(0),
                    }
        return None

    def _risk_label_detail(self, risk: RiskBlock, text: str) -> dict[str, Any] | None:
        # Detect explicit category assertions, not the mere occurrence of category words.
        category = r"(?:средн\w*|умеренн\w*|повышенн\w*|пониженн\w*|высок\w*|низк\w*|medium|moderate|intermediate|high|low)"
        patterns = (
            rf"\b(?:уровень\s+риска|риск|risk(?:\s+level)?|predicted\s+class|класс(?:\s+риска)?)\s*(?:(?:is|was|является|считается|оценивается\s+как|находится\s+на)\s*|[-—:=]\s*)?(?P<label>{category})\b",
            rf"\b(?P<label>{category})\s+(?:риск|risk)\b",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                prefix = text[max(0, match.start() - 65):match.start()]
                suffix = text[match.end():match.end() + 55]
                if re.search(r"(?:не|not|не является|нельзя назвать|нельзя считать|не означает|не включает)\s*$", prefix):
                    continue
                if re.search(r"\bне\s+(?:вводит|выделяет|определяет|использует|предусматривает)\s*$|\bdoes not\s+(?:define|use|include)\s*$", prefix):
                    continue
                if re.search(r"(?:неизвестно|не подтверждено).{0,40}\bли\s*$|\b(?:если|if)\s*$", prefix):
                    continue
                if re.search(r"(?:вклад|shap|impact|contribution).{0,45}(?:в сторону|направлен|toward|target)", prefix):
                    continue
                if re.match(r"[\s\"'»”]*(?:как\s+отдельный\s+класс\s+)?(?:здесь\s+)?(?:нет\b|не\s+(?:предусмотрен|задан|определен|подтвержден)|is not defined)", suffix):
                    continue
                label = match.group("label")
                if re.match(r"средн|умеренн|medium|moderate|intermediate", label):
                    return {"rule": "unsupported_risk_scale", "field": "risk.prediction", "expected": {"classes": [0, 1], "prediction": risk.prediction}, "detected": match.group(0)}
                predicted = 1 if re.match(r"повышенн|высок|high", label) else 0
                if risk.prediction in (0, 1) and predicted != risk.prediction:
                    return {"rule": "reported_class_label", "field": "risk.prediction", "expected": risk.prediction, "detected": match.group(0)}
        return None

    def _reported_number_detail(
        self, text: str, field: str, actual: float | int | None, labels: tuple[str, ...],
    ) -> dict[str, Any] | None:
        if actual is None:
            return None
        label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
        negative = rf"\b(?:{label_pattern})\b\s*(?:is\s+|does\s+)?(?:not\s+(?:equal\s+to\s+)?|не\s+(?:равен\s+|равно\s+)?)(\d+(?:\.\d+)?)"
        for match in re.finditer(negative, text):
            value = float(match.group(1))
            same = value == actual if field == "risk.prediction" else self._same_reported_number(value, actual)
            if same:
                return {"rule": "negated_actual_number", "field": field, "expected": actual, "detected": match.group(0)}
        asserted_text = re.sub(negative, "", text)
        for value in self._extract_labeled_numbers(asserted_text, labels):
            same = value == actual if field == "risk.prediction" else self._same_reported_number(value, actual)
            if not same:
                return {"rule": "reported_class" if field == "risk.prediction" else "reported_number", "field": field, "expected": actual, "detected": value}
        return None

    def _score_threshold_relation(self, risk: RiskBlock) -> str | None:
        if risk.score is None or risk.threshold is None:
            return None
        if risk.score > risk.threshold:
            return "above"
        if risk.score < risk.threshold:
            return "below"
        return "equal"

    def _text_contradicts_score_relation(self, lowered: str, relation: str) -> bool:
        # Negated comparisons assert the complement, rather than the quoted relation.
        # This deliberately covers a bounded set of explicit forms, not arbitrary natural language.
        for pattern, excluded in (
            (r"\b(?:not\s+(?:below|under)|не\s+ниже)\s+(?:the\s+)?(?:threshold|порога)\b", "below"),
            (r"\b(?:not\s+(?:above|over)|не\s+выше)\s+(?:the\s+)?(?:threshold|порога)\b", "above"),
            (r"\b(?:not\s+equal\s+to|не\s+равен|не\s+равно)\s+(?:the\s+)?(?:threshold|порогу)\b", "equal"),
        ):
            if re.search(pattern, lowered) and relation == excluded:
                return True
            lowered = re.sub(pattern, "", lowered)
        russian_says_above = "\u0432\u044b\u0448\u0435 \u043f\u043e\u0440\u043e\u0433\u0430" in lowered
        russian_says_below = "\u043d\u0438\u0436\u0435 \u043f\u043e\u0440\u043e\u0433\u0430" in lowered
        russian_says_equal = "\u0440\u0430\u0432\u0435\u043d \u043f\u043e\u0440\u043e\u0433\u0443" in lowered or "\u0440\u0430\u0432\u043d\u043e \u043f\u043e\u0440\u043e\u0433\u0443" in lowered
        says_above = bool(
            russian_says_above
            or
            re.search(r"\b(above|over|exceeds?|greater\s+than|higher\s+than)\s+(the\s+)?threshold\b", lowered)
            or re.search(r"\bthreshold\b.{0,40}\b(below|under|less\s+than|lower\s+than)\s+(the\s+)?score\b", lowered)
            or re.search(r"\bscore\b.{0,40}\b(above|over|exceeds?|greater\s+than|higher\s+than)\b.{0,40}\bthreshold\b", lowered)
            or re.search(r"\b(score|скор)\b.{0,50}\b(выше|больше|превышает)\b.{0,50}\b(порог|threshold)\b", lowered)
            or re.search(r"\b(порог|threshold)\b.{0,50}\b(ниже|меньше)\b.{0,50}\b(score|скор)\b", lowered)
        )
        says_below = bool(
            russian_says_below
            or
            re.search(r"\b(below|under|less\s+than|lower\s+than)\s+(the\s+)?threshold\b", lowered)
            or re.search(r"\bthreshold\b.{0,40}\b(above|over|exceeds?|greater\s+than|higher\s+than)\s+(the\s+)?score\b", lowered)
            or re.search(r"\bscore\b.{0,40}\b(below|under|less\s+than|lower\s+than)\b.{0,40}\bthreshold\b", lowered)
            or re.search(r"\b(score|скор|значение)\b.{0,50}\b(ниже|меньше)\b.{0,50}\b(порог|threshold)\b", lowered)
            or re.search(r"\b(порог|threshold)\b.{0,50}\b(выше|больше|превышает)\b.{0,50}\b(score|скор)\b", lowered)
        )
        says_equal = bool(
            russian_says_equal
            or
            re.search(r"\b(equal\s+to|equals?|at|matches?)\s+(the\s+)?threshold\b", lowered)
            or re.search(r"\bscore\b.{0,40}\b(equal\s+to|equals?|matches?)\b.{0,40}\bthreshold\b", lowered)
            or re.search(r"\b(score|скор|значение)\b.{0,50}\b(равен|равно|совпадает)\b.{0,50}\b(порог|threshold)\b", lowered)
        )
        if relation == "above":
            return says_below or says_equal
        if relation == "below":
            return says_above or says_equal
        return says_above or says_below

    def _extract_labeled_numbers(self, lowered: str, labels: tuple[str, ...]) -> list[float]:
        number_pattern = r"([0-9]+(?:\.[0-9]+)?)"
        values: list[float] = []
        field_labels = (
            "risk score",
            "model score",
            "score",
            "\u0441\u043a\u043e\u0440",
            "\u043e\u0446\u0435\u043d\u043a\u0430",
            "оценка модели",
            "threshold",
            "\u043f\u043e\u0440\u043e\u0433",
            "пороге",
            "prediction",
            "predicted class",
            "class",
            "\u043f\u0440\u0435\u0434\u0441\u043a\u0430\u0437\u0430\u043d\u0438\u0435",
            "\u043a\u043b\u0430\u0441\u0441",
        )
        field_label_pattern = "|".join(re.escape(item) for item in sorted(field_labels, key=len, reverse=True))
        seen_spans: set[tuple[str, int, int]] = set()
        sentence_parts = re.split(r"(?<=[.!?])\s+|[\n\r;]+", lowered)
        for label in sorted(labels, key=len, reverse=True):
            escaped = re.escape(label)
            for sentence in sentence_parts:
                for label_match in re.finditer(rf"\b{escaped}\b", sentence):
                    after = sentence[label_match.end() :]
                    next_label = re.search(rf"\b(?:{field_label_pattern})\b", after)
                    if next_label is not None:
                        after = after[: next_label.start()]
                    next_number = re.search(rf"^[\s:=—–*`]*(?:(?:is|of|equals?|составляет|равен|равно)\s+)?{number_pattern}", after)
                    if next_number is not None:
                        number_span = (
                            sentence,
                            label_match.end() + next_number.start(1),
                            label_match.end() + next_number.end(1),
                        )
                        if number_span not in seen_spans:
                            seen_spans.add(number_span)
                            values.append(float(next_number.group(1)))
                        continue

                    before = sentence[: label_match.start()]
                    previous_labels = list(re.finditer(rf"\b(?:{field_label_pattern})\b", before))
                    if previous_labels:
                        before = before[previous_labels[-1].end() :]
                    previous_number = re.search(rf"{number_pattern}[\s:=—–*`]*$", before)
                    if previous_number is not None:
                        number_span = (sentence, previous_number.start(1), previous_number.end(1))
                        if number_span not in seen_spans:
                            seen_spans.add(number_span)
                            values.append(float(previous_number.group(1)))
        return values

    def _extract_labeled_ints(self, lowered: str, labels: tuple[str, ...]) -> list[int]:
        values = self._extract_labeled_numbers(lowered, labels)
        return [int(value) for value in values if value.is_integer()]

    def _same_reported_number(self, reported: float, actual: float) -> bool:
        return abs(reported - actual) <= 5e-4

    def _text_claims_model_score_from_retrieval(self, lowered: str) -> bool:
        score_terms = r"(model\s+)?(risk\s+)?score|lightgbm\s+score|скор|оценка модели"
        source_terms = r"synthetic(_demo)?|synthetic note|retrieval|document|note|документ|заметк"
        causal_terms = r"(?<!not )(?<!не )(?:based on|derived from|calculated from|computed from|основан|рассчитан|вычислен)"
        return bool(
            re.search(rf"({score_terms}).{{0,80}}({causal_terms}).{{0,80}}({source_terms})", lowered)
            or re.search(rf"({source_terms}).{{0,80}}({causal_terms}).{{0,80}}({score_terms})", lowered)
        )

    def _violates_shap_interpretation(self, state: dict[str, Any], text: str) -> bool:
        risk: RiskBlock = state.get("risk", RiskBlock(status="not_requested"))
        if risk.status != "scored" or not risk.top_features:
            return False
        lowered = text.lower()
        for segment in re.split(r"(?<=[.!?])\s+|[\n;]+", lowered):
            if re.search(
                r"\b(?:shap|impact|contribution|вклад)\b.{0,35}"
                r"\b(?:is|are|это)\s+(?:a\s+)?(?:probabilit\w*|вероятност\w*)", segment,
            ):
                return True
        percentage_terms = (
            r"(?:\d+(?:[.,]\d+)?\s*%|percent(?:age)?\s+points?|percentage\s+points?|"
            r"процентн(?:ых|ые|ыми)?\s+пункт|в\s+процентах|как\s+процент)"
        )
        if not re.search(percentage_terms, lowered):
            return False
        shap_value_terms = (
            r"(shap|impact|contribution|contributes?|influences?|affects?|"
            r"вклад|влия[еюя][тл]?|повлиял|повлияли|характеристик[аи].{0,40}влия)"
        )
        target_terms = r"(score|risk score|model score|risk|model|оценк|риск|модел)"
        segments = re.split(r"(?<=[.!?])\s+|[\n\r;]+", lowered)
        shap_context = False
        for segment in segments:
            if not segment.strip():
                continue
            has_shap_context = bool(re.search(shap_value_terms, segment) and re.search(target_terms, segment))
            if has_shap_context:
                shap_context = True
            elif segment.lstrip().startswith(("-", "*")) or re.match(r"\s*\d+[\.)]", segment):
                has_shap_context = shap_context
            else:
                shap_context = False
            if shap_context:
                # Ignore explicitly denied unit claims; keep checking later clauses independently.
                for clause in re.split(r"[;,]|\bbut\b|\bно\b", segment):
                    if re.search(
                        r"\b(?:not|не)\s+(?:(?:expressed|measured)\s+(?:in|as)\s+|"
                        r"выража[ею]тся\s+в\s+|измеря[ею]тся\s+в\s+|является\s+|в\s+)?"
                        r"(?:\d+(?:[.,]\d+)?\s*%|percent\w*|процент\w*)", clause,
                    ):
                        continue
                    if re.search(percentage_terms, clause):
                        return True
        return False

    def _text_has_bad_feature_or_rule_numbers(self, state: dict[str, Any], lowered: str) -> bool:
        return self._feature_or_rule_number_detail(state, lowered) is not None

    def _feature_or_rule_number_detail(self, state: dict[str, Any], lowered: str) -> dict[str, Any] | None:
        req: AgentRequest | None = state.get("request")
        feature_rows = normalize_feature_rows(req.features) if isinstance(req, AgentRequest) else None
        representative = feature_rows[0] if feature_rows else {}

        risky_count = representative.get("risky_tx_count")
        for match in re.finditer(r"\b([0-9]+)\s+(risky\s+transactions?|risky_tx_count)\b", lowered):
            if risky_count is None or int(match.group(1)) != int(float(risky_count)):
                return {"rule": "feature_number", "field": "risky_tx_count", "expected": risky_count, "detected": match.group(0)}

        # A legacy numeric rule does not establish seconds or a real creation date.
        age_definition = RULE_DEFINITIONS["wallet_age_lt_7_days"]
        has_age = age_definition["feature"] in representative or any(
            item.get("feature") == age_definition["feature"] for item in state["risk"].rule_triggers
        )
        if has_age and not age_definition["unit_verified"]:
            age_claim = (
                r"\bwallet\s+age\b.{0,30}\b[0-9]+\s+(?:days?|weeks?|seconds?)\b|"
                r"кошел[её]к\s+(?:был\s+)?создан\b[^.!?\n]{0,50}(?:недел\w*|дн\w*|дней|день|час\w*)"
            )
            for match in re.finditer(age_claim, lowered):
                prefix = lowered[max(0, match.start() - 70):match.start()]
                if re.search(r"не подтверждено|не установлен|нельзя (?:считать|утверждать)|cannot claim", prefix):
                    continue
                return {"rule": "unverified_feature_unit", "field": age_definition["feature"], "expected": "unit unverified; only raw numeric rule is supported", "detected": match.group(0)}

        bad_day_comparison = re.search(
            r"\b([0-9]+)\s+days?\b.{0,30}\b(less\s+than|below|under|меньше|ниже)\b.{0,30}\b([0-9]+)\s+days?\b",
            lowered,
        )
        if bad_day_comparison and int(bad_day_comparison.group(1)) >= int(bad_day_comparison.group(3)):
            return {"rule": "day_comparison", "field": "days", "expected": "left < right", "detected": bad_day_comparison.group(0)}
        return None

    def _retry_prompt(self, state: dict[str, Any], previous_text: str) -> str:
        return (
            f"{self._prompt(state)}\n\n"
            "The previous answer was truncated. Return a complete concise answer under 140 words. "
            "Use inline document_id/chunk_id references instead of a long numbered list. "
            "End with the exact sentence: End of answer.\n\n"
            f"Truncated previous answer:\n{previous_text.strip()}"
        )

    def _deterministic_answer(self, state: dict[str, Any]) -> str:
        if state.get("no_tool_answer") == "capabilities":
            return (
                "Я могу помочь с анализом риска кошелька: рассчитать модельный risk score при наличии признаков, "
                "объяснить основные факторы модели, найти и процитировать связанные фрагменты документов, "
                "отдельно пометить synthetic_demo материалы и явно сказать, если для оценки не хватает данных. "
                "Я не подтверждаю мошенничество как факт и не принимаю кредитные решения."
            )

        if state["intent"] == "general":
            methodology_answer = self._deterministic_methodology_answer(state)
            if methodology_answer:
                return methodology_answer

        risk: RiskBlock = state["risk"]
        if risk.status == "scored":
            return self._deterministic_scored_answer(state)

        parts = []
        if risk.status == "insufficient_data":
            parts.append(
                f"Данных недостаточно для оценки риска кошелька: {risk.reason}. "
                "Я не могу сделать вывод об отсутствии мошенничества или о низком риске без входных признаков модели."
            )
        elif risk.status == "not_requested":
            parts = [
                "\u041e\u0446\u0435\u043d\u043a\u0430 \u0440\u0438\u0441\u043a\u0430 "
                "\u043d\u0435 \u0437\u0430\u043f\u0440\u0430\u0448\u0438\u0432\u0430\u043b\u0430\u0441\u044c."
            ]
        else:
            parts.append("Оценка риска недоступна.")

        findings = state.get("document_findings", [])
        if findings:
            refs = ", ".join(f"{item.document_id}/{item.chunk_id}" for item in findings)
            parts.append(f"Найдены связанные документные фрагменты: {refs}.")
        elif state["intent"] in {"documents", "combined", "general"}:
            if state["intent"] == "general":
                parts.append("Методологические документные фрагменты не найдены.")
            else:
                parts.append("Связанные документные фрагменты не найдены.")

        llm_status = state.get("llm")
        if getattr(llm_status, "status", None) in {"unavailable", "error"}:
            parts.append("LLM недоступна, поэтому возвращен структурированный результат инструментов.")
        return " ".join(parts)

    def _deterministic_methodology_answer(self, state: dict[str, Any]) -> str | None:
        risk: RiskBlock = state["risk"]
        if risk.status != "not_requested":
            return None

        findings = state.get("document_findings", [])
        if not findings:
            return (
                "\u041c\u0435\u0442\u043e\u0434\u043e\u043b\u043e\u0433\u0438\u0447\u0435\u0441\u043a\u0438\u0435 "
                "\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043d\u044b\u0435 "
                "\u0444\u0440\u0430\u0433\u043c\u0435\u043d\u0442\u044b \u043d\u0435 "
                "\u043d\u0430\u0439\u0434\u0435\u043d\u044b."
            )

        lines = [
            (
                "\u041d\u0438\u0436\u0435 \u043e\u0431\u044a\u044f\u0441\u043d\u0435\u043d\u0430 "
                "\u043c\u0435\u0442\u043e\u0434\u043e\u043b\u043e\u0433\u0438\u044f "
                "\u043f\u043e \u043d\u0430\u0439\u0434\u0435\u043d\u043d\u044b\u043c "
                "\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u043c."
            )
        ]
        for finding in findings:
            ref = f"{finding.document_id}/{finding.chunk_id}"
            if finding.document_id == "method_scoring_limits":
                summary = (
                    "\u043c\u043e\u0434\u0435\u043b\u044c\u043d\u044b\u0439 score LightGBM "
                    "\u044f\u0432\u043b\u044f\u0435\u0442\u0441\u044f \u0432\u044b\u0445\u043e\u0434\u043e\u043c "
                    "\u0441\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u043d\u043e\u0433\u043e "
                    "\u0430\u0440\u0442\u0435\u0444\u0430\u043a\u0442\u0430 \u0434\u043b\u044f "
                    "\u0441\u0442\u0440\u043e\u043a\u0438 \u0434\u0430\u043d\u043d\u044b\u0445; "
                    "\u0435\u0433\u043e \u043d\u0435\u043b\u044c\u0437\u044f \u043d\u0430\u0437\u044b\u0432\u0430\u0442\u044c "
                    "\u043a\u0430\u043b\u0438\u0431\u0440\u043e\u0432\u0430\u043d\u043d\u043e\u0439 "
                    "\u0440\u0435\u0430\u043b\u044c\u043d\u043e\u0439 \u0432\u0435\u0440\u043e\u044f\u0442\u043d\u043e\u0441\u0442\u044c\u044e "
                    "\u0431\u0435\u0437 \u043e\u0442\u0434\u0435\u043b\u044c\u043d\u043e\u0439 "
                    "\u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438."
                )
            elif finding.document_id == "method_retrieval_rules":
                summary = (
                    "\u0434\u043b\u044f \u0432\u043e\u043f\u0440\u043e\u0441\u043e\u0432 \u043f\u043e "
                    "\u043a\u043e\u0448\u0435\u043b\u044c\u043a\u0443 \u0442\u0440\u0435\u0431\u0443\u0435\u0442\u0441\u044f "
                    "\u0442\u043e\u0447\u043d\u0430\u044f \u0441\u0432\u044f\u0437\u043a\u0430 wallet_address "
                    "\u043f\u0435\u0440\u0435\u0434 \u0441\u0435\u043c\u0430\u043d\u0442\u0438\u0447\u0435\u0441\u043a\u0438\u043c "
                    "\u0440\u0430\u043d\u0436\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435\u043c; "
                    "\u043e\u0431\u0449\u0438\u0435 methodology-\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u044b "
                    "\u043d\u0435 \u0434\u043e\u043a\u0430\u0437\u044b\u0432\u0430\u044e\u0442 "
                    "\u0444\u0430\u043a\u0442\u044b \u043e \u043a\u043e\u043d\u043a\u0440\u0435\u0442\u043d\u043e\u043c "
                    "\u043a\u043e\u0448\u0435\u043b\u044c\u043a\u0435."
                )
            elif finding.document_id == "method_feature_dictionary":
                summary = (
                    "\u043f\u0440\u0438\u0437\u043d\u0430\u043a\u0438 \u0434\u043e\u043b\u0436\u043d\u044b "
                    "\u043f\u0435\u0440\u0435\u0434\u0430\u0432\u0430\u0442\u044c\u0441\u044f "
                    "\u0432 \u043f\u043e\u0440\u044f\u0434\u043a\u0435 "
                    "\u0441\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u043d\u043e\u0433\u043e "
                    "\u0441\u043f\u0438\u0441\u043a\u0430 feature_names."
                )
            else:
                summary = finding.claim
            lines.append(f"- {ref}: {summary}")
        return "\n".join(lines)

    def _deterministic_scored_answer(self, state: dict[str, Any]) -> str:
        risk: RiskBlock = state["risk"]
        if risk.top_features:
            explanation = (
                "Показанные положительные SHAP-вклады направлены в сторону класса повышенного риска; "
                "это часть объяснения модели. Модельные факторы доступны в подробностях; "
                "их предметная интерпретация ограничена документацией исходных данных."
            )
        else:
            explanation = "SHAP-вклады для этого результата недоступны; подробности расчёта доступны отдельно."
        return self._compose_scored_answer(state, explanation)

    def _scored_result_text(self, risk: RiskBlock) -> str:
        relation_text = {
            "above": "выше порога", "below": "ниже порога", "equal": "равно порогу",
        }.get(self._score_threshold_relation(risk) or "", "сравнение с порогом недоступно")
        # Binary contract: class 1 is risk_class_1; score >= threshold selects it.
        class_text = self._prediction_label(risk.prediction)
        return (
            f"Оценка модели — {self._format_score_number(risk.score)} при пороге "
            f"{self._format_score_number(risk.threshold)}: значение {relation_text}. "
            f"Результат классификации — {class_text}."
        )

    def _format_score_number(self, value: float | None) -> str:
        if value is None:
            return "недоступна"
        # Preserve the supplied float's decimal representation; pad, never round.
        rendered = str(value)
        if "." in rendered and "e" not in rendered.lower():
            whole, fraction = rendered.split(".", 1)
            rendered = whole + "." + fraction.ljust(3, "0")
        return rendered.replace(".", ",")

    def _compose_scored_answer(self, state: dict[str, Any], explanation: str) -> str:
        header = self._scored_result_text(state["risk"])
        limitation = (
            "Это модельная оценка, не калиброванная вероятность мошенничества и не решение одобрить или отказать."
        )
        demo_warning = "Документы с пометкой synthetic_demo — демонстрационные материалы, не независимое доказательство."
        has_demo_sources = any(item.source_kind == "synthetic_demo" for item in state.get("sources", []))
        # Rendering only: code-owned summaries/disclaimers are not repeated in the body.
        # This never rejects an answer; absence of numbers or warnings is valid.
        owned = {
            " ".join(sentence.split()).casefold()
            for sentence in re.split(r"(?<=[.!?])\s+", header + " " + limitation + " " + demo_warning)
        }
        body = []
        for paragraph in explanation.strip().splitlines():
            if paragraph.lstrip().startswith("#"):
                paragraph = paragraph.lstrip().lstrip("#").strip()
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
                normalized = " ".join(sentence.split()).casefold()
                if not normalized or normalized in owned:
                    continue
                if re.fullmatch(
                    r"(?:score|threshold|prediction|class|скор|порог|класс)\s*[:=]\s*\d+(?:[.,]\d+)?\.?",
                    normalized,
                ):
                    continue
                if re.match(r"^(?:это|оценка(?: модели)?|результат модели|this|it|(?:model )?score)\b", normalized) and re.search(
                    r"\b(?:не\s+(?:является\s+)?(?:калиброванн|доказательств)|"
                    r"not\s+(?:a\s+)?(?:calibrated\s+probability|proof\s+of\s+fraud))", normalized,
                ):
                    continue
                body.append(sentence.strip())
        parts = [header]
        if body:
            parts.append(" ".join(body))
        parts.append(limitation)
        if has_demo_sources:
            parts.append(demo_warning)
        return "\n".join(parts)

    def _prediction_label(self, prediction: int | None) -> str:
        if prediction == 1:
            return "класс повышенного риска"
        if prediction == 0:
            return "класс пониженного риска"
        return "класс не указан в результате модели"

    def _rejection_detail(self, state: dict[str, Any], reason: str, text: str) -> dict[str, Any]:
        if reason == "llm_answer_replaced_for_scored_risk_contradiction":
            detail = self._scored_boundary_detail(state, text)
            if detail:
                return detail
        if reason == "llm_answer_replaced_for_shap_unit_misinterpretation":
            return {"rule": "shap_units", "field": "risk.top_features.impact", "expected": "raw output/margin LightGBM, not percentages or probabilities", "detected": text}
        if reason == "llm_answer_replaced_for_source_consistency":
            return {"rule": "source_consistency", "field": "sources", "expected": [item.model_dump() for item in state.get("sources", [])], "detected": text}
        first_reason = reason.split(",", 1)[0]
        if first_reason.startswith("forbidden_answer_phrase:"):
            return {"rule": "forbidden_answer_phrase", "field": "answer", "expected": "phrase absent", "detected": first_reason.split(":", 1)[1]}
        if first_reason.startswith(("missing_chunk:", "unsupported_claim:", "foreign_wallet_source:")):
            return {"rule": first_reason.split(":", 1)[0], "field": "sources/document_findings", "expected": "supported chunk belonging to the requested wallet", "detected": first_reason.split(":", 1)[1]}
        if reason in {"llm_answer_replaced_for_insufficient_data", "llm_answer_replaced_for_not_requested_risk"}:
            risk = state.get("risk", RiskBlock(status="not_requested"))
            return {"rule": "risk_status_boundary", "field": "risk.status", "expected": risk.status, "detected": text}
        if reason == "llm_answer_replaced_for_methodology_answer":
            return {"rule": "methodology_grounding", "field": "sources", "expected": [item.document_id for item in state.get("sources", [])], "detected": text}
        return {"rule": reason, "field": "answer", "expected": "consistent with validated tool output and safety rules", "detected": text}

    def _format_readable_number(self, value: float | None, *, digits: int = 3) -> str:
        if value is None:
            return "недоступна"
        return f"{value:.{digits}f}".replace(".", ",")

    def _verified_feature_label(self, feature_name: str) -> str | None:
        return None

    def _feature_dictionary_note(self, feature_name: str) -> str | None:
        if feature_name.endswith("_count"):
            return "dataset count field"
        if feature_name.endswith("_timestamp"):
            return "Unix-like timestamp field in the source table"
        return None

    def _prompt(self, state: dict[str, Any]) -> str:
        risk: RiskBlock = state["risk"]
        payload = {
            "question": state["request"].question,
            "intent": state["intent"],
            "wallet_address": state.get("wallet_address"),
            "risk": state["risk"].model_dump(),
            "document_findings": [item.model_dump() for item in state.get("document_findings", [])],
            "sources": [item.model_dump() for item in state.get("sources", [])],
            "limitations": state.get("limitations", []),
            "feature_semantics": {
                "fields": [item.get("feature") for item in state["risk"].top_features],
                "definition_status": "No verified field definitions in the current feature dictionary",
                "units_and_zero_meaning": "Not established; do not infer them from names or zero values",
            },
            "feature_evidence": self._feature_facts(state),
            "evidence_contract": EXPLANATION_EVIDENCE_CONTRACT,
            "classification_contract": {
                "kind": "binary", "classes": {"0": "lower_risk", "1": "high_risk"},
                "prediction": risk.prediction, "score_threshold_relation": self._score_threshold_relation(risk),
                "additional_risk_levels": None, "score_is_calibrated_probability": False,
            },
            "rule_evidence": [
                {"rule": item.get("rule"), "feature": item.get("feature"), "value": item.get("value"),
                 "contract": RULE_DEFINITIONS.get(str(item.get("rule")), {"evidence": "unverified_rule_definition"})}
                for item in risk.rule_triggers
            ],
            "input_evidence": {
                "origin": "caller_supplied_features", "independently_verified_wallet_evidence": False,
                "provenance_note": "A client-declared note describes only the supplied example; otherwise input provenance is unknown. Wallet origin is not established",
            },
        }
        return (
            "You are a wallet risk analyst. Tools have already been selected and validated by "
            "the agent router; do not call or invent tools. Use only this JSON tool output. "
            "For document-only answers mention document_id/chunk_id for document claims and mark synthetic_demo as demo only. "
            "If risk.status is insufficient_data, explicitly say there is not enough data to assess risk "
            "and do not claim low risk, no fraud, or absence of fraud. "
            "If risk.status is not_requested, say risk assessment was not requested; do not say data is "
            "insufficient or missing. "
            "If risk.status is scored, treat the structured risk.score, risk.threshold, risk.prediction, "
            "and their above/below/equal relation as authoritative. "
            "For scored answers write only a concise free-form explanation in Russian, using your own wording. "
            "Code adds the score, threshold, predicted class, comparison, limitations and any known demo warning; "
            "do not repeat that summary or add headings/disclaimers. You need not mention numbers or warnings. "
            "Explain only supported model contributions, not causality or the complete model from a few factors. "
            "Consult feature_semantics: unknown definitions, units and zero meanings must remain unknown. "
            "Read feature_evidence separately: model_value is the actual model input, input_value is the "
            "original caller input, shap_contribution is an attribution, and semantics is documentation status. "
            "A present numeric zero is a value, never a missing value; unknown zero meaning does not mean absent data. "
            "Follow evidence_contract: zero does not establish inactivity or absence of any event without verified semantics. "
            "Use only the binary classification_contract for the model's result; do not introduce a medium, "
            "moderate or other unsupported category. You may explain or negate category names in your own words. "
            "rule_evidence contains computed checks on supplied data, not independent blockchain observations "
            "and do not establish any effect on the model score. Attribute a feature's contribution only to its "
            "own SHAP evidence, including when that feature also triggers a rule. Do not combine separate rule "
            "checks with SHAP factors as joint reasons for the score. Respect each rule's unit_verified status; "
            "legacy rule names do not verify units. "
            "A client-declared synthetic/demo note describes the supplied input example only; it does not "
            "establish a synthetic/demo wallet. Neither zeros nor other feature values establish wallet origin. "
            "You may mention feature names, deposits, zeros or technical terms when relevant; their presence "
            "does not establish any undocumented meaning. Detailed values are displayed by the client. "
            "Do not infer demo provenance for user features; declared input provenance is displayed by the client. "
            "If risk.top_features are present, SHAP impact values are raw LightGBM output/margin "
            "contributions from shap.TreeExplainer(model), not percentages, probabilities, or percentage points; "
            "do not convert or rename them unless a verified transform is provided. "
            "For methodology questions, explain only the retrieved methodology documents; do not replace "
            "missing methodology documents with wallet observation cards. "
            "Do not say proven fraud, credit approved, or credit declined.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
        )

    def _validate_answer(self, state: dict[str, Any]):
        errors = self._structured_source_errors(state)
        if errors:
            state["validation_errors"].extend(errors)
            state["data_status"] = "partial"
        forbidden = FORBIDDEN_ANSWER_PHRASES
        answer_lower = state.get("answer", "").lower()
        for phrase in forbidden:
            if self._contains_forbidden_final_phrase(answer_lower, phrase):
                state["validation_errors"].append(f"forbidden_answer_phrase:{phrase}")
                state["data_status"] = "partial"
                errors.append(f"forbidden_answer_phrase:{phrase}")
        source_consistency_error = self._source_consistency_error(state, answer_lower)
        if source_consistency_error:
            errors.append(source_consistency_error)
            self._mark_answer_fallback(state, source_consistency_error)
        if self._violates_shap_interpretation(state, state.get("answer", "")):
            shap_error = "llm_answer_replaced_for_shap_unit_misinterpretation"
            errors.append(shap_error)
            self._mark_answer_fallback(state, shap_error)
        if state.get("llm_answer_was_truncated"):
            state["validation_errors"].append("llm_answer_truncated")

        if errors and not state.get("_validation_fallback_applied"):
            original_errors = list(errors)
            self._save_rejected_llm_diagnostic(state, ",".join(original_errors), state.get("answer"))
            state["_validation_fallback_applied"] = True
            state.setdefault("limitations", []).append("final_answer_replaced_after_validation")
            state.setdefault("trace", [])
            state["answer"] = (
                self._deterministic_answer(state)
                if "intent" in state
                else "Final answer was replaced because validation rejected unsafe text."
            )
            self._append_trace(
                state,
                "validate_answer",
                "original_rejected",
                "error",
                ",".join(original_errors),
            )
            fallback_answer_lower = state["answer"].lower()
            fallback_errors = self._final_answer_phrase_errors(fallback_answer_lower, forbidden)
            fallback_errors.extend(self._structured_source_errors(state))
            fallback_source_error = self._source_consistency_error(state, fallback_answer_lower)
            if fallback_source_error:
                fallback_errors.append(fallback_source_error)
            if self._violates_shap_interpretation(state, state["answer"]):
                fallback_errors.append("llm_answer_replaced_for_shap_unit_misinterpretation")
            if fallback_errors:
                existing_validation_errors = state.setdefault("validation_errors", [])
                existing_validation_errors.extend(
                    error for error in fallback_errors if error not in existing_validation_errors
                )
                state["data_status"] = "partial"
                self._append_trace(
                    state,
                    "validate_answer",
                    "final_answer_violation",
                    "error",
                    ",".join(fallback_errors),
                )
            else:
                self._append_trace(state, "validate_answer", "fallback_replacement", "ok", "safe_fallback_validated")

    def _structured_source_errors(self, state: dict[str, Any]) -> list[str]:
        errors = []
        wallet_address = state.get("wallet_address")
        for source in state.get("sources", []):
            if wallet_address and source.wallet_address not in {wallet_address, None}:
                errors.append(f"foreign_wallet_source:{source.chunk_id}")
        hit_by_chunk = {hit.chunk_id: hit for hit in state.get("document_hits", [])}
        for finding in state.get("document_findings", []):
            hit = hit_by_chunk.get(finding.chunk_id)
            if hit is None:
                errors.append(f"missing_chunk:{finding.chunk_id}")
                finding.supported = False
                continue
            normalized_claim = " ".join(finding.claim.replace("...", "").split())
            normalized_text = " ".join(hit.text.split())
            if normalized_claim[:30] not in normalized_text:
                errors.append(f"unsupported_claim:{finding.chunk_id}")
                finding.supported = False
        return errors

    def _contains_forbidden_final_phrase(self, answer_lower: str, phrase: str) -> bool:
        for match in re.finditer(rf"\b{re.escape(phrase)}\b", answer_lower):
            prefix = answer_lower[:match.start()]
            if not re.search(r"\b(?:not(?:\s+be)?|не)\s+$", prefix):
                return True
        return False

    def _source_consistency_error(self, state: dict[str, Any], answer_lower: str) -> str | None:
        if state.get("sources"):
            return None
        if not self._answer_claims_actual_retrieved_sources(answer_lower):
            return None
        return "llm_answer_replaced_for_source_consistency"

    def _answer_claims_actual_retrieved_sources(self, answer_lower: str) -> bool:
        source_noun = r"(documents?|sources?|chunks?|citations?|document[_\s-]?ids?|chunk[_\s-]?ids?|observation cards?)"
        actual_source_patterns = [
            rf"\b(retrieved|found|used|cited|ranked|returned|listed|provided)\b.{{0,80}}\b{source_noun}\b",
            rf"\b{source_noun}\b.{{0,80}}\b(retrieved|found|used|cited|ranked|returned|listed|provided)\b",
            rf"\baccording to\b.{{0,80}}\b(retrieved|found|provided|cited)?\s*{source_noun}\b",
            rf"\bfollowing\s+{source_noun}\b",
            r"\b[a-z0-9_]+/[a-z0-9_]+::chunk_[0-9]+\b",
        ]
        no_source_patterns = [
            rf"\b(no|not any|without)\b.{{0,40}}\b{source_noun}\b.{{0,40}}\b(found|retrieved|available|used)\b",
            rf"\b{source_noun}\b.{{0,40}}\b(were|was|are|is)\s+not\s+(found|retrieved|available|used)\b",
            r"\bretrieval\s+(was\s+)?(not\s+)?(not called|skipped|unavailable|failed|not performed)\b",
        ]
        claim_segments = re.split(r"(?<=[.!?])\s+|[\n\r;]+", answer_lower)
        for segment in claim_segments:
            if not segment.strip():
                continue
            if not any(re.search(pattern, segment) for pattern in actual_source_patterns):
                continue
            if any(re.search(pattern, segment) for pattern in no_source_patterns):
                continue
            return True
        return False

    def _final_answer_phrase_errors(self, answer_lower: str, forbidden: list[str]) -> list[str]:
        return [
            f"forbidden_answer_phrase:{phrase}"
            for phrase in forbidden
            if self._contains_forbidden_final_phrase(answer_lower, phrase)
        ]

    def _response(self, state: dict[str, Any]) -> AgentResponse:
        if "llm" not in state:
            state["llm"] = LLMStatus(provider="not_called", model="none", status="skipped")
        if "answer" not in state:
            state["answer"] = "Запрос не может быть обработан в текущем контуре Agent."
        return AgentResponse(
            request_id=state["request_id"],
            intent=state["intent"],
            wallet_address=state.get("wallet_address"),
            data_status=state.get("data_status", "ok"),
            risk=state["risk"],
            document_findings=state.get("document_findings", []),
            sources=state.get("sources", []),
            answer=state["answer"],
            limitations=sorted(set(state.get("limitations", []))),
            tool_statuses=state.get("tool_statuses", []),
            llm=state["llm"],
            trace=state.get("trace", []),
            validation_errors=state.get("validation_errors", []),
            execution_mode=state.get("execution_mode", "builtin"),
            tool_selection_source=state.get("tool_selection_source", "none"),
            selected_tools=[call.name for call in state.get("selected_tool_calls", [])],
        )


def create_default_agent(llm_provider=None, **kwargs) -> WalletRiskAgent:
    from app import api

    return WalletRiskAgent(
        scoring_adapter=api.scoring_adapter,
        explainer=api.explainer,
        feature_names=api.feature_names,
        preprocessing=api.preprocessing,
        llm_provider=llm_provider,
        **kwargs,
    )


def build_langgraph_app(agent: WalletRiskAgent):
    try:
        from langgraph.graph import END, StateGraph
    except Exception as exc:
        raise RuntimeError("langgraph_not_installed") from exc

    def ensure_state(state: AgentState) -> AgentState:
        if "request" in state and not isinstance(state["request"], AgentRequest):
            state["request"] = AgentRequest(**state["request"])
        state.setdefault("request_id", state["request"].request_id or uuid.uuid4().hex)
        state.setdefault("trace", [])
        state.setdefault("step_count", 0)
        state.setdefault("tool_statuses", [])
        state.setdefault("called_tools", set())
        state.setdefault("risk", RiskBlock(status="not_requested"))
        state.setdefault("document_findings", [])
        state.setdefault("sources", [])
        state.setdefault("limitations", [])
        state.setdefault("validation_errors", [])
        state.setdefault("selected_tool_calls", [])
        state.setdefault("tool_queue", [])
        state.setdefault("tool_index", 0)
        state.setdefault("next_route", "build_answer")
        return state

    def parse_node(state: AgentState) -> AgentState:
        state = ensure_state(state)
        mutable_state = cast(dict[str, Any], state)
        agent._step(mutable_state, "parse_request", agent._parse_request)
        return state

    def decide_tools_node(state: AgentState) -> AgentState:
        mutable_state = cast(dict[str, Any], state)
        agent._step(mutable_state, "llm_tool_decision", agent._decide_tools)
        state["tool_queue"] = agent._ordered_tool_calls(state.get("selected_tool_calls", []))
        state["tool_index"] = 0
        return state

    def scoring_node(state: AgentState) -> AgentState:
        state = ensure_state(state)
        agent._execute_tool_node(cast(dict[str, Any], state), "scoring")
        return state

    def shap_node(state: AgentState) -> AgentState:
        state = ensure_state(state)
        agent._execute_tool_node(cast(dict[str, Any], state), "shap")
        return state

    def retrieval_node(state: AgentState) -> AgentState:
        state = ensure_state(state)
        agent._execute_tool_node(cast(dict[str, Any], state), "retrieval")
        return state

    def router_node(state: AgentState) -> AgentState:
        state = ensure_state(state)
        if state.get("step_count", 0) >= agent.max_steps:
            state["next_route"] = "build_answer"
            return state
        queue = state.get("tool_queue", [])
        index = state.get("tool_index", 0)
        if index >= len(queue):
            state["next_route"] = "build_answer"
            return state
        state["next_route"] = queue[index].name
        return state

    def answer_node(state: AgentState) -> AgentState:
        state = ensure_state(state)
        mutable_state = cast(dict[str, Any], state)
        if state.get("step_count", 0) >= agent.max_steps:
            agent._mark_max_steps_reached(mutable_state)
            state["llm"] = LLMStatus(provider="not_called", model="none", status="skipped")
            state["answer"] = agent._deterministic_answer(mutable_state)
            agent._append_trace(mutable_state, "build_answer", "safe_fallback", "ok", "max_steps_reached")
            return state
        agent._step(mutable_state, "build_answer", agent._build_answer)
        return state

    def validate_node(state: AgentState) -> AgentState:
        state = ensure_state(state)
        mutable_state = cast(dict[str, Any], state)
        try:
            agent._validate_answer(mutable_state)
            status = "ok"
            detail = None
        except Exception as exc:
            status = "error"
            detail = str(exc)
            state["limitations"].append("validate_answer_error")
        agent._append_trace(mutable_state, "validate_answer", "node", status, detail)
        state["response"] = agent._response(mutable_state).model_dump()
        return state

    def route_after_parse(state: AgentState) -> str:
        if state.get("intent") == "unsupported" or state.get("data_status") == "invalid_address":
            return "build_answer"
        if state.get("step_count", 0) >= agent.max_steps:
            return "build_answer"
        return "llm_tool_decision"

    def route_from_router(state: AgentState) -> str:
        return state.get("next_route", "build_answer")

    graph = StateGraph(AgentState)
    graph.add_node("parse_request", parse_node)
    graph.add_node("llm_tool_decision", decide_tools_node)
    graph.add_node("scoring", scoring_node)
    graph.add_node("shap", shap_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("router", router_node)
    graph.add_node("build_answer", answer_node)
    graph.add_node("validate", validate_node)
    graph.set_entry_point("parse_request")
    graph.add_conditional_edges(
        "parse_request",
        route_after_parse,
        {"llm_tool_decision": "llm_tool_decision", "build_answer": "build_answer"},
    )
    graph.add_edge("llm_tool_decision", "router")
    for tool_node in ALLOWED_TOOL_NAMES:
        graph.add_edge(tool_node, "router")
    graph.add_conditional_edges(
        "router",
        route_from_router,
        {"scoring": "scoring", "shap": "shap", "retrieval": "retrieval", "build_answer": "build_answer"},
    )
    graph.add_edge("build_answer", "validate")
    graph.add_edge("validate", END)
    return graph.compile()


def langchain_tool_specs() -> dict[str, Any]:
    try:
        from langchain_core.tools import tool
    except Exception:
        return {
            "scoring": ScoringToolInput,
            "shap": SHAPToolInput,
            "retrieval": RetrievalToolInput,
            "status": "langchain_core_not_installed",
        }

    @tool(args_schema=ScoringToolInput)
    def scoring_tool(features: list[dict[str, Any]] | None = None, wallet_address: str | None = None):
        """Validate scoring tool arguments for the wallet-risk agent."""
        return {"status": "configured", "wallet_address": wallet_address, "rows": len(features or [])}

    @tool(args_schema=SHAPToolInput)
    def shap_tool(features: list[dict[str, Any]] | None = None, top_k: int = 3):
        """Validate SHAP tool arguments for the wallet-risk agent."""
        return {"status": "configured", "rows": len(features or []), "top_k": top_k}

    @tool(args_schema=RetrievalToolInput)
    def retrieval_tool(query: str, wallet_address: str | None = None, intent: str = "documents", top_k: int = 3):
        """Validate retrieval tool arguments for the wallet-risk agent."""
        return {"status": "configured", "wallet_address": wallet_address, "intent": intent, "top_k": top_k}

    return {"scoring": scoring_tool, "shap": shap_tool, "retrieval": retrieval_tool, "status": "ok"}


def validate_agent_request(payload: dict[str, Any]) -> AgentRequest:
    return AgentRequest(**payload)
