import os
import re
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import Any, Callable, Literal

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.inference import build_model_input, get_rule_triggers, get_top_risk_factors
from src.retrieval import DEFAULT_INDEX_PATH, LocalDenseRetriever, RetrievalHit


PROMPT_VERSION = "wallet_risk_agent_prompt_v1"
AGENT_VERSION = "wallet_risk_agent_v1"
ADDRESS_RE = re.compile(r"\b0x[a-fA-F0-9]{40}\b")
INTENTS = {"score", "explain", "documents", "combined", "general", "unsupported"}
ALLOWED_TOOL_NAMES = ("scoring", "shap", "retrieval")


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
    asks_wallet_assessment = any(
        term in q
        for term in [
            "risk score for",
            "score for",
            "score wallet",
            "calculate",
            "compute",
            "probability for",
            "скор для",
            "риск для",
            "рассч",
            "посч",
            "оцени кошелек",
            "оценить кошелек",
        ]
    )

    if asks_general and not asks_wallet_assessment:
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
        retriever: LocalDenseRetriever | None = None,
        llm_provider=None,
        max_steps: int = 8,
        tool_timeout_seconds: float = 3.0,
    ):
        self.scoring_adapter = scoring_adapter
        self.explainer = explainer
        self.feature_names = feature_names or []
        self.preprocessing = preprocessing or {}
        self.retriever = retriever or LocalDenseRetriever(DEFAULT_INDEX_PATH)
        self.llm_provider = llm_provider or OllamaProvider()
        self.max_steps = max_steps
        self.tool_timeout_seconds = tool_timeout_seconds

    def run(self, request: AgentRequest | dict[str, Any]) -> AgentResponse:
        req = request if isinstance(request, AgentRequest) else AgentRequest(**request)
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
            "limitations": [],
            "validation_errors": [],
        }

        self._step(state, "parse_request", self._parse_request)
        if state.get("intent") == "unsupported" or state.get("data_status") == "invalid_address":
            return self._response(state)

        if state["intent"] in {"score", "explain", "combined"}:
            self._step(state, "call_scoring_tool", self._call_scoring)
        if state["intent"] in {"explain", "combined"} and state["risk"].status == "scored":
            self._step(state, "call_shap_tool", self._call_shap)
        if state["intent"] in {"documents", "combined", "general"}:
            self._step(state, "call_retrieval_tool", self._call_retrieval)

        self._step(state, "build_answer", self._build_answer)
        self._step(state, "validate_answer", self._validate_answer)
        return self._response(state)

    def run_langgraph(self, request: AgentRequest | dict[str, Any]) -> AgentResponse:
        req = request if isinstance(request, AgentRequest) else AgentRequest(**request)
        graph = build_langgraph_app(self)
        state = graph.invoke({"request": req.model_dump()})
        response = AgentResponse(**state["response"])
        response.execution_mode = "langgraph"
        return response

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

        started = time.perf_counter()
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func)
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
            executor.shutdown(wait=False, cancel_futures=True)

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
        for call in decision.tool_calls:
            if call.name not in ALLOWED_TOOL_NAMES:
                invalid_names.append(call.name)
                continue
            if not self._tool_allowed_for_intent(state, call.name):
                intent_removed.append(call.name)
                continue
            valid_calls.append(call)

        coverage_added = []
        if decision.status == "ok" and valid_calls:
            selected_calls = dedupe_tool_calls(valid_calls)
            selected_names = {call.name for call in selected_calls}
            for required_call in self._deterministic_tool_calls(state):
                if required_call.name not in selected_names:
                    selected_calls.append(required_call)
                    selected_names.add(required_call.name)
                    coverage_added.append(required_call.name)
            state["selected_tool_calls"] = dedupe_tool_calls(selected_calls)
            state["tool_selection_source"] = "llm_tool_calling"
        else:
            state["selected_tool_calls"] = self._deterministic_tool_calls(state)
            state["tool_selection_source"] = "deterministic_fallback"
            state["limitations"].append("tool_selection_fallback")

        if invalid_names:
            state["limitations"].append("invalid_llm_tool_name")
        if intent_removed:
            state["limitations"].append("tool_selection_intent_guard")
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
            f"coverage_added={','.join(coverage_added) or 'none'}"
        )

    def _tool_allowed_for_intent(self, state: dict[str, Any], tool_name: str) -> bool:
        intent = state.get("intent")
        if intent == "general":
            return tool_name == "retrieval"
        if intent == "documents":
            return tool_name in {"scoring", "retrieval"}
        if intent == "score":
            return tool_name == "scoring"
        if intent == "explain":
            return tool_name in {"scoring", "shap"}
        if intent == "combined":
            return True
        return False

    def _execute_selected_tools(self, state: dict[str, Any]):
        order = {"scoring": 0, "shap": 1, "retrieval": 2}
        calls = sorted(state.get("selected_tool_calls", []), key=lambda call: order[call.name])
        if not calls:
            state["_trace_detail"] = "selected=none"
            return

        for call in calls:
            if state.get("step_count", 0) >= self.max_steps:
                state["limitations"].append("max_steps_reached")
                self._append_trace(state, call.name, "tool_call_blocked", "error", "max_steps_reached")
                break
            state["step_count"] = state.get("step_count", 0) + 1

            validation_error = self._validate_selected_tool_arguments(state, call.name)
            if validation_error:
                state["limitations"].append(f"{call.name}_argument_error")
                self._append_trace(state, call.name, "tool_call", "error", validation_error)
                continue

            self._append_trace(state, call.name, "tool_call", "ok", f"arguments={call.arguments or {}}")
            before_status_count = len(state["tool_statuses"])
            if call.name == "scoring":
                self._call_scoring(state)
            elif call.name == "shap":
                self._call_shap(state)
            elif call.name == "retrieval":
                self._call_retrieval(state)

            status = state["tool_statuses"][-1] if len(state["tool_statuses"]) > before_status_count else None
            if status is not None:
                self._append_trace(
                    state,
                    call.name,
                    "tool_result",
                    status.status,
                    status.error or f"elapsed_ms={status.elapsed_ms:.3f}",
                )

        selected = ",".join(call.name for call in calls)
        state["_trace_detail"] = f"executed={selected}"

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

        scoring_result = result.value
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
            state["risk"].top_features = result.value["top_features"]
            state["risk"].rule_triggers = result.value["rule_triggers"]
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

        hits: list[RetrievalHit] = result.value
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

    def _build_answer(self, state: dict[str, Any]):
        if state.get("no_tool_answer") == "capabilities":
            state["llm"] = LLMStatus(provider="not_called", model="none", status="skipped")
            state["answer"] = self._deterministic_answer(state)
            return
        llm_status, llm_text = self._generate_answer(self._prompt(state))
        state["llm"] = llm_status
        if llm_text and llm_status.status == "ok":
            if not self._is_truncated_llm_answer(llm_status, llm_text):
                replacement_reason = self._answer_replacement_reason(state, llm_text)
                if replacement_reason:
                    state["limitations"].append(replacement_reason)
                    state["answer"] = self._deterministic_answer(state)
                    return
                state["answer"] = llm_text.strip()
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
                    state["limitations"].append(replacement_reason)
                    state["answer"] = self._deterministic_answer(state)
                    return
                state["answer"] = retry_text.strip()
                return

            state["llm_answer_was_truncated"] = True
            state["llm_fallback_reason"] = "truncated"
            state["limitations"].append("llm_answer_fallback_after_truncation")
        state["answer"] = self._deterministic_answer(state)
        if state.get("llm_fallback_reason") == "truncated":
            state["answer"] = (
                f"{state['answer']} LLM answer was truncated by generation limits; "
                "returning structured tool output instead."
            )

    def _answer_replacement_reason(self, state: dict[str, Any], text: str) -> str | None:
        if self._violates_insufficient_data_boundary(state, text):
            return "llm_answer_replaced_for_insufficient_data"
        if self._violates_not_requested_boundary(state, text):
            return "llm_answer_replaced_for_not_requested_risk"
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
            "insufficient data",
            "not enough information",
            "cannot assess risk",
            "can't assess risk",
            "no risk assessment can be made",
            "unable to assess risk",
            "missing data",
            "missing features",
            "данных недостаточно",
            "недостаточно данных",
        ]
        return any(phrase in lowered for phrase in phrases)

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

        parts = []
        risk: RiskBlock = state["risk"]
        if risk.status == "scored":
            parts.append(
                "Оценка риска рассчитана моделью: "
                f"score={risk.score:.6f}, threshold={risk.threshold:.6f}, prediction={risk.prediction}."
            )
        elif risk.status == "insufficient_data":
            parts.append(
                f"Данных недостаточно для оценки риска кошелька: {risk.reason}. "
                "Я не могу сделать вывод об отсутствии мошенничества или о низком риске без входных признаков модели."
            )
        elif risk.status == "not_requested":
            parts.append("Оценка риска не запрашивалась.")
        else:
            parts.append("Оценка риска недоступна.")

        if risk.status == "not_requested":
            parts = ["Оценка риска не запрашивалась."]

        if risk.status == "not_requested":
            parts = [
                "\u041e\u0446\u0435\u043d\u043a\u0430 \u0440\u0438\u0441\u043a\u0430 "
                "\u043d\u0435 \u0437\u0430\u043f\u0440\u0430\u0448\u0438\u0432\u0430\u043b\u0430\u0441\u044c."
            ]

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

    def _prompt(self, state: dict[str, Any]) -> str:
        payload = {
            "intent": state["intent"],
            "wallet_address": state.get("wallet_address"),
            "risk": state["risk"].model_dump(),
            "document_findings": [item.model_dump() for item in state.get("document_findings", [])],
            "sources": [item.model_dump() for item in state.get("sources", [])],
            "limitations": state.get("limitations", []),
        }
        return (
            "You are a wallet risk analyst. Tools have already been selected by deterministic "
            "intent routing; do not call or invent tools. Use only this JSON tool output. "
            "Mention document_id/chunk_id for document claims. Mark synthetic_demo as demo only. "
            "If risk.status is insufficient_data, explicitly say there is not enough data to assess risk "
            "and do not claim low risk, no fraud, or absence of fraud. "
            "If risk.status is not_requested, say risk assessment was not requested; do not say data is "
            "insufficient or missing. "
            "For methodology questions, explain only the retrieved methodology documents; do not replace "
            "missing methodology documents with wallet observation cards. "
            "Do not say proven fraud, credit approved, or credit declined.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
        )

    def _validate_answer(self, state: dict[str, Any]):
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
        if errors:
            state["validation_errors"].extend(errors)
            state["data_status"] = "partial"
        forbidden = ["proved fraud", "proven fraud", "доказано мошенничество", "credit approved", "credit declined"]
        answer_lower = state.get("answer", "").lower()
        for phrase in forbidden:
            if phrase in answer_lower:
                state["validation_errors"].append(f"forbidden_answer_phrase:{phrase}")
                state["data_status"] = "partial"
        if state.get("llm_answer_was_truncated"):
            state["validation_errors"].append("llm_answer_truncated")

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

    def ensure_state(state: dict[str, Any]):
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
        return state

    def parse_node(state: dict[str, Any]):
        state = ensure_state(state)
        agent._step(state, "parse_request", agent._parse_request)
        return state

    def decide_tools_node(state: dict[str, Any]):
        agent._step(state, "llm_tool_decision", agent._decide_tools)
        return state

    def execute_tools_node(state: dict[str, Any]):
        agent._step(state, "execute_tools", agent._execute_selected_tools)
        return state

    def scoring_node(state: dict[str, Any]):
        agent._step(state, "call_scoring_tool", agent._call_scoring)
        return state

    def shap_node(state: dict[str, Any]):
        agent._step(state, "call_shap_tool", agent._call_shap)
        return state

    def retrieval_node(state: dict[str, Any]):
        agent._step(state, "call_retrieval_tool", agent._call_retrieval)
        return state

    def answer_node(state: dict[str, Any]):
        agent._step(state, "build_answer", agent._build_answer)
        return state

    def validate_node(state: dict[str, Any]):
        agent._step(state, "validate_answer", agent._validate_answer)
        state["response"] = agent._response(state).model_dump()
        return state

    def route_after_parse(state: dict[str, Any]):
        if state.get("intent") == "unsupported" or state.get("data_status") == "invalid_address":
            return "answer"
        return "llm_tool_decision"

    graph = StateGraph(dict)
    graph.add_node("parse_request", parse_node)
    graph.add_node("llm_tool_decision", decide_tools_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("scoring", scoring_node)
    graph.add_node("shap", shap_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("answer", answer_node)
    graph.add_node("validate", validate_node)
    graph.set_entry_point("parse_request")
    graph.add_conditional_edges(
        "parse_request",
        route_after_parse,
        {"llm_tool_decision": "llm_tool_decision", "answer": "answer"},
    )
    graph.add_edge("llm_tool_decision", "execute_tools")
    graph.add_edge("execute_tools", "answer")
    graph.add_edge("answer", "validate")
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
    try:
        return AgentRequest(**payload)
    except ValidationError:
        raise
