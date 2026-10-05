import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_agent_rehearsal import (  # noqa: E402
    MISSING_OLLAMA_MODEL,
    OLLAMA_MODEL,
    build_agent,
    feature_row,
    first_wallet_address,
    trace_summary,
)
from src.agent import DocumentFinding, RiskBlock, SourceRef  # noqa: E402
from src.retrieval import RetrievalHit, load_chunks  # noqa: E402


RESULT_PATH = PROJECT_ROOT / "results" / "agent_composite_answer_audit.json"
REPORT_PATH = PROJECT_ROOT / "docs" / "agent_composite_answer_audit.md"


class BrokenRetriever:
    def search_bm25(self, *_args, **_kwargs):
        raise RuntimeError("qdrant_unavailable")


def normalize_text(value: str) -> str:
    return " ".join((value or "").replace("...", "").split())


def chunk_text_by_id() -> dict[str, str]:
    return {chunk["chunk_id"]: chunk["text"] for chunk in load_chunks()}


def document_ids(response: dict[str, Any]) -> set[str]:
    return {source["document_id"] for source in response["sources"]}


def tool_status(response: dict[str, Any], name: str) -> str | None:
    for status in response["tool_statuses"]:
        if status["name"] == name:
            return status["status"]
    return None


def grounding_checks(response: dict[str, Any], chunks: dict[str, str]) -> list[dict[str, Any]]:
    checks = []
    for finding in response["document_findings"]:
        chunk_id = finding["chunk_id"]
        claim = normalize_text(finding["claim"])
        chunk_text = normalize_text(chunks.get(chunk_id, ""))
        supported_by_text = bool(chunk_text) and claim[:30] in chunk_text
        checks.append(
            {
                "chunk_id": chunk_id,
                "document_id": finding["document_id"],
                "chunk_exists": chunk_id in chunks,
                "supported_flag": finding["supported"],
                "supported_by_text": supported_by_text,
                "source_kind": finding["source_kind"],
            }
        )
    return checks


def coverage_checks(name: str, response: dict[str, Any], chunks: dict[str, str]) -> list[dict[str, Any]]:
    ids = document_ids(response)
    grounded = grounding_checks(response, chunks)
    all_grounded = bool(grounded) and all(item["supported_by_text"] for item in grounded)

    if name == "wallet_score_factors_documents_boundary":
        return [
            {"part": "risk_score", "status": "ok" if response["risk"]["status"] == "scored" else "failed"},
            {"part": "model_factors", "status": "ok" if response["risk"]["top_features"] else "failed"},
            {"part": "document_sources", "status": "ok" if response["sources"] else "failed"},
            {
                "part": "source_kind_boundary",
                "status": "ok"
                if response["sources"]
                and all(source["source_kind"] in {"derived_data", "synthetic_demo"} for source in response["sources"])
                else "failed",
            },
            {"part": "document_claims_supported_by_chunk_text", "status": "ok" if all_grounded else "failed"},
        ]

    if name == "general_methodology_multi_fragment":
        return [
            {
                "part": "retrieval_rules_chunk",
                "status": "ok" if "method_retrieval_rules" in ids else "failed",
            },
            {
                "part": "scoring_limits_chunk",
                "status": "ok" if "method_scoring_limits" in ids else "failed",
            },
            {
                "part": "feature_dictionary_chunk",
                "status": "ok" if "method_feature_dictionary" in ids else "failed",
            },
            {"part": "document_claims_supported_by_chunk_text", "status": "ok" if all_grounded else "failed"},
        ]

    if name == "missing_features_plus_documents":
        missing_reason = response["risk"]["reason"] or ",".join(response["limitations"])
        return [
            {
                "part": "risk_reports_missing_features",
                "status": "ok"
                if response["risk"]["status"] == "insufficient_data" and "features_required" in missing_reason
                else "failed",
            },
            {
                "part": "model_factors_missing_data_is_explicit",
                "status": "ok"
                if response["risk"]["top_features"] == [] and "features_required" in missing_reason
                else "failed",
            },
            {"part": "document_sources", "status": "ok" if response["sources"] else "failed"},
            {"part": "document_claims_supported_by_chunk_text", "status": "ok" if all_grounded else "failed"},
        ]

    return []


def run_scenario(agent, name: str, request: dict[str, Any], chunks: dict[str, str]) -> dict[str, Any]:
    response = agent.run_langgraph(request).model_dump()
    checks = coverage_checks(name, response, chunks)
    return {
        "name": name,
        "request": request,
        "response": response,
        "coverage_checks": checks,
        "grounding_checks": grounding_checks(response, chunks),
        "status": "ok"
        if checks
        and all(item["status"] == "ok" for item in checks)
        and response["validation_errors"] == []
        else "failed",
        "trace": trace_summary(response),
    }


def validation_guard_examples(agent, wallet_address: str) -> list[dict[str, Any]]:
    hit = RetrievalHit(
        chunk_id="guard_demo::chunk_001",
        document_id="guard_demo",
        score=1.0,
        wallet_address=wallet_address,
        source_kind="derived_data",
        text="wallet_address is linked. Counts and amounts are dataset fields only.",
        rank=1,
    )
    unsupported_finding = DocumentFinding(
        claim="This chunk proves confirmed fraud and legal guilt.",
        document_id=hit.document_id,
        chunk_id=hit.chunk_id,
        source_kind=hit.source_kind,
        supported=True,
        synthetic_demo=False,
    )
    unsupported_state = {
        "wallet_address": wallet_address,
        "sources": [
            SourceRef(
                document_id=hit.document_id,
                chunk_id=hit.chunk_id,
                source_kind=hit.source_kind,
                wallet_address=wallet_address,
                rank=1,
                score=1.0,
            )
        ],
        "document_hits": [hit],
        "document_findings": [unsupported_finding],
        "validation_errors": [],
        "data_status": "ok",
        "answer": "Structured answer.",
    }
    agent._validate_answer(unsupported_state)

    forbidden_state = {
        "sources": [],
        "document_hits": [],
        "document_findings": [],
        "validation_errors": [],
        "data_status": "ok",
        "risk": RiskBlock(status="not_requested"),
        "answer": "The document proves this is proven fraud.",
    }
    agent._validate_answer(forbidden_state)

    return [
        {
            "name": "existing_chunk_unsupported_claim",
            "status": "blocked"
            if unsupported_state["data_status"] == "partial"
            and any(error.startswith("unsupported_claim:") for error in unsupported_state["validation_errors"])
            else "failed",
            "validation_errors": unsupported_state["validation_errors"],
            "claim_supported_after_validation": unsupported_finding.supported,
        },
        {
            "name": "forbidden_answer_phrase",
            "status": "blocked"
            if forbidden_state["data_status"] == "partial"
            and "forbidden_answer_phrase:proven fraud" in forbidden_state["validation_errors"]
            else "failed",
            "validation_errors": forbidden_state["validation_errors"],
        },
    ]


def failure_checks(wallet_address: str, features: dict[str, Any]) -> list[dict[str, Any]]:
    retrieval_failure_agent = build_agent()
    retrieval_failure_agent.retriever = BrokenRetriever()
    retrieval_failure = retrieval_failure_agent.run_langgraph(
        {
            "question": f"Show document sources for wallet {wallet_address}",
            "wallet_address": wallet_address,
            "request_id": "agent_8a_retrieval_failure",
        }
    ).model_dump()

    llm_failure_agent = build_agent(model=MISSING_OLLAMA_MODEL, llm_timeout_seconds=1.0)
    llm_failure = llm_failure_agent.run_langgraph(
        {
            "question": f"Explain risk score and document sources for wallet {wallet_address}",
            "wallet_address": wallet_address,
            "features": features,
            "request_id": "agent_8a_llm_failure",
        }
    ).model_dump()

    return [
        {
            "name": "retrieval_error_partial_structured_response",
            "status": "ok"
            if retrieval_failure["data_status"] == "partial"
            and tool_status(retrieval_failure, "retrieval") == "error"
            and retrieval_failure["answer"]
            else "failed",
            "data_status": retrieval_failure["data_status"],
            "tool_statuses": retrieval_failure["tool_statuses"],
            "llm_status": retrieval_failure["llm"]["status"],
            "trace": trace_summary(retrieval_failure),
        },
        {
            "name": "llm_unavailable_returns_tool_output",
            "status": "ok"
            if llm_failure["llm"]["status"] == "unavailable"
            and llm_failure["risk"]["status"] == "scored"
            and llm_failure["sources"]
            else "failed",
            "data_status": llm_failure["data_status"],
            "risk_status": llm_failure["risk"]["status"],
            "source_count": len(llm_failure["sources"]),
            "tool_statuses": llm_failure["tool_statuses"],
            "llm_status": llm_failure["llm"]["status"],
            "trace": trace_summary(llm_failure),
        },
    ]


def write_report(result: dict[str, Any]):
    lines = [
        "# Agent composite answer audit",
        "",
        "Command 8a audit: composite questions, per-part coverage, chunk-level grounding, partial responses, and stopped bad answers.",
        "",
        f"- Ollama model: `{result['ollama_model']}`.",
        f"- Wallet address: `{result['wallet_address']}`.",
        f"- Scenario count: `{len(result['scenarios'])}`.",
        "",
        "## Composite Scenarios",
        "",
        "| Scenario | Result | Intent | Selection | Selected tools | Risk | Sources | Validation errors |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for scenario in result["scenarios"]:
        response = scenario["response"]
        lines.append(
            "| {name} | {status} | {intent} | {selection} | {tools} | {risk} | {sources} | {errors} |".format(
                name=scenario["name"],
                status=scenario["status"],
                intent=response["intent"],
                selection=response["tool_selection_source"],
                tools=", ".join(response["selected_tools"]) or "none",
                risk=response["risk"]["status"],
                sources=len(response["sources"]),
                errors=", ".join(response["validation_errors"]) or "none",
            )
        )

    lines.extend(["", "## Per-Part Coverage", ""])
    for scenario in result["scenarios"]:
        lines.append(f"### {scenario['name']}")
        for check in scenario["coverage_checks"]:
            lines.append(f"- `{check['part']}`: `{check['status']}`")
        lines.append(f"- trace: `{scenario['trace']}`")
        lines.append("")

    lines.extend(
        [
            "## Chunk Grounding",
            "",
            "| Scenario | Chunk | Exists | Supported by text | Source kind |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for scenario in result["scenarios"]:
        for check in scenario["grounding_checks"]:
            lines.append(
                f"| {scenario['name']} | {check['chunk_id']} | {check['chunk_exists']} | "
                f"{check['supported_by_text']} | {check['source_kind']} |"
            )

    lines.extend(
        [
            "",
            "## Failure And Fallback Checks",
            "",
            "| Check | Result | Detail |",
            "| --- | --- | --- |",
        ]
    )
    for check in result["failure_checks"]:
        detail = (
            f"data_status={check['data_status']}; llm={check['llm_status']}; "
            f"trace={check['trace']}"
        )
        lines.append(f"| {check['name']} | {check['status']} | {detail} |")

    lines.extend(
        [
            "",
            "## Stopped Bad Answers",
            "",
            "| Example | Result | Validation errors |",
            "| --- | --- | --- |",
        ]
    )
    for guard in result["validation_guard_examples"]:
        lines.append(
            f"| {guard['name']} | {guard['status']} | {', '.join(guard['validation_errors']) or 'none'} |"
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    chunks = chunk_text_by_id()
    wallet_address = first_wallet_address()
    from app import api

    features = feature_row(api.feature_names)
    agent = build_agent(num_predict=128)

    scenarios = [
        (
            "wallet_score_factors_documents_boundary",
            {
                "question": (
                    f"For wallet {wallet_address}, explain the risk score, top model factors, "
                    "document sources, and whether the chunks are synthetic demo or derived data."
                ),
                "wallet_address": wallet_address,
                "features": features,
                "top_k": 4,
                "request_id": "agent_8a_wallet_composite",
            },
        ),
        (
            "general_methodology_multi_fragment",
            {
                "question": (
                    "Summarize retrieval rules, scoring interpretation limits, and feature dictionary notes. "
                    "Cite document chunks."
                ),
                "top_k": 3,
                "request_id": "agent_8a_general_multi_fragment",
            },
        ),
        (
            "missing_features_plus_documents",
            {
                "question": (
                    f"For wallet {wallet_address}, give the risk score, top model factors, and document sources."
                ),
                "wallet_address": wallet_address,
                "top_k": 3,
                "request_id": "agent_8a_missing_features",
            },
        ),
    ]

    scenario_results = [run_scenario(agent, name, request, chunks) for name, request in scenarios]
    result = {
        "ollama_model": OLLAMA_MODEL,
        "wallet_address": wallet_address,
        "scenarios": scenario_results,
        "failure_checks": failure_checks(wallet_address, features),
        "validation_guard_examples": validation_guard_examples(agent, wallet_address),
    }
    result["status"] = "ok" if all(
        item["status"] == "ok"
        for item in result["scenarios"] + result["failure_checks"]
    ) and all(item["status"] == "blocked" for item in result["validation_guard_examples"]) else "failed"

    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
