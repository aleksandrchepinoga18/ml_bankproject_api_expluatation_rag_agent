import json

from monitoring import observability
from monitoring.observability import collect_metrics
from monitoring.observability import source_correctness
from src.agent import AgentResponse, DocumentFinding, LLMStatus, RiskBlock, SourceRef


def _response(**overrides):
    payload = {
        "request_id": "obs-test",
        "intent": "documents",
        "wallet_address": "0x1111111111111111111111111111111111111111",
        "data_status": "ok",
        "risk": RiskBlock(status="not_requested"),
        "answer": "Structured answer.",
        "llm": LLMStatus(provider="disabled", model="disabled", status="unavailable"),
    }
    payload.update(overrides)
    return AgentResponse(**payload)


def test_source_correctness_counts_supported_grounded_document_answer():
    response = _response(
        sources=[
            SourceRef(
                document_id="doc",
                chunk_id="doc::chunk_001",
                source_kind="derived_data",
                wallet_address="0x1111111111111111111111111111111111111111",
                rank=1,
                score=1.0,
            )
        ],
        document_findings=[
            DocumentFinding(
                claim="Supported claim.",
                document_id="doc",
                chunk_id="doc::chunk_001",
                source_kind="derived_data",
                supported=True,
                synthetic_demo=False,
            )
        ],
    )

    result = source_correctness(response)

    assert result["eligible"] is True
    assert result["correct"] is True


def test_source_correctness_rejects_unsupported_finding():
    response = _response(
        document_findings=[
            DocumentFinding(
                claim="Unsupported claim.",
                document_id="doc",
                chunk_id="doc::chunk_001",
                source_kind="derived_data",
                supported=False,
                synthetic_demo=False,
            )
        ]
    )

    result = source_correctness(response)

    assert result["eligible"] is True
    assert result["correct"] is False
    assert result["unsupported_findings"] == ["doc::chunk_001"]


def test_collect_metrics_counts_replaced_llm_answers(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setattr(observability, "LOG_DIR", log_dir)

    rows = [
        {
            "event": "agent_response",
            "limitations": ["llm_answer_replaced_for_insufficient_data"],
            "source_correctness": {"eligible": False, "correct": False},
        },
        {
            "event": "agent_response",
            "limitations": ["llm_answer_replaced_for_not_requested_risk"],
            "source_correctness": {"eligible": True, "correct": True},
        },
        {
            "event": "agent_response",
            "limitations": [],
            "source_correctness": {"eligible": True, "correct": True},
        },
    ]
    path = log_dir / "agent_requests_2026-10-05.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    metrics = collect_metrics()

    assert metrics["agent"]["total"] == 3
    assert metrics["agent"]["answer_replacement_count"] == 2
    assert metrics["agent"]["answer_replacement_rate"] == 0.666667
    assert "llm_answer_replaced_*" in metrics["agent"]["answer_replacement_definition"]
