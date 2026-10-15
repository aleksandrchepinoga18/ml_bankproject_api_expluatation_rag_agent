import json
import sys
from argparse import Namespace

import pytest

from scripts import demo_agent

from scripts.demo_agent import (
    final_answer_validation_errors,
    replacement_reasons,
    save_client_context,
    summarize_state,
)


def test_demo_summary_distinguishes_safe_fallback_from_final_validation_failure():
    analysis = {
        "risk": {"status": "scored"},
        "llm": {"status": "ok"},
        "limitations": ["llm_answer_replaced_for_scored_risk_contradiction", "llm_answer_fallback"],
        "validation_errors": ["llm_answer_replaced_for_scored_risk_contradiction"],
    }

    assert replacement_reasons(analysis) == ["llm_answer_replaced_for_scored_risk_contradiction"]
    assert final_answer_validation_errors(analysis) == []
    assert summarize_state(analysis) == "Original LLM text was rejected; safe deterministic fallback was returned."


def test_demo_summary_keeps_real_final_validation_errors_visible():
    analysis = {
        "risk": {"status": "scored"},
        "llm": {"status": "ok"},
        "limitations": ["llm_answer_fallback"],
        "validation_errors": ["forbidden_answer_phrase:credit declined"],
    }

    assert replacement_reasons(analysis) == []
    assert final_answer_validation_errors(analysis) == ["forbidden_answer_phrase:credit declined"]
    assert summarize_state(analysis) == "Final answer failed validation: review validation_errors and trace."


def test_demo_summary_uses_trace_to_identify_validated_fallback():
    analysis = {
        "risk": {"status": "scored"},
        "llm": {"status": "ok"},
        "limitations": ["final_answer_replaced_after_validation"],
        "validation_errors": ["forbidden_answer_phrase:credit declined"],
        "trace": [{"action": "fallback_replacement", "status": "ok"}],
    }

    assert replacement_reasons(analysis) == ["final_answer_replaced_after_validation"]
    assert final_answer_validation_errors(analysis) == []
    assert summarize_state(analysis) == "Original LLM text was rejected; safe deterministic fallback was returned."


def test_demo_summary_does_not_hide_structural_errors_after_fallback():
    analysis = {
        "risk": {"status": "not_requested"},
        "llm": {"status": "ok"},
        "limitations": ["final_answer_replaced_after_validation"],
        "validation_errors": ["missing_chunk:missing_chunk"],
        "trace": [{"action": "fallback_replacement", "status": "ok"}],
    }

    assert final_answer_validation_errors(analysis) == ["missing_chunk:missing_chunk"]
    assert summarize_state(analysis) == "Final answer failed validation: review validation_errors and trace."


def test_demo_client_context_preserves_demo_feature_warning(tmp_path):
    raw_path = tmp_path / "raw.json"
    raw_path.write_text("{}\n", encoding="utf-8")

    context_path = save_client_context(
        raw_path,
        features_file=tmp_path / "features.json",
        features_note="synthetic demo features; not production evidence",
    )

    assert context_path is not None
    context = json.loads(context_path.read_text(encoding="utf-8"))
    assert context["features_origin_warning"] == "synthetic demo features; not production evidence"


@pytest.mark.parametrize("details,trace", [(False, False), (True, False), (False, True), (True, True)])
def test_demo_display_modes_keep_raw_response_unchanged(tmp_path, monkeypatch, capsys, details, trace):
    body = {
        "answer": "Оценка модели — 0,359 при пороге 0,344. Класс повышенного риска.",
        "agent_version": "test_agent", "prompt_version": "test_prompt", "data_status": "partial",
        "risk": {
            "status": "scored", "score": 0.35894730197831315, "threshold": 0.34438552759270996,
            "prediction": 1, "model_version": "test_model", "schema_version": "test_schema",
            "top_features": [{"feature": "time_since_last_liquidated", "value": 0, "impact": 0.4586088125}],
        },
        "sources": [{"document_id": "doc", "chunk_id": "chunk", "source_kind": "derived_data"}],
        "llm": {"status": "ok", "model": "test_llm"},
        "limitations": ["llm_answer_replaced_for_scored_risk_contradiction"],
        "validation_errors": ["llm_answer_replaced_for_scored_risk_contradiction"],
        "trace": [{"step": 1, "node": "build_answer", "action": "node", "status": "ok", "detail": "trace_only"}],
    }
    features = tmp_path / "features.json"
    features.write_text(json.dumps({"features": {}, "note": "synthetic demo input"}), encoding="utf-8")
    raw = tmp_path / "raw.json"
    argv = ["demo_agent.py", "--question", "Объясни оценку", "--features-file", str(features), "--save-json", str(raw)]
    if details:
        argv.append("--details")
    if trace:
        argv.append("--trace")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(demo_agent, "request_agent", lambda *_args: (body, 125.0))

    assert demo_agent.main() == 0
    output = capsys.readouterr().out
    assert "Объясни оценку" in output
    assert body["answer"] in output
    assert "synthetic demo input" in output
    assert "125.0 ms" in output
    assert ("LightGBM Result" in output) is details
    assert ("raw output/margin LightGBM" in output) is details
    assert ("does not verify field definitions" in output) is details
    assert ("time_since_last_liquidated" in output) is details
    assert ("test_model" in output) is details
    assert ("test_llm" in output) is details
    assert ("doc/chunk" in output) is details
    assert ("llm_answer_replaced_for_scored_risk_contradiction" in output) is details
    assert ("trace_only" in output) is trace
    if not details:
        assert "Текст LLM заменён" in output
    assert json.loads(raw.read_text(encoding="utf-8")) == body
    assert json.loads(raw.with_suffix(".json.client_context.json").read_text(encoding="utf-8"))["features_origin_warning"] == "synthetic demo input"


def test_default_demo_does_not_invent_provenance(tmp_path, monkeypatch, capsys):
    features = tmp_path / "real.json"
    features.write_text('{"wallet_age": 100}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["demo_agent.py", "--question", "Оценка?", "--features-file", str(features)])
    monkeypatch.setattr(demo_agent, "request_agent", lambda *_args: ({"answer": "Краткий ответ."}, 5.0))

    assert demo_agent.main() == 0
    output = capsys.readouterr().out
    assert "Feature Input" not in output
    assert "synthetic" not in output
    assert "demo" not in output
    assert "LLM" not in output


def test_default_demo_surfaces_final_validation_failure(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["demo_agent.py", "--question", "Оценка?"])
    body = {"answer": "Ответ.", "validation_errors": ["missing_chunk:missing"]}
    monkeypatch.setattr(demo_agent, "request_agent", lambda *_args: (body, 5.0))
    demo_agent.main()
    assert "итоговый ответ не прошёл проверку" in capsys.readouterr().out


@pytest.mark.parametrize("endpoint,field", [("analyze", "question"), ("chat", "message")])
def test_declared_demo_origin_reaches_llm_through_existing_request_field(tmp_path, endpoint, field):
    path = tmp_path / "features.json"
    path.write_text(json.dumps({"note": "Synthetic/demo input, not real wallet evidence", "features": {"wallet_age": 10}}), encoding="utf-8")
    args = Namespace(endpoint=endpoint, question="Поясни результат", top_k=3, wallet_address=None, features_file=path)
    payload, note = demo_agent.build_payload(args)
    assert payload[field].startswith(args.question)
    assert note in payload[field]
    assert set(payload) == {field, "top_k", "features"}
    assert payload["features"] == {"wallet_age": 10}


def test_input_without_declared_provenance_does_not_get_demo_context(tmp_path):
    path = tmp_path / "features.json"
    path.write_text('{"features": {"metric": 0}}', encoding="utf-8")
    args = Namespace(endpoint="analyze", question="Поясни результат", top_k=3, wallet_address=None, features_file=path)
    payload, note = demo_agent.build_payload(args)
    assert note is None
    assert payload["question"] == args.question
