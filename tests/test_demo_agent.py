import json

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
