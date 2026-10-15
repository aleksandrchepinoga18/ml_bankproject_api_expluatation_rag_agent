"""Acceptance of free explanations and rejection of bounded, explicit contradictions."""

import json

import pytest

from src.agent import AgentRequest, LLMStatus, RiskBlock, SourceRef, WalletRiskAgent


class EmptyRetriever:
    def search_bm25(self, *_args, **_kwargs):
        return []


class ExplanationProvider:
    def __init__(self, text):
        self.text = text
        self.prompts = []

    def generate(self, prompt, num_predict=None):
        self.prompts.append(prompt)
        return LLMStatus(provider="fake", model="explanation", status="ok", done_reason="stop"), self.text


def explanation_state(score, threshold, feature):
    return {
        "request": AgentRequest(question="Поясни вклад факторов", features={feature: 0}),
        "request_id": "free-explanation", "intent": "explain", "data_status": "ok",
        "risk": RiskBlock(
            status="scored", score=score, threshold=threshold, prediction=int(score >= threshold),
            top_features=[{"feature": feature, "value": 0, "impact": 0.17}],
        ),
        "sources": [], "document_findings": [], "document_hits": [],
        "limitations": [], "validation_errors": [], "trace": [],
    }


@pytest.mark.parametrize("score,threshold,feature,text", [
    (0.18, 0.4, "activity_count", "Вклад activity_count направлен в сторону риск-класса, но не описывает модель целиком."),
    (0.88, 0.63, "volatility_index", "### Пояснение\nУ volatility_index положительный SHAP-вклад в raw margin."),
    (0.29, 0.29, "deposit_gap", "Неизвестно, означает ли ноль отсутствие депозитов. Документации для такой интерпретации недостаточно."),
    (0.123456789, 0.2, "balance_delta", "Это не является доказательством мошенничества. Наблюдаем только вклад признака balance_delta."),
    (0.67, 0.51, "time_since_first_deposit", "Смысл нулевого значения не подтверждён; нельзя заключить, что депозитов не было."),
    (0.41, 0.6, "total_available_borrows_eth", "Слова demo и synthetic сами по себе не подтверждают происхождение входа."),
    (0.73, 0.55, "exposure_index", "Score is not based on documents. SHAP contributions are not percentages or probabilities."),
    (0.81, 0.62, "event_count", "Оценка не ниже порога. SHAP-вклад не выражается в процентах."),
    (0.22, 0.45, "reserve_ratio", "Score is not 0.9. This wallet is not proven fraud."),
    (0.19, 0.39, "flow_index", "Для положительных SHAP-вкладов объясняется class 1, а не предсказанный класс всей строки."),
])
def test_free_explanation_is_accepted_without_required_wording(score, threshold, feature, text):
    provider = ExplanationProvider(text)
    agent = WalletRiskAgent(retriever=EmptyRetriever(), llm_provider=provider)
    state = explanation_state(score, threshold, feature)
    agent._build_answer(state)
    agent._validate_answer(state)

    assert provider.prompts  # LLM generation remains part of the path.
    assert state["validation_errors"] == []
    assert "llm_answer_fallback" not in state["limitations"]
    assert state["data_status"] == "ok"
    assert state["answer"].startswith(agent._scored_result_text(state["risk"]))
    assert state["answer"].count("Результат классификации") == 1
    assert state["answer"].count("не калиброванная вероятность") == 1
    assert not state["answer"].endswith("LLM explanation:")
    assert "synthetic_demo — демонстрационные материалы" not in state["answer"]
    # Exact input floats are rendered by code, independent of the explanation's omissions.
    assert str(score).replace(".", ",") in state["answer"]
    context = json.loads(provider.prompts[0].split("\n\n", 1)[1])
    assert context["feature_semantics"]["fields"] == [feature]
    assert "Not established" in context["feature_semantics"]["units_and_zero_meaning"]


@pytest.mark.parametrize("score,threshold,feature,text,reason", [
    (0.81, 0.62, "activity_count", "Score=0.19.", "scored_risk_contradiction"),
    (0.16, 0.45, "reserve_ratio", "Threshold=0.12.", "scored_risk_contradiction"),
    (0.26, 0.48, "volatility_index", "Prediction=1.", "scored_risk_contradiction"),
    (0.83, 0.57, "deposit_gap", "Класс пониженного риска.", "scored_risk_contradiction"),
    (0.86, 0.66, "liquidity_index", "The predicted class is low risk.", "scored_risk_contradiction"),
    (0.17, 0.33, "utilization_index", "Модель отнесла пример к классу повышенного риска.", "scored_risk_contradiction"),
    (0.91, 0.76, "balance_delta", "The score is below threshold.", "scored_risk_contradiction"),
    (0.12, 0.31, "flow_index", "Score is not 0.12.", "scored_risk_contradiction"),
    (0.82, 0.61, "exposure_index", "Score is calculated from retrieved documents.", "scored_risk_contradiction"),
    (0.34, 0.52, "event_count", "According to retrieved documents, the answer is supported.", "source_consistency"),
    (0.76, 0.56, "reserve_ratio", "SHAP contribution raises model risk by 17%.", "shap_unit_misinterpretation"),
    (0.46, 0.64, "flow_index", "SHAP contribution is a probability.", "shap_unit_misinterpretation"),
    (0.74, 0.59, "activity_count", "SHAP contribution is not 40%; but impact raises model risk by 30%.", "shap_unit_misinterpretation"),
])
def test_explicit_contradictions_still_use_safe_fallback(score, threshold, feature, text, reason):
    agent = WalletRiskAgent(retriever=EmptyRetriever(), llm_provider=ExplanationProvider(text))
    state = explanation_state(score, threshold, feature)
    agent._build_answer(state)
    agent._validate_answer(state)

    assert "llm_answer_fallback" in state["limitations"]
    assert f"llm_answer_replaced_for_{reason}" in state["validation_errors"]
    assert text not in state["answer"]
    assert state["answer"].startswith(agent._scored_result_text(state["risk"]))


def test_code_owned_summary_and_limitations_are_not_repeated():
    state = explanation_state(0.68, 0.42, "deposit_count")
    agent = WalletRiskAgent(retriever=EmptyRetriever(), llm_provider=ExplanationProvider("unused"))
    explanation = (
        agent._scored_result_text(state["risk"]) + "\n"
        "У deposit_count есть положительный вклад.\n"
        "Оценка не является калиброванной вероятностью мошенничества."
    )
    agent.llm_provider = ExplanationProvider(explanation)
    agent._build_answer(state)
    agent._validate_answer(state)
    assert state["validation_errors"] == []
    assert state["answer"].count("Оценка модели") == 1
    assert state["answer"].count("Результат классификации") == 1
    assert state["answer"].count("вероятност") == 1
    assert "У deposit_count есть положительный вклад." in state["answer"]


def test_demo_source_warning_is_added_from_explicit_source_metadata():
    state = explanation_state(0.32, 0.28, "liquidity_index")
    state["sources"] = [SourceRef(
        document_id="example", chunk_id="example::chunk_1", source_kind="synthetic_demo",
        wallet_address=None, rank=1, score=0.8,
    )]
    agent = WalletRiskAgent(retriever=EmptyRetriever(), llm_provider=ExplanationProvider("Один вклад не объясняет модель целиком."))
    agent._build_answer(state)
    agent._validate_answer(state)
    assert state["validation_errors"] == []
    assert state["answer"].count("synthetic_demo — демонстрационные материалы") == 1


def test_rendering_cannot_hide_a_real_violation_in_a_disclaimer():
    state = explanation_state(0.72, 0.43, "activity_index")
    agent = WalletRiskAgent(
        retriever=EmptyRetriever(),
        llm_provider=ExplanationProvider("Это не калиброванная вероятность, но credit approved."),
    )
    agent._build_answer(state)
    agent._validate_answer(state)
    assert "forbidden_answer_phrase:credit approved" in state["validation_errors"]
    assert "llm_answer_fallback" in state["limitations"] or "final_answer_replaced_after_validation" in state["limitations"]
    assert "credit approved" not in state["answer"]


@pytest.mark.parametrize("feature,value,text", [
    ("deposit_interval", 0, "deposit_interval — не было значений, влияние составило 0,17."),
    ("record_count", 0.0, "record_count: значение отсутствует."),
    ("utilization_ratio", 7.5, "utilization_ratio — данных нет."),
    ("cash_delta", -3, "cash_delta: no value was provided."),
])
def test_present_numeric_feature_cannot_be_described_as_missing(feature, value, text):
    state = explanation_state(0.69, 0.44, feature)
    state["request"].features = {feature: value}
    state["risk"].top_features[0]["value"] = value
    agent = WalletRiskAgent(retriever=EmptyRetriever(), llm_provider=ExplanationProvider(text))
    detail = agent._scored_boundary_detail(state, text.lower())
    assert detail["rule"] == "feature_value_presence"
    assert detail["expected"] == {"status": "present_numeric", "value": value}
    agent._build_answer(state)
    assert "llm_answer_replaced_for_scored_risk_contradiction" in state["validation_errors"]


@pytest.mark.parametrize("text", [
    "deposit_interval равен нулю; его предметный смысл не установлен.",
    "deposit_interval: неизвестно определение, а не значение признака.",
    "Нельзя утверждать, что для deposit_interval не было значений.",
    "deposit_interval: отсутствует определение единиц, числовое значение передано.",
    "Риск не средний: используется бинарный результат модели.",
    "Средний риск как отдельный класс здесь не предусмотрен.",
    "Нельзя назвать риск средним без отдельной шкалы.",
    "Модель не выделяет средний риск.",
    "Класса «средний риск» нет: есть только бинарный результат.",
])
def test_unknown_meaning_and_negated_unsupported_category_are_accepted(text):
    state = explanation_state(0.79, 0.61, "deposit_interval")
    agent = WalletRiskAgent(retriever=EmptyRetriever(), llm_provider=ExplanationProvider(text))
    agent._build_answer(state)
    agent._validate_answer(state)
    assert state["validation_errors"] == []
    assert "llm_answer_fallback" not in state["limitations"]


@pytest.mark.parametrize("score,threshold,text", [
    (0.358947, 0.344386, "Это означает, что модель считает уровень риска средним."),
    (0.81, 0.55, "Риск умеренный."),
    (0.13, 0.4, "The model assigns medium risk."),
    (0.51, 0.51, "Risk level is moderate."),
    (0.77, 0.58, "The predicted class is medium."),
])
def test_binary_model_does_not_gain_an_unconfigured_middle_category(score, threshold, text):
    state = explanation_state(score, threshold, "arbitrary_metric")
    agent = WalletRiskAgent(retriever=EmptyRetriever(), llm_provider=ExplanationProvider(text))
    detail = agent._scored_boundary_detail(state, text.lower())
    assert detail["rule"] == "unsupported_risk_scale"
    assert detail["expected"]["classes"] == [0, 1]
    agent._build_answer(state)
    assert "llm_answer_fallback" in state["limitations"]


def test_positive_contribution_target_is_not_a_statement_of_predicted_class():
    state = explanation_state(0.2, 0.6, "flow_index")
    text = "SHAP-вклад направлен в сторону высокого риска, но не задаёт класс всей строки."
    agent = WalletRiskAgent(retriever=EmptyRetriever(), llm_provider=ExplanationProvider(text))
    assert agent._answer_replacement_reason(state, text) is None


def test_context_separates_original_missing_input_from_present_model_zero():
    state = explanation_state(0.64, 0.42, "imputed_metric")
    state["request"].features = {"imputed_metric": None}
    provider = ExplanationProvider("imputed_metric: исходное значение отсутствует, после предобработки в модели ноль.")
    agent = WalletRiskAgent(retriever=EmptyRetriever(), llm_provider=provider)
    agent._build_answer(state)
    assert "llm_answer_fallback" not in state["limitations"]
    payload = json.loads(provider.prompts[0].split("\n\n", 1)[1])
    fact = payload["feature_evidence"][0]
    assert fact["input_value_status"] == "missing"
    assert fact["model_value"] == 0
    assert fact["model_value_status"] == "present_numeric"
    assert fact["is_numeric_zero"] is True
    assert fact["shap_contribution"]["value"] == 0.17
    assert fact["semantics"]["zero_meaning"] is None
    assert payload["classification_contract"]["classes"] == {"0": "lower_risk", "1": "high_risk"}
    assert payload["classification_contract"]["additional_risk_levels"] is None
    assert payload["input_evidence"]["independently_verified_wallet_evidence"] is False


@pytest.mark.parametrize("feature,value,impact,rule_feature,rule_value,note", [
    ("event_gap", 0, 0.31, "alert_count", 42, "Client-declared synthetic input example."),
    ("reserve_delta", -2.5, -0.12, "exposure_count", 8, ""),
    ("flow_metric", 7.25, 0.09, "flow_metric", 7.25, "Client-declared demonstration row."),
])
def test_prompt_keeps_attribution_rules_provenance_and_semantics_separate(
    feature, value, impact, rule_feature, rule_value, note,
):
    state = explanation_state(0.68, 0.47, feature)
    state["request"].question += " " + note
    state["request"].features = {feature: value, rule_feature: rule_value}
    state["risk"].top_features = [{"feature": feature, "value": value, "impact": impact}]
    state["risk"].rule_triggers = [{
        "rule": "separate_input_check", "feature": rule_feature, "value": rule_value,
        "message": "A separate input predicate matched.",
    }]
    agent = WalletRiskAgent(retriever=EmptyRetriever(), llm_provider=ExplanationProvider(""))
    payload = json.loads(agent._prompt(state).split("\n\n", 1)[1])

    fact = payload["feature_evidence"][0]
    assert fact["model_value"] == value
    assert fact["input_value"] == value
    assert fact["model_value_status"] == "present_numeric"
    assert fact["is_numeric_zero"] is (value == 0)
    assert fact["shap_contribution"]["value"] == impact
    assert fact["semantics"]["definition"] is None
    assert fact["semantics"]["zero_meaning"] is None
    rule = payload["rule_evidence"][0]
    assert rule["feature"] == rule_feature
    assert rule["value"] == rule_value
    assert rule["contract"]["evidence"] == "unverified_rule_definition"
    assert "shap_contribution" not in rule

    contract = payload["evidence_contract"]
    assert contract["shap"]["source"] == "feature_evidence[].shap_contribution"
    assert contract["shap"]["complete_model_explanation"] is False
    assert contract["shap"]["establishes_wallet_history_or_causality"] is False
    assert contract["rules"]["establishes_score_contribution"] is False
    assert contract["semantics"]["zero_establishes_absence_of_activity"] is False
    assert contract["provenance"]["scope"] == "supplied example only"
    assert contract["provenance"]["wallet_origin"] == "not established"
    assert contract["provenance"]["inference_from_feature_values"] == "not established"
    assert payload["question"] == state["request"].question
    assert payload["input_evidence"]["origin"] == "caller_supplied_features"
    assert payload["input_evidence"]["independently_verified_wallet_evidence"] is False


def test_rule_context_is_computed_input_evidence_not_a_wallet_history_claim():
    from src.inference import get_rule_triggers

    state = explanation_state(0.72, 0.5, "activity_count")
    state["request"].features = {"risky_tx_count": 30, "wallet_age": 10}
    state["risk"].rule_triggers = get_rule_triggers(state["request"].features)
    text = "Во входной строке risky_tx_count=30; правило возраста проверяет число, а единицы возраста не подтверждены."
    provider = ExplanationProvider(text)
    agent = WalletRiskAgent(retriever=EmptyRetriever(), llm_provider=provider)
    agent._build_answer(state)
    assert "llm_answer_fallback" not in state["limitations"]
    payload = json.loads(provider.prompts[0].split("\n\n", 1)[1])
    age = next(item for item in payload["rule_evidence"] if item["feature"] == "wallet_age")
    assert age["value"] == 10
    assert age["contract"]["unit_verified"] is False
    assert age["contract"]["evidence"] == "computed_input_check_not_independent_wallet_evidence"
    assert agent._scored_boundary_detail(state, "кошелек создан менее чем за неделю.")["rule"] == "unverified_feature_unit"
    assert agent._scored_boundary_detail(state, "не подтверждено, что кошелёк создан менее недели назад.") is None
