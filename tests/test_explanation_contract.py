import json
from pathlib import Path

import pytest

from src.explanation_contract import RULE_DEFINITIONS, feature_evidence
from src.inference import get_rule_triggers


@pytest.mark.parametrize("value,status,is_zero", [
    (0, "present_numeric", True), (0.0, "present_numeric", True),
    (7.5, "present_numeric", False), (-4, "present_numeric", False),
    (None, "missing", False), (float("nan"), "missing", False),
])
def test_model_value_presence_is_independent_of_semantic_knowledge(value, status, is_zero):
    fact = feature_evidence([{"feature": "arbitrary_metric", "value": value, "impact": 0.23}], {"arbitrary_metric": value})[0]
    assert fact["model_value_status"] == status
    assert fact["is_numeric_zero"] is is_zero
    assert fact["shap_contribution"] == {"value": 0.23, "unit": "raw output/margin LightGBM", "target_class": 1}
    assert fact["semantics"]["feature_unit"] is None
    assert fact["semantics"]["zero_meaning"] is None


@pytest.mark.parametrize("row,input_status", [({}, "not_supplied"), ({"metric": None}, "missing"), ({"metric": 0}, "present_numeric")])
def test_imputed_model_zero_does_not_rewrite_original_input_availability(row, input_status):
    fact = feature_evidence([{"feature": "metric", "value": 0, "impact": 0.1}], row)[0]
    assert fact["input_value_status"] == input_status
    assert fact["model_value_status"] == "present_numeric"


def test_demo_rule_messages_describe_input_checks_not_verified_wallet_history():
    path = Path(__file__).resolve().parents[1] / "docs/demo_features_valid.json"
    demo = json.loads(path.read_text(encoding="utf-8"))
    assert "Synthetic/demo" in demo["note"]
    assert demo["features"]["risky_tx_count"] == 30
    assert demo["features"]["wallet_age"] == 10
    rules = {item["rule"]: item for item in get_rule_triggers(demo["features"])}
    count = rules["risky_tx_count_gt_20"]
    age = rules["wallet_age_lt_7_days"]
    assert count["value"] == 30
    assert "risky_tx_count=30" in count["message"]
    assert "во входных данных" in count["message"]
    assert age["value"] == 10
    assert "wallet_age=10" in age["message"]
    assert "единицы возраста не подтверждены" in age["message"]
    assert "кошелёк создан" not in age["message"]
    assert RULE_DEFINITIONS[age["rule"]]["unit_verified"] is False
    # Public rule shape is unchanged; explanation metadata stays internal.
    assert set(age) == {"rule", "feature", "value", "message"}


def test_rule_predicates_keep_their_existing_numeric_boundaries():
    assert get_rule_triggers({"risky_tx_count": 20, "wallet_age": 604800}) == []
    rules = get_rule_triggers({"risky_tx_count": 21, "wallet_age": 604799})
    assert [item["value"] for item in rules] == [21, 604799]
