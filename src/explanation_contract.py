"""Internal explanation evidence; does not extend the public API schema."""

import math
from typing import Any


EXPLANATION_EVIDENCE_CONTRACT: dict[str, dict[str, Any]] = {
    "shap": {
        "source": "feature_evidence[].shap_contribution",
        "supports": "local contributions to the model output for the explained input",
        "complete_model_explanation": False,
        "establishes_wallet_history_or_causality": False,
    },
    "rules": {
        "source": "rule_evidence",
        "supports": "separate predicate checks of supplied input values",
        "establishes_score_contribution": False,
        "attribution_requirement": "A rule feature's model contribution needs its own SHAP evidence; triggering is not attribution",
    },
    "provenance": {
        "source": "client-declared input note, if supplied",
        "scope": "supplied example only",
        "wallet_origin": "not established",
        "inference_from_feature_values": "not established",
    },
    "semantics": {
        "numeric_zero": "present value, not missing data",
        "zero_establishes_absence_of_activity": False,
        "interpretation_requirement": "Verified field definitions and zero meaning are required to infer activity or events",
        "unknown_definition_or_unit": "remain unknown; names and values are not documentation",
    },
}


RULE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "risky_tx_count_gt_20": {
        "feature": "risky_tx_count", "predicate": "value > 20", "threshold": 20,
        "unit": "dataset_count", "unit_verified": True,
        "evidence": "computed_input_check_not_independent_wallet_evidence",
    },
    "wallet_age_lt_7_days": {
        "feature": "wallet_age", "predicate": "value < 604800", "threshold": 604800,
        "unit": None, "unit_verified": False,
        "legacy_assumption": "rule name assumed seconds; 604800 = 7 * 86400; not verified in source data",
        "evidence": "computed_input_check_not_independent_wallet_evidence",
    },
}


def numeric_value_status(value: Any, *, supplied: bool = True) -> str:
    if not supplied:
        return "not_supplied"
    if value is None:
        return "missing"
    try:
        return "present_numeric" if math.isfinite(float(value)) else "missing"
    except (TypeError, ValueError, OverflowError):
        return "non_numeric"


def feature_evidence(top_features: list[dict[str, Any]], input_row: dict[str, Any] | None) -> list[dict[str, Any]]:
    facts = []
    for item in top_features:
        name = str(item.get("feature"))
        value = item.get("value")
        status = numeric_value_status(value, supplied="value" in item)
        facts.append({
            "feature": name,
            "model_value": value, "model_value_status": status,
            "is_numeric_zero": value is not None and status == "present_numeric" and float(value) == 0,
            "input_value": None if input_row is None else input_row.get(name),
            "input_value_status": "not_available" if input_row is None else numeric_value_status(
                input_row.get(name), supplied=name in input_row,
            ),
            "shap_contribution": {"value": item.get("impact"), "unit": "raw output/margin LightGBM", "target_class": 1},
            "semantics": {"definition": None, "feature_unit": None, "zero_meaning": None, "status": "unverified"},
        })
    return facts
