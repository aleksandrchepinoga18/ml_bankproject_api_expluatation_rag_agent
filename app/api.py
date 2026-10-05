from datetime import datetime
import json
import os
import re
import sys
import uuid

from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.log_predictions import log_prediction
from src.inference import (
    build_model_input,
    get_rule_triggers,
    get_top_risk_factors,
    load_explainer_and_metadata,
    save_explanation,
)
from src.scoring_adapter import ScoringAdapter

app = Flask(__name__)
app.config.setdefault(
    "EXPLANATION_DIR",
    os.getenv("EXPLANATION_DIR", os.path.join("monitoring", "explanations")),
)

(
    explainer,
    feature_names,
    model,
    best_threshold,
    preprocessing,
    model_metadata,
) = load_explainer_and_metadata()
MODEL_VERSION = model_metadata["model_version"]
SCHEMA_VERSION = model_metadata["schema_version"]
scoring_adapter = ScoringAdapter(
    model=model,
    feature_names=feature_names,
    preprocessing=preprocessing,
    threshold=best_threshold,
    model_version=MODEL_VERSION,
    schema_version=SCHEMA_VERSION,
)


def _safe_identifier(value):
    safe = re.sub(r"[^\w-]", "_", str(value))
    return safe[:128] or f"prediction_{uuid.uuid4().hex}"


def _is_valid_identifier(value):
    return bool(re.match(r"^[\w-]{1,128}$", str(value)))


def _explanation_path(identifier):
    return os.path.join(app.config["EXPLANATION_DIR"], f"{_safe_identifier(identifier)}.json")


def _extract_identity(item, index, features=None):
    wallet_address = item.get("wallet_address")
    if wallet_address is None and isinstance(features, dict):
        wallet_address = features.get("wallet_address")

    raw_user_id = item.get("user_id")
    if raw_user_id is None and isinstance(features, dict):
        raw_user_id = features.get("user_id")

    if raw_user_id is None:
        raw_user_id = wallet_address or f"prediction_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{index}"

    row_id = item.get("row_id")
    if row_id is None and isinstance(features, dict):
        row_id = features.get("row_id") or features.get("source_row_index")
    if row_id is None:
        row_id = f"input_row_{index}"

    return _safe_identifier(raw_user_id), wallet_address, str(row_id)


def _normalize_prediction_payload(data):
    if not isinstance(data, list):
        data = [data]
    if not all(isinstance(item, dict) for item in data):
        raise ValueError("Each prediction item must be a JSON object")

    feature_rows = []
    identities = []
    for i, item in enumerate(data):
        if "features" in item:
            features = item["features"]
            if not isinstance(features, dict):
                raise ValueError("features must be a JSON object")
            feature_row = dict(features)
        else:
            feature_row = dict(item)

        user_id, wallet_address, row_id = _extract_identity(item, i, feature_row)
        feature_rows.append(feature_row)
        identities.append({"user_id": user_id, "wallet_address": wallet_address, "row_id": row_id})

    return feature_rows, identities


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _insufficient_data_response(identity, reason):
    return jsonify(
        {
            "user_id": identity["user_id"],
            "wallet_address": identity["wallet_address"],
            "row_id": identity["row_id"],
            "model_version": MODEL_VERSION,
            "schema_version": SCHEMA_VERSION,
            "risk": {
                "status": "insufficient_data",
                "reason": reason,
            },
        }
    )


def _find_latest_current_explanation_for_user(user_id):
    explanation_dir = app.config["EXPLANATION_DIR"]
    if not os.path.isdir(explanation_dir):
        return None, None

    matches = []
    for name in os.listdir(explanation_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(explanation_dir, name)
        try:
            exp = _load_json(path)
        except Exception:
            continue
        if exp.get("user_id") != user_id:
            continue
        if exp.get("model_version") != MODEL_VERSION:
            continue
        if exp.get("schema_version") != SCHEMA_VERSION:
            continue
        matches.append((exp.get("timestamp", ""), path, exp))

    if not matches:
        return None, None
    _, path, exp = sorted(matches, key=lambda item: item[0])[-1]
    return path, exp


def _load_explanation_for_request(data):
    request_id = data.get("request_id")
    user_id = data.get("user_id")

    if request_id:
        if not _is_valid_identifier(request_id):
            return None, None, (jsonify({"error": "Недопустимый request_id"}), 400)
        path = _explanation_path(request_id)
        if not os.path.exists(path):
            return None, None, (jsonify({"error": "Объяснение не найдено"}), 404)
        return path, _load_json(path), None

    if not user_id:
        return None, None, (jsonify({"error": "request_id или user_id обязателен"}), 400)
    if not _is_valid_identifier(user_id):
        return None, None, (jsonify({"error": "Недопустимый user_id"}), 400)

    user_id = _safe_identifier(user_id)
    path, exp = _find_latest_current_explanation_for_user(user_id)
    if exp is not None:
        return path, exp, None

    legacy_path = _explanation_path(user_id)
    if not os.path.exists(legacy_path):
        return None, None, (jsonify({"error": "Объяснение не найдено"}), 404)
    return legacy_path, _load_json(legacy_path), None


def _is_stale_explanation(exp):
    return (
        exp.get("schema_version") != SCHEMA_VERSION
        or exp.get("model_version") != MODEL_VERSION
    )


def _stale_response(exp):
    return jsonify(
        {
            "user_id": exp.get("user_id"),
            "wallet_address": exp.get("wallet_address"),
            "row_id": exp.get("row_id"),
            "request_id": exp.get("request_id"),
            "score": exp.get("score"),
            "threshold": exp.get("threshold"),
            "model_version": exp.get("model_version"),
            "schema_version": exp.get("schema_version"),
            "current_model_version": MODEL_VERSION,
            "current_schema_version": SCHEMA_VERSION,
            "risk": {
                "status": "stale",
                "reason": "explanation_version_mismatch",
            },
            "explanation": (
                "Объяснение устарело: версия модели или схема объяснения не совпадает "
                "с текущим контрактом."
            ),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "JSON body is required"}), 400

        feature_rows, identities = _normalize_prediction_payload(data)
        adapter_result = scoring_adapter.score_rows(
            feature_rows,
            wallet_address=identities[0]["wallet_address"] if len(identities) == 1 else None,
            row_ids=[identity["row_id"] for identity in identities],
        )
        if adapter_result.status == "insufficient_data":
            return _insufficient_data_response(identities[0], adapter_result.reason)

        X = build_model_input(feature_rows, feature_names, preprocessing)
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= best_threshold).astype(int)
        shap_values = explainer.shap_values(X)

        for i in range(len(X)):
            request_id = uuid.uuid4().hex
            log_prediction(
                features=X.iloc[i].to_dict(),
                score=float(proba[i]),
                model_usage=MODEL_VERSION,
            )
            item_shap_values = (
                [values[i:i + 1] for values in shap_values]
                if isinstance(shap_values, list)
                else shap_values[i:i + 1]
            )
            top_features = get_top_risk_factors(
                item_shap_values,
                feature_names,
                X.iloc[i],
            )
            rule_triggers = get_rule_triggers(X.iloc[i])
            save_explanation(
                user_id=identities[i]["user_id"],
                wallet_address=identities[i]["wallet_address"],
                request_id=request_id,
                row_id=identities[i]["row_id"],
                model_version=MODEL_VERSION,
                schema_version=SCHEMA_VERSION,
                score=float(proba[i]),
                top_features=top_features,
                best_threshold=best_threshold,
                rule_triggers=rule_triggers,
                feature_order=feature_names,
                output_dir=app.config["EXPLANATION_DIR"],
            )
            identities[i]["request_id"] = request_id

        result = [
            {
                "user_id": identity["user_id"],
                "wallet_address": identity["wallet_address"],
                "row_id": identity["row_id"],
                "request_id": identity["request_id"],
                "model_version": MODEL_VERSION,
                "schema_version": SCHEMA_VERSION,
                "prediction": int(p),
                "risk_probability": float(pr),
                "threshold": float(best_threshold),
            }
            for identity, p, pr in zip(identities, pred, proba)
        ]
        if len(result) > 1:
            return jsonify(
                {
                    "status": "scored",
                    "model_version": MODEL_VERSION,
                    "schema_version": SCHEMA_VERSION,
                    "rows": result,
                    "aggregation": (
                        None
                        if adapter_result.aggregation is None
                        else {
                            "rule": adapter_result.aggregation.rule,
                            "score": adapter_result.aggregation.score,
                            "prediction": adapter_result.aggregation.prediction,
                            "records_count": adapter_result.aggregation.records_count,
                            "representative_row_id": (
                                adapter_result.aggregation.representative_row_id
                            ),
                        }
                    ),
                }
            )
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/explain", methods=["POST"])
def explain():
    data = request.get_json(silent=True) or {}
    _, exp, error = _load_explanation_for_request(data)
    if error is not None:
        return error

    if _is_stale_explanation(exp):
        return _stale_response(exp)

    rule_reasons = []
    triggered_features = set()
    for trigger in exp.get("rule_triggers", []):
        message = trigger.get("message")
        if message:
            rule_reasons.append(message)
            feature = trigger.get("feature")
            if feature:
                triggered_features.add(feature)

    shap_reasons = []
    for feat in exp.get("top_features", []):
        name = feat.get("feature")
        val = feat.get("value")
        if val is None or name in triggered_features:
            continue
        if name == "risky_tx_count" and val > 20:
            shap_reasons.append(f"обнаружено {int(val)} рисковых транзакций")
        elif name == "wallet_age" and val < 604800:
            days = max(0, int(val // 86400))
            shap_reasons.append(f"кошелёк создан всего {days} дней назад")
        else:
            shap_reasons.append(f"признак {name} повысил оценку риска")

    if exp.get("risk_class") == "high_risk" and rule_reasons:
        text = "Оценка риска кошелька выше порога: " + ", ".join(rule_reasons) + "."
        if shap_reasons:
            text += " Дополнительно модель учитывает: " + ", ".join(shap_reasons) + "."
    elif exp.get("risk_class") == "high_risk" and shap_reasons:
        text = "Оценка риска кошелька выше порога: " + ", ".join(shap_reasons) + "."
    elif exp.get("risk_class") == "lower_risk":
        text = "Оценка риска кошелька ниже текущего порога модели."
    else:
        text = "Оценка основана на комплексной модели риска кошелька."

    return jsonify(
        {
            "user_id": exp.get("user_id"),
            "wallet_address": exp.get("wallet_address"),
            "row_id": exp.get("row_id"),
            "request_id": exp.get("request_id"),
            "model_version": exp.get("model_version"),
            "schema_version": exp.get("schema_version"),
            "decision": exp.get("decision"),
            "risk_class": exp.get("risk_class"),
            "risk": {
                "status": "current",
                "class": exp.get("risk_class"),
            },
            "score": exp.get("score"),
            "threshold": exp.get("threshold"),
            "model_class": exp.get("model_class"),
            "top_features": exp.get("top_features", []),
            "rule_triggers": exp.get("rule_triggers", []),
            "explanation": text,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
