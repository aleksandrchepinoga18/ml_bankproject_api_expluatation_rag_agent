import json
from pathlib import Path

from app import api


def _features():
    return {feature: 0 for feature in api.feature_names}


def _post_predict(client, user_id="contract_user", row_id="row-1"):
    payload = {
        "user_id": user_id,
        "wallet_address": "0xcontractwallet0001",
        "row_id": row_id,
        "features": _features(),
    }
    return client.post("/predict", json=payload)


def _read_explanation(explanation_dir, request_id):
    with (Path(explanation_dir) / f"{request_id}.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def test_predict_returns_versioned_contract_and_does_not_overwrite_user_id(tmp_path, monkeypatch):
    api.app.config["EXPLANATION_DIR"] = str(tmp_path)
    monkeypatch.setattr(api, "log_prediction", lambda **_: None)

    client = api.app.test_client()
    first = _post_predict(client, user_id="same_user", row_id="row-a")
    second = _post_predict(client, user_id="same_user", row_id="row-b")

    assert first.status_code == 200
    assert second.status_code == 200

    first_body = first.get_json()[0]
    second_body = second.get_json()[0]
    assert first_body["request_id"] != second_body["request_id"]
    assert first_body["model_version"] == api.MODEL_VERSION
    assert first_body["schema_version"] == api.SCHEMA_VERSION
    assert first_body["row_id"] == "row-a"
    assert second_body["row_id"] == "row-b"

    assert not (tmp_path / "same_user.json").exists()
    assert (tmp_path / f"{first_body['request_id']}.json").exists()
    assert (tmp_path / f"{second_body['request_id']}.json").exists()


def test_explain_uses_current_request_id_and_same_saved_shap_row(tmp_path, monkeypatch):
    api.app.config["EXPLANATION_DIR"] = str(tmp_path)
    monkeypatch.setattr(api, "log_prediction", lambda **_: None)

    client = api.app.test_client()
    predict_response = _post_predict(client, user_id="explain_user", row_id="source-row-42")
    predict_body = predict_response.get_json()[0]
    saved = _read_explanation(tmp_path, predict_body["request_id"])

    explain_response = client.post("/explain", json={"request_id": predict_body["request_id"]})
    explain_body = explain_response.get_json()

    assert explain_response.status_code == 200
    assert explain_body["risk"]["status"] == "current"
    assert explain_body["request_id"] == predict_body["request_id"]
    assert explain_body["row_id"] == "source-row-42"
    assert explain_body["model_version"] == api.MODEL_VERSION
    assert explain_body["top_features"] == saved["top_features"]
    assert "заявк" not in explain_body["explanation"].lower()
    assert "одобр" not in explain_body["explanation"].lower()


def test_legacy_user_id_explanation_is_stale(tmp_path):
    api.app.config["EXPLANATION_DIR"] = str(tmp_path)
    legacy = {
        "user_id": "legacy_user",
        "score": 0.9,
        "decision": "отказ",
        "top_features": [{"feature": "wallet_age", "value": 1, "impact": 0.5}],
    }
    (tmp_path / "legacy_user.json").write_text(
        json.dumps(legacy, ensure_ascii=False),
        encoding="utf-8",
    )

    response = api.app.test_client().post("/explain", json={"user_id": "legacy_user"})
    body = response.get_json()

    assert response.status_code == 200
    assert body["risk"]["status"] == "stale"
    assert body["current_model_version"] == api.MODEL_VERSION


def test_mismatched_model_version_explanation_is_stale(tmp_path):
    api.app.config["EXPLANATION_DIR"] = str(tmp_path)
    request_id = "mismatched_version"
    explanation = {
        "user_id": "version_user",
        "wallet_address": "0xcontractwallet0002",
        "request_id": request_id,
        "row_id": "row-version",
        "schema_version": api.SCHEMA_VERSION,
        "model_version": "lightgbm_old",
        "score": 0.1,
        "threshold": 0.2,
        "decision": "низкий_риск",
        "risk_class": "lower_risk",
        "top_features": [],
        "rule_triggers": [],
    }
    (tmp_path / f"{request_id}.json").write_text(
        json.dumps(explanation, ensure_ascii=False),
        encoding="utf-8",
    )

    response = api.app.test_client().post("/explain", json={"request_id": request_id})
    body = response.get_json()

    assert response.status_code == 200
    assert body["risk"]["status"] == "stale"
    assert body["model_version"] == "lightgbm_old"
    assert body["current_model_version"] == api.MODEL_VERSION


def test_predict_address_without_features_returns_insufficient_data(tmp_path, monkeypatch):
    api.app.config["EXPLANATION_DIR"] = str(tmp_path)
    monkeypatch.setattr(api, "log_prediction", lambda **_: None)

    response = api.app.test_client().post(
        "/predict",
        json={"user_id": "address_only", "wallet_address": "0xonlyaddress"},
    )
    body = response.get_json()

    assert response.status_code == 200
    assert body["wallet_address"] == "0xonlyaddress"
    assert body["risk"]["status"] == "insufficient_data"
    assert body["risk"]["reason"] == "no_model_features_provided"
    assert not list(tmp_path.glob("*.json"))


def test_predict_multiple_rows_returns_aggregation(tmp_path, monkeypatch):
    api.app.config["EXPLANATION_DIR"] = str(tmp_path)
    monkeypatch.setattr(api, "log_prediction", lambda **_: None)

    first = _features()
    second = _features()
    second["risky_tx_count"] = 100

    response = api.app.test_client().post(
        "/predict",
        json=[
            {
                "user_id": "multi_user",
                "wallet_address": "0xmulti",
                "row_id": "row-a",
                "features": first,
            },
            {
                "user_id": "multi_user",
                "wallet_address": "0xmulti",
                "row_id": "row-b",
                "features": second,
            },
        ],
    )
    body = response.get_json()
    row_scores = {row["row_id"]: row["risk_probability"] for row in body["rows"]}
    representative_id = max(row_scores, key=row_scores.get)

    assert response.status_code == 200
    assert body["status"] == "scored"
    assert body["aggregation"]["rule"] == "max_row_probability"
    assert body["aggregation"]["records_count"] == 2
    assert body["aggregation"]["representative_row_id"] == representative_id
