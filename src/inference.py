import hashlib
import json
import os
import tempfile
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import shap

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

SCORING_SCHEMA_VERSION = "scoring_explanation_v1"


def load_model(path="models/lightgbm_model.pkl"):
    return joblib.load(path)


def predict(model, X):
    return model.predict(X), model.predict_proba(X)[:, 1]


def load_preprocessing(path="models/lightgbm_preprocessing.pkl"):
    if os.path.exists(path):
        return joblib.load(path)
    if os.path.exists("models/lightgbm_feature_names.pkl"):
        return {
            "feature_names": list(joblib.load("models/lightgbm_feature_names.pkl")),
            "imputation_values": {},
            "preprocessing": "legacy feature order only",
        }
    return {"feature_names": None, "imputation_values": {}, "preprocessing": "model fallback"}


def _sha256_file(path):
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def build_model_metadata(
    feature_names,
    preprocessing,
    best_threshold,
    model_path="models/lightgbm_model.pkl",
    schema_version=SCORING_SCHEMA_VERSION,
):
    version_payload = {
        "model_artifact_sha256": _sha256_file(model_path),
        "feature_names": list(feature_names),
        "preprocessing": preprocessing.get("preprocessing"),
        "imputation_values": preprocessing.get("imputation_values", {}),
        "threshold": float(best_threshold),
        "schema_version": schema_version,
    }
    canonical = json.dumps(
        version_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    model_version = "lightgbm_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return {
        "model_version": model_version,
        "schema_version": schema_version,
        "version_payload": version_payload,
    }


def load_explainer_and_metadata():
    model = load_model()
    explainer = shap.TreeExplainer(model)
    preprocessing = load_preprocessing()
    feature_names = preprocessing.get("feature_names") or model.feature_name_
    best_threshold = joblib.load("models/lightgbm_best_threshold.pkl")
    model_metadata = build_model_metadata(feature_names, preprocessing, best_threshold)
    return explainer, list(feature_names), model, best_threshold, preprocessing, model_metadata


def build_model_input(feature_rows, feature_names, preprocessing):
    X = pd.DataFrame(feature_rows)
    X = X.replace([np.inf, -np.inf], np.nan)
    for feature in feature_names:
        if feature not in X.columns:
            X[feature] = np.nan
    X = X[feature_names].apply(pd.to_numeric, errors="coerce")
    for feature, value in preprocessing.get("imputation_values", {}).items():
        if feature in X.columns:
            X[feature] = X[feature].fillna(value)
    return X


def extract_positive_class_shap_values(shap_values):
    if isinstance(shap_values, list):
        class_1_values = shap_values[1].values if hasattr(shap_values[1], "values") else shap_values[1]
        return np.asarray(class_1_values)[0]

    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    values = np.asarray(values)
    if values.ndim == 3:
        return values[0, :, 1]
    if values.ndim == 2:
        return values[0]
    return values


def _feature_values(features):
    if isinstance(features, pd.DataFrame):
        return features.iloc[0].tolist()
    if isinstance(features, pd.Series):
        return features.tolist()

    values = np.asarray(features)
    if values.ndim == 2:
        values = values[0]
    return values.tolist()


def get_top_risk_factors(shap_values, feature_names, features, top_k=2):
    class_1_values = extract_positive_class_shap_values(shap_values)
    values = _feature_values(features)
    impacts = zip(feature_names, values, class_1_values)
    risk_factors = sorted(
        [item for item in impacts if item[2] > 0],
        key=lambda item: item[2],
        reverse=True,
    )
    return [
        {"feature": feature, "value": value, "impact": float(impact)}
        for feature, value, impact in risk_factors[:top_k]
    ]


def get_rule_triggers(features):
    if isinstance(features, pd.DataFrame):
        values = features.iloc[0].to_dict()
    elif isinstance(features, pd.Series):
        values = features.to_dict()
    else:
        values = dict(features)

    triggers = []
    risky_tx_count = values.get("risky_tx_count")
    if risky_tx_count is not None and risky_tx_count > 20:
        triggers.append(
            {
                "rule": "risky_tx_count_gt_20",
                "feature": "risky_tx_count",
                "value": float(risky_tx_count),
                "message": f"обнаружено {int(risky_tx_count)} рисковых транзакций",
            }
        )

    wallet_age = values.get("wallet_age")
    if wallet_age is not None and wallet_age < 604800:
        days = max(0, int(wallet_age // 86400))
        triggers.append(
            {
                "rule": "wallet_age_lt_7_days",
                "feature": "wallet_age",
                "value": float(wallet_age),
                "message": f"кошелёк создан менее недели назад ({days} дней)",
            }
        )

    return triggers


def save_explanation(
    user_id,
    score,
    top_features,
    best_threshold,
    rule_triggers=None,
    wallet_address=None,
    request_id=None,
    row_id=None,
    model_version=None,
    schema_version=SCORING_SCHEMA_VERSION,
    model_class="risk_class_1",
    feature_order=None,
    output_dir="monitoring/explanations",
):
    if request_id is None:
        request_id = f"prediction_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

    risk_class = "high_risk" if score >= best_threshold else "lower_risk"
    decision = "повышенный_риск" if risk_class == "high_risk" else "низкий_риск"
    explanation = {
        "user_id": user_id,
        "wallet_address": wallet_address,
        "request_id": request_id,
        "row_id": row_id,
        "schema_version": schema_version,
        "model_version": model_version,
        "score": float(score),
        "threshold": float(best_threshold),
        "decision": decision,
        "risk_class": risk_class,
        "model_class": model_class,
        "feature_order": feature_order,
        "top_features": top_features,
        "rule_triggers": rule_triggers or [],
        "timestamp": datetime.utcnow().isoformat(),
    }
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, f"{request_id}.json")
    fd, tmp_path = tempfile.mkstemp(prefix=f".{request_id}.", suffix=".tmp", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(explanation, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
