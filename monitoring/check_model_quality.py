import json
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

best_threshold = joblib.load("models/lightgbm_best_threshold.pkl")


def _has_trusted_labels(df):
    if "label_source" not in df.columns:
        print("Label source is missing; refusing to treat labels as verified quality evidence")
        return False
    trusted = df["label_source"].astype(str).str.lower().isin({"observed", "verified", "production"})
    if not trusted.all():
        print("Synthetic or unverified labels detected; skipping model quality calculation")
        return False
    return True


def check_model_quality():
    label_file = "monitoring/logs/predictions_with_labels.csv"
    if not os.path.exists(label_file):
        print("No label file; skipping model quality check")
        return None

    df = pd.read_csv(label_file)
    required_cols = {"score", "true_label"}
    if not required_cols.issubset(df.columns):
        print(f"Missing columns {required_cols} in {label_file}")
        return None
    if not _has_trusted_labels(df):
        return None

    df = df.dropna(subset=["score", "true_label"])
    if len(df) < 10:
        print("Not enough verified labeled rows")
        return None

    y_true = df["true_label"].astype(int)
    y_pred_proba = df["score"]
    y_pred = (y_pred_proba >= best_threshold).astype(int)

    auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)

    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1-score: {f1:.4f}")

    os.makedirs("monitoring/drift_logs", exist_ok=True)
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "component": "model_quality",
        "roc_auc": float(auc),
        "f1_score": float(f1),
        "n_samples": int(len(df)),
        "label_source": "verified",
    }

    with open("monitoring/drift_logs/drift_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return auc, f1


if __name__ == "__main__":
    check_model_quality()
