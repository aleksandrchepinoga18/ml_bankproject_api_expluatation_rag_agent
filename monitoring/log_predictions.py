# monitoring/log_predictions.py
import json
import os
from datetime import datetime

def log_prediction(features: dict, score: float, model_usage="lightgbm_v1"):
    """
    Логирует предикт в JSONL-файл с датой и фичами
    """
    # Используем datetime.utcnow() — работает везде, несмотря на предупреждение
    timestamp = datetime.utcnow().isoformat()
    date_str = datetime.utcnow().strftime("%Y-%m-%d")

    log_entry = {
        "timestamp": timestamp,
        "model_version": model_usage,
        "score": float(score),
        "features": features
    }

    os.makedirs("monitoring/logs", exist_ok=True)
    log_path = f"monitoring/logs/predictions_{date_str}.jsonl"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")