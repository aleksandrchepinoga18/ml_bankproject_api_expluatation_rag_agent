import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def simulate_labels(days_back=7):
    """Create synthetic labels for UI/plumbing demos only.

    These labels are sampled from 1 - score and must not be used as evidence of
    model quality.
    """
    all_features = []
    all_scores = []

    for i in range(days_back):
        date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        log_file = f"monitoring/logs/predictions_{date}.jsonl"
        if not os.path.exists(log_file):
            continue

        with open(log_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    features = entry.get("features") if isinstance(entry, dict) else None
                    if isinstance(features, dict) and "score" in entry:
                        all_features.append(features)
                        all_scores.append(entry["score"])
                except Exception as e:
                    print(f"Error reading {log_file}:{line_num}: {e}")

    if not all_features:
        print("No valid logs for synthetic label generation")
        return

    df_features = pd.DataFrame(all_features)
    df = pd.DataFrame({"score": all_scores})
    df = pd.concat([df, df_features], axis=1)

    np.random.seed(42)
    df["true_label"] = np.random.binomial(1, 1 - df["score"])
    df["label_source"] = "synthetic_inverse_score"
    df["label_warning"] = "demo label; do not use for model quality"

    output_file = "monitoring/logs/predictions_with_labels.csv"
    df.to_csv(output_file, index=False)
    print(f"Synthetic labels saved to {output_file}; do not use them as quality evidence")


if __name__ == "__main__":
    simulate_labels()
