import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

PSI_THRESHOLD = 0.1
KS_PVALUE_THRESHOLD = 0.05


def calculate_psi(expected, actual, bins=10):
    expected = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    actual = pd.Series(actual).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(expected) < 2 or len(actual) < 2:
        return np.nan

    bin_edges = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    expected_bins = np.histogram(expected, bins=bin_edges)[0] + 1
    actual_bins = np.histogram(actual, bins=bin_edges)[0] + 1

    expected_dist = expected_bins / expected_bins.sum()
    actual_dist = actual_bins / actual_bins.sum()
    return float(np.sum((expected_dist - actual_dist) * np.log(expected_dist / actual_dist)))


def _load_recent_scores():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    yesterday = (datetime.utcnow() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    scores = []
    for date in [today, yesterday]:
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
                    if isinstance(entry, dict) and "score" in entry:
                        scores.append(entry["score"])
                except Exception as e:
                    print(f"Error reading {log_file}:{line_num}: {e}")
    return np.array(scores, dtype=float)


def check_score_drift():
    ref_path = "monitoring/reference/reference_scores.parquet"
    if not os.path.exists(ref_path):
        print("No reference scores; skipping score drift check")
        return False

    ref_scores = pd.read_parquet(ref_path)["score"].dropna().values
    current_scores = _load_recent_scores()
    if len(current_scores) < 10:
        print("Not enough recent scores for drift analysis")
        return False

    psi = calculate_psi(ref_scores, current_scores)
    _, pval = stats.ks_2samp(ref_scores, current_scores)
    drift_detected = bool((psi > PSI_THRESHOLD) or (pval < KS_PVALUE_THRESHOLD))

    print(f"Score PSI: {psi:.4f}")
    print(f"KS p-value: {pval:.4f}")

    os.makedirs("monitoring/drift_logs", exist_ok=True)
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "component": "score_drift",
        "psi": float(psi),
        "ks_pvalue": float(pval),
        "drift_detected": drift_detected,
        "verified": False,
        "psi_method": "reference_quantile_bins_on_original_scale",
        "n_ref": int(len(ref_scores)),
        "n_current": int(len(current_scores)),
    }

    with open("monitoring/drift_logs/drift_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    if drift_detected:
        print("Score drift candidate detected; manual verification required")
    else:
        print("Score drift not detected")
    return drift_detected


if __name__ == "__main__":
    check_score_drift()
