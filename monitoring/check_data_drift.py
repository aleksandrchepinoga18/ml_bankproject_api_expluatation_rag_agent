import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

PSI_THRESHOLD = 0.2
KS_PVALUE_THRESHOLD = 0.05


def calculate_psi(expected, actual, bins=10):
    expected = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    actual = pd.Series(actual).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(expected) < 2 or len(actual) < 2:
        return np.nan

    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.unique(np.percentile(expected, quantiles))
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    expected_bins = np.histogram(expected, bins=bin_edges)[0] + 1
    actual_bins = np.histogram(actual, bins=bin_edges)[0] + 1

    expected_dist = expected_bins / expected_bins.sum()
    actual_dist = actual_bins / actual_bins.sum()
    return float(np.sum((expected_dist - actual_dist) * np.log(expected_dist / actual_dist)))


def _load_recent_feature_logs():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    yesterday = (datetime.utcnow() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    frames = []
    for date in [today, yesterday]:
        log_file = f"monitoring/logs/predictions_{date}.jsonl"
        if not os.path.exists(log_file):
            continue

        features_list = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    features = entry.get("features") if isinstance(entry, dict) else None
                    if isinstance(features, dict):
                        features_list.append(features)
                    else:
                        print(f"Invalid feature log format in {log_file}:{line_num}")
                except Exception as e:
                    print(f"Error reading {log_file}:{line_num}: {e}")

        if features_list:
            frames.append(pd.DataFrame(features_list))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def check_data_drift():
    ref_path = "monitoring/reference/reference_features.parquet"
    if not os.path.exists(ref_path):
        print("No reference features; skipping feature drift check")
        return False

    ref_df = pd.read_parquet(ref_path)
    current_df = _load_recent_feature_logs()
    if current_df.empty:
        print("No recent feature logs for drift analysis")
        return False

    common_features = ref_df.columns.intersection(current_df.columns)
    numeric_features = [
        col
        for col in common_features
        if pd.api.types.is_numeric_dtype(ref_df[col]) and pd.api.types.is_numeric_dtype(current_df[col])
    ]
    if not numeric_features:
        print("No common numeric features between reference and current data")
        return False

    drift_detected = False
    results = {}
    for col in numeric_features:
        ref_vals = ref_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        curr_vals = current_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ref_vals) < 10 or len(curr_vals) < 10:
            continue

        _, pval = stats.ks_2samp(ref_vals, curr_vals)
        psi = calculate_psi(ref_vals, curr_vals)
        if np.isnan(psi):
            continue

        results[col] = {"psi": float(psi), "ks_pvalue": float(pval)}
        if psi > PSI_THRESHOLD or pval < KS_PVALUE_THRESHOLD:
            print(f"Feature drift candidate '{col}': PSI={psi:.4f}, KS p-value={pval:.4f}")
            drift_detected = True

    os.makedirs("monitoring/drift_logs", exist_ok=True)
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "component": "feature_drift",
        "drift_detected": bool(drift_detected),
        "verified": False,
        "psi_method": "reference_quantile_bins_on_original_scale",
        "details": results,
    }
    with open("monitoring/drift_logs/drift_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    if not drift_detected:
        print("Feature drift not detected")
    return drift_detected


if __name__ == "__main__":
    check_data_drift()
