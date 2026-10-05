import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "monitoring" / "reference"
LOG_DIR = PROJECT_ROOT / "monitoring" / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
FIXED_BINS_PATH = RESULTS_DIR / "reference_bins.json"
PSI_REPORT_PATH = RESULTS_DIR / "psi_fixed_bins_report.json"


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)


def fixed_reference_edges(reference: pd.Series, bins: int = 10) -> list[float]:
    values = _clean_numeric(reference)
    if len(values) < 2:
        return []
    edges = np.unique(np.percentile(values, np.linspace(0, 100, bins + 1))).astype(float)
    if len(edges) < 2:
        return []
    edges[0] = -np.inf
    edges[-1] = np.inf
    return [float(edge) for edge in edges]


def serializable_edges(edges: list[float]) -> list[float | str]:
    values: list[float | str] = []
    for edge in edges:
        if edge == -np.inf:
            values.append("-inf")
        elif edge == np.inf:
            values.append("inf")
        else:
            values.append(float(edge))
    return values


def psi_from_edges(reference: pd.Series, current: pd.Series, edges: list[float]) -> dict[str, Any]:
    if len(edges) < 2:
        return {"status": "not_run", "reason": "not_enough_reference_variation"}
    ref = _clean_numeric(reference)
    cur = _clean_numeric(current)
    if len(ref) < 2 or len(cur) < 2:
        return {
            "status": "not_run",
            "reason": "not_enough_values",
            "reference_count": int(len(ref)),
            "current_count": int(len(cur)),
        }
    edge_array = np.array(edges, dtype=float)
    ref_bins = np.histogram(ref, bins=edge_array)[0]
    cur_bins = np.histogram(cur, bins=edge_array)[0]
    empty_reference_bins = [idx for idx, value in enumerate(ref_bins.tolist()) if value == 0]
    empty_current_bins = [idx for idx, value in enumerate(cur_bins.tolist()) if value == 0]

    ref_smoothed = ref_bins + 1
    cur_smoothed = cur_bins + 1
    ref_dist = ref_smoothed / ref_smoothed.sum()
    cur_dist = cur_smoothed / cur_smoothed.sum()
    psi = float(np.sum((ref_dist - cur_dist) * np.log(ref_dist / cur_dist)))
    finite_edges = [edge for edge in edges if np.isfinite(edge)]
    below_reference_min = int((cur < min(finite_edges)).sum()) if finite_edges else 0
    above_reference_max = int((cur > max(finite_edges)).sum()) if finite_edges else 0
    return {
        "status": "ok",
        "psi": psi,
        "reference_count": int(len(ref)),
        "current_count": int(len(cur)),
        "empty_reference_bins": empty_reference_bins,
        "empty_current_bins": empty_current_bins,
        "current_below_reference_min": below_reference_min,
        "current_above_reference_max": above_reference_max,
        "method": "fixed_reference_quantile_bins_with_plus_one_smoothing",
    }


def load_reference_features() -> pd.DataFrame:
    parquet_path = REFERENCE_DIR / "reference_features.parquet"
    csv_path = REFERENCE_DIR / "reference_features.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(csv_path)


def load_reference_scores() -> pd.Series:
    parquet_path = REFERENCE_DIR / "reference_scores.parquet"
    csv_path = REFERENCE_DIR / "reference_scores.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)["score"]
    return pd.read_csv(csv_path)["score"]


def load_current_logs() -> tuple[pd.DataFrame, pd.Series]:
    features = []
    scores = []
    for path in sorted(LOG_DIR.glob("predictions_*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload.get("features"), dict):
                    features.append(payload["features"])
                if "score" in payload:
                    scores.append(payload["score"])
    return pd.DataFrame(features), pd.Series(scores, name="score")


def build_fixed_bins_report(max_features: int = 20) -> dict[str, Any]:
    ref_features = load_reference_features()
    ref_scores = load_reference_scores()
    current_features, current_scores = load_current_logs()

    feature_columns = [
        column
        for column in current_features.columns
        if column in ref_features.columns and pd.api.types.is_numeric_dtype(ref_features[column])
    ][:max_features]

    bins = {"score": fixed_reference_edges(ref_scores)}
    results = {"score": psi_from_edges(ref_scores, current_scores, bins["score"])}
    for column in feature_columns:
        bins[column] = fixed_reference_edges(ref_features[column])
        results[column] = psi_from_edges(ref_features[column], current_features[column], bins[column])

    report = {
        "status": "ok",
        "reference_source": {
            "features": "monitoring/reference/reference_features.parquet"
            if (REFERENCE_DIR / "reference_features.parquet").exists()
            else "monitoring/reference/reference_features.csv",
            "scores": "monitoring/reference/reference_scores.parquet"
            if (REFERENCE_DIR / "reference_scores.parquet").exists()
            else "monitoring/reference/reference_scores.csv",
        },
        "current_source": "monitoring/logs/predictions_*.jsonl",
        "fixed_bins_path": str(FIXED_BINS_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "psi_report_path": str(PSI_REPORT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "drift_is_quality_evidence": False,
        "auto_retrain_started": False,
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    display_bins = {name: serializable_edges(edges) for name, edges in bins.items()}
    FIXED_BINS_PATH.write_text(json.dumps(display_bins, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PSI_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
