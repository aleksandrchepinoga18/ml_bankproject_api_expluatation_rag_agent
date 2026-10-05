import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preparation import load_and_clean_data, remove_high_corr_features, split_data


DATASET_PATH = PROJECT_ROOT / "data" / "dataset.parquet"
MANIFEST_PATH = PROJECT_ROOT / "data" / "data_manifest.json"
METRICS_PATH = PROJECT_ROOT / "results" / "LightGBM_metrics.json"
TOP50_PATH = PROJECT_ROOT / "results" / "LightGBM_top50_risky_wallets.csv"
FEATURE_NAMES_PATH = PROJECT_ROOT / "models" / "lightgbm_feature_names.pkl"
PREPROCESSING_PATH = PROJECT_ROOT / "models" / "lightgbm_preprocessing.pkl"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"

RANDOM_SEED = 42
GROUP_COL = "wallet_address"
TARGET_COL = "target"
PREDICTION_TIME_PROXY = "borrow_timestamp"

REMOVED_HIGH_CORR_FEATURES = [
    "market_rocp",
    "market_apo",
    "market_macdsignal_macdfix",
    "market_macd_macdfix",
    "borrow_block_number",
    "borrow_timestamp",
    "risky_first_tx_timestamp",
    "market_macd_macdext",
    "market_macd",
    "market_macdsignal",
    "liquidation_count",
]

VERSION_PACKAGES = [
    "Flask",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "scipy",
    "psycopg2-binary",
    "matplotlib",
    "seaborn",
    "shap",
    "pyarrow",
    "requests",
    "mlflow",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_text(values):
    text = "\n".join(str(v) for v in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def read_requirements(path):
    requirements = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "==" in stripped:
            name, version = stripped.split("==", 1)
            requirements[name] = version
        else:
            requirements[stripped] = None
    return requirements


def installed_versions(packages):
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def schema_for(df):
    schema = []
    for column in df.columns:
        role = "feature"
        if column == GROUP_COL:
            role = "entity_id"
        elif column == TARGET_COL:
            role = "label"
        elif column == PREDICTION_TIME_PROXY:
            role = "candidate_prediction_time"

        schema.append(
            {
                "name": column,
                "dtype": str(df[column].dtype),
                "role": role,
                "non_null": int(df[column].notna().sum()),
                "nulls": int(df[column].isna().sum()),
            }
        )
    return schema


def split_summary(df):
    train, validation, test = split_data(df, random_state=RANDOM_SEED, group_col=GROUP_COL)
    parts = {"train": train, "validation": validation, "test": test}
    wallet_sets = {
        name: set(part[GROUP_COL].dropna().astype(str))
        for name, part in parts.items()
    }
    return {
        "strategy": "group split by wallet_address; wallet target is max(target)",
        "random_seed": RANDOM_SEED,
        "rows": {name: int(len(part)) for name, part in parts.items()},
        "unique_wallets": {name: int(len(wallets)) for name, wallets in wallet_sets.items()},
        "wallet_intersections": {
            "train_validation": int(len(wallet_sets["train"] & wallet_sets["validation"])),
            "train_test": int(len(wallet_sets["train"] & wallet_sets["test"])),
            "validation_test": int(len(wallet_sets["validation"] & wallet_sets["test"])),
        },
        "wallet_group_hashes": {
            name: sha256_text(sorted(wallets))
            for name, wallets in wallet_sets.items()
        },
    }


def temporal_checks(df, model_features):
    timestamp_columns = [
        "first_tx_timestamp",
        "last_tx_timestamp",
        "risky_first_tx_timestamp",
        "risky_last_tx_timestamp",
    ]
    post_prediction_counts = {}
    for column in timestamp_columns:
        if column in df.columns:
            post_prediction_counts[column] = int((df[column] > df[PREDICTION_TIME_PROXY]).sum())

    leakage_candidates = [
        column
        for column, count in post_prediction_counts.items()
        if count > 0 and column in model_features
    ]
    model_temporal_features = [
        column
        for column in model_features
        if "timestamp" in column.lower()
        or "time_since" in column.lower()
        or column.endswith("_age")
        or "last_" in column.lower()
        or "first_" in column.lower()
    ]

    return {
        "prediction_time_proxy": PREDICTION_TIME_PROXY,
        "prediction_time_status": "candidate_proxy_requires_source_confirmation",
        "post_prediction_timestamp_counts": post_prediction_counts,
        "model_temporal_features": model_temporal_features,
        "leakage_candidates": leakage_candidates,
        "status": "requires_review" if leakage_candidates else "no_timestamp_order_issue_detected",
    }


def artifact_summary(model_features):
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    top50 = pd.read_csv(TOP50_PATH)
    return {
        "metrics_path": str(METRICS_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "metrics": {
            "threshold_source": metrics["threshold_source"],
            "best_threshold": metrics["best_threshold"],
            "validation_f1_at_threshold": metrics["validation_f1_at_threshold"],
            "train_roc_auc": metrics["train_roc_auc"],
            "val_roc_auc": metrics["val_roc_auc"],
            "test_roc_auc": metrics["test_roc_auc"],
            "test_accuracy": metrics["test_report"]["accuracy"],
            "test_class_1_precision": metrics["test_report"]["1"]["precision"],
            "test_class_1_recall": metrics["test_report"]["1"]["recall"],
            "test_class_1_f1": metrics["test_report"]["1"]["f1-score"],
        },
        "top50_path": str(TOP50_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "top50": {
            "rows": int(len(top50)),
            "unique_wallets": int(top50["wallet_address"].nunique()),
            "aggregation_rule": "max_row_probability",
            "min_probability": float(top50["probability"].min()),
            "max_probability": float(top50["probability"].max()),
            "min_records_count": int(top50["records_count"].min()),
            "max_records_count": int(top50["records_count"].max()),
        },
        "model_features_sha256": sha256_text(model_features),
    }


def build_manifest():
    raw_df = pd.read_parquet(DATASET_PATH)
    clean_df = remove_high_corr_features(load_and_clean_data(str(DATASET_PATH)))
    model_features = list(joblib.load(FEATURE_NAMES_PATH))
    preprocessing = joblib.load(PREPROCESSING_PATH)
    requirements = read_requirements(REQUIREMENTS_PATH)

    raw_feature_columns = [
        column for column in raw_df.columns if column not in {GROUP_COL, TARGET_COL}
    ]
    removed_present = [column for column in REMOVED_HIGH_CORR_FEATURES if column in raw_df.columns]

    return {
        "manifest_version": "1.0",
        "audit_date": "2026-09-29",
        "dataset": {
            "path": str(DATASET_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(DATASET_PATH),
            "rows": int(len(raw_df)),
            "columns": int(len(raw_df.columns)),
            "unique_wallets": int(raw_df[GROUP_COL].nunique()),
            "target_counts": {
                str(key): int(value)
                for key, value in raw_df[TARGET_COL].value_counts().sort_index().items()
            },
            "schema": schema_for(raw_df),
        },
        "model_boundary": {
            "unit_of_prediction": "one row-level wallet risk observation",
            "entity_id": GROUP_COL,
            "target": TARGET_COL,
            "prediction_time_proxy": PREDICTION_TIME_PROXY,
            "prediction_time_status": "candidate_proxy_requires_source_confirmation",
            "feature_store_status": "not_available",
        },
        "features": {
            "raw_candidate_feature_count": len(raw_feature_columns),
            "removed_before_training": removed_present,
            "model_feature_count": len(model_features),
            "model_features": model_features,
            "preprocessing": preprocessing.get("preprocessing"),
            "imputation_value_count": len(preprocessing.get("imputation_values", {})),
        },
        "split": split_summary(clean_df),
        "temporal_checks": temporal_checks(raw_df, model_features),
        "artifacts": artifact_summary(model_features),
        "dependencies": {
            "requirements_txt": requirements,
            "installed_versions": installed_versions(VERSION_PACKAGES),
        },
        "limitations": [
            "Dataset provenance and external source version are not documented.",
            "Feature availability at prediction time is not fully proven.",
            "Temporal leakage review remains open for timestamp and aggregate features.",
            "Saved metrics are row-level test metrics, not wallet-level or production quality metrics.",
        ],
    }


def main():
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
