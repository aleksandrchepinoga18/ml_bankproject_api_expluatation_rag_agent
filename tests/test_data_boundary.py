import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.data_preparation import load_and_clean_data, remove_high_corr_features, split_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.parquet"
MANIFEST_PATH = PROJECT_ROOT / "data" / "data_manifest.json"
METRICS_PATH = PROJECT_ROOT / "results" / "LightGBM_metrics.json"


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _manifest():
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _full_dataset_path() -> Path:
    if not DATASET_PATH.exists():
        pytest.skip(
            "full data check requires unpublished data/dataset.parquet; "
            "CI intentionally does not download or commit it"
        )
    return DATASET_PATH


def test_synthetic_wallet_group_split_has_no_intersections(synthetic_dataset_path: Path):
    df = remove_high_corr_features(load_and_clean_data(str(synthetic_dataset_path)))
    train, validation, test = split_data(df, random_state=42, group_col="wallet_address")

    parts = {"train": train, "validation": validation, "test": test}
    wallet_sets = {
        name: set(part["wallet_address"].dropna().astype(str))
        for name, part in parts.items()
    }

    assert {name: len(part) for name, part in parts.items()} == {
        "train": 4,
        "validation": 2,
        "test": 2,
    }
    assert all(len(wallets) == len(parts[name]) for name, wallets in wallet_sets.items())
    assert not (wallet_sets["train"] & wallet_sets["validation"])
    assert not (wallet_sets["train"] & wallet_sets["test"])
    assert not (wallet_sets["validation"] & wallet_sets["test"])
    assert "borrow_timestamp" not in train.columns
    assert "market_rocp" not in train.columns
    assert "wallet_address" in train.columns


def test_full_dataset_manifest_matches_dataset_contract():
    dataset_path = _full_dataset_path()
    manifest = _manifest()
    df = pd.read_parquet(dataset_path)

    assert manifest["dataset"]["sha256"] == _sha256_file(dataset_path)
    assert manifest["dataset"]["rows"] == len(df)
    assert manifest["dataset"]["columns"] == len(df.columns)
    assert manifest["dataset"]["unique_wallets"] == df["wallet_address"].nunique()
    assert manifest["dataset"]["target_counts"] == {
        str(key): int(value)
        for key, value in df["target"].value_counts().sort_index().items()
    }

    schema_names = [column["name"] for column in manifest["dataset"]["schema"]]
    assert schema_names == list(df.columns)
    assert manifest["model_boundary"]["unit_of_prediction"] == (
        "one row-level wallet risk observation"
    )
    assert manifest["model_boundary"]["feature_store_status"] == "not_available"


def test_full_dataset_wallet_group_split_matches_manifest():
    dataset_path = _full_dataset_path()
    manifest = _manifest()
    df = remove_high_corr_features(load_and_clean_data(str(dataset_path)))
    train, validation, test = split_data(df, random_state=42, group_col="wallet_address")

    parts = {"train": train, "validation": validation, "test": test}
    wallet_sets = {
        name: set(part["wallet_address"].dropna().astype(str))
        for name, part in parts.items()
    }

    assert manifest["split"]["rows"] == {name: len(part) for name, part in parts.items()}
    assert manifest["split"]["unique_wallets"] == {
        name: len(wallets) for name, wallets in wallet_sets.items()
    }
    assert not (wallet_sets["train"] & wallet_sets["validation"])
    assert not (wallet_sets["train"] & wallet_sets["test"])
    assert not (wallet_sets["validation"] & wallet_sets["test"])
    assert manifest["split"]["wallet_intersections"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }


def test_full_dataset_temporal_boundary_records_open_leakage_review():
    dataset_path = _full_dataset_path()
    manifest = _manifest()
    df = pd.read_parquet(dataset_path)
    temporal = manifest["temporal_checks"]

    assert temporal["prediction_time_proxy"] == "borrow_timestamp"
    assert temporal["status"] == "requires_review"
    assert "last_tx_timestamp" in temporal["leakage_candidates"]
    assert temporal["post_prediction_timestamp_counts"]["last_tx_timestamp"] == int(
        (df["last_tx_timestamp"] > df["borrow_timestamp"]).sum()
    )


def test_saved_metrics_are_preserved_in_manifest():
    manifest = _manifest()
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    artifact_metrics = manifest["artifacts"]["metrics"]
    assert artifact_metrics["threshold_source"] == "validation_f1"
    assert artifact_metrics["best_threshold"] == metrics["best_threshold"]
    assert artifact_metrics["test_roc_auc"] == metrics["test_roc_auc"]
    assert artifact_metrics["test_accuracy"] == metrics["test_report"]["accuracy"]
    assert artifact_metrics["test_class_1_f1"] == metrics["test_report"]["1"]["f1-score"]
