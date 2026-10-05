import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = "sqlite:///mlflow/mlflow.db"
DEFAULT_EXPERIMENT_NAME = "wallet-risk-analyst"
DEFAULT_REGISTERED_MODEL_NAME = "WalletRiskLightGBMPackage"


@dataclass(frozen=True)
class MlflowRunResult:
    status: str
    run_id: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    tracking_uri: str | None = None
    registry_uri: str | None = None
    message: str | None = None


def import_mlflow():
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ModuleNotFoundError as exc:
        if exc.name == "mlflow":
            return None, None
        raise
    return mlflow, MlflowClient


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def configure_mlflow(mlflow, tracking_uri=None, registry_uri=None, experiment_name=None):
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    registry_uri = registry_uri or os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)
    experiment_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)

    if tracking_uri.startswith("sqlite:///"):
        db_path = PROJECT_ROOT / tracking_uri.replace("sqlite:///", "")
        db_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri, registry_uri, experiment_name


def load_json(path):
    with (PROJECT_ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def log_params_prefixed(mlflow, params: dict[str, Any], prefix="param"):
    for key, value in params.items():
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        mlflow.log_param(f"{prefix}.{key}", value)


def log_metrics_from_results(mlflow, metrics):
    mlflow.log_metric("threshold", float(metrics["best_threshold"]))
    mlflow.log_metric("validation_f1_at_threshold", float(metrics["validation_f1_at_threshold"]))
    mlflow.log_metric("train_roc_auc", float(metrics["train_roc_auc"]))
    mlflow.log_metric("val_roc_auc", float(metrics["val_roc_auc"]))
    mlflow.log_metric("test_roc_auc", float(metrics["test_roc_auc"]))
    mlflow.log_metric("test_accuracy", float(metrics["test_report"]["accuracy"]))
    mlflow.log_metric("test_class_1_precision", float(metrics["test_report"]["1"]["precision"]))
    mlflow.log_metric("test_class_1_recall", float(metrics["test_report"]["1"]["recall"]))
    mlflow.log_metric("test_class_1_f1", float(metrics["test_report"]["1"]["f1-score"]))


def log_artifact_if_exists(mlflow, relative_path, artifact_path=None):
    path = PROJECT_ROOT / relative_path
    if path.exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_model_package(
    *,
    run_name,
    registered_model_name=DEFAULT_REGISTERED_MODEL_NAME,
    tracking_uri=None,
    registry_uri=None,
    experiment_name=None,
    tags=None,
) -> MlflowRunResult:
    mlflow, _ = import_mlflow()
    if mlflow is None:
        return MlflowRunResult(
            status="pending_missing_mlflow",
            message="Install mlflow to execute tracking and registry logging.",
        )

    tracking_uri, registry_uri, _ = configure_mlflow(
        mlflow,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        experiment_name=experiment_name,
    )

    metrics = load_json("results/LightGBM_metrics.json")
    manifest = load_json("data/data_manifest.json")
    model = joblib.load(PROJECT_ROOT / "models" / "lightgbm_model.pkl")
    best_params = joblib.load(PROJECT_ROOT / "models" / "lightgbm_best_params.pkl")
    feature_names = joblib.load(PROJECT_ROOT / "models" / "lightgbm_feature_names.pkl")
    preprocessing = joblib.load(PROJECT_ROOT / "models" / "lightgbm_preprocessing.pkl")
    threshold = joblib.load(PROJECT_ROOT / "models" / "lightgbm_best_threshold.pkl")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "component": "scoring",
                "project": "wallet-risk-analyst",
                "git_commit": git_commit(),
                "data_sha256": manifest["dataset"]["sha256"],
                "threshold_source": metrics["threshold_source"],
                "package_kind": "model_preprocessing_threshold",
                **(tags or {}),
            }
        )
        log_params_prefixed(mlflow, best_params, prefix="lightgbm")
        mlflow.log_param("feature_count", len(feature_names))
        mlflow.log_param("threshold", float(threshold))
        mlflow.log_param("preprocessing", preprocessing.get("preprocessing"))
        mlflow.log_param("split_strategy", manifest["split"]["strategy"])
        mlflow.log_param("split_seed", manifest["split"]["random_seed"])
        log_metrics_from_results(mlflow, metrics)

        for path, artifact_path in [
            ("data/data_manifest.json", "data"),
            ("models/lightgbm_feature_names.pkl", "model_package"),
            ("models/lightgbm_preprocessing.pkl", "model_package"),
            ("models/lightgbm_best_threshold.pkl", "model_package"),
            ("models/lightgbm_best_params.pkl", "model_package"),
            ("models/lightgbm_param_search.pkl", "model_package"),
            ("results/LightGBM_metrics.json", "evaluation"),
            ("results/LightGBM_classification_report.txt", "evaluation"),
            ("requirements.txt", "environment"),
        ]:
            log_artifact_if_exists(mlflow, path, artifact_path=artifact_path)

        input_example = pd.DataFrame([{name: 0.0 for name in feature_names}])
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
            serialization_format="cloudpickle",
        )
        model_uri = f"runs:/{run.info.run_id}/model"
        version = mlflow.register_model(model_uri, registered_model_name)

    return MlflowRunResult(
        status="logged",
        run_id=run.info.run_id,
        model_name=registered_model_name,
        model_version=str(version.version),
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )


def set_model_alias(model_name, alias, version):
    mlflow, MlflowClient = import_mlflow()
    if mlflow is None:
        return MlflowRunResult(status="pending_missing_mlflow")
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, str(version))
    return MlflowRunResult(status="alias_set", model_name=model_name, model_version=str(version))


def reset_local_mlflow_store():
    store = PROJECT_ROOT / "mlflow"
    if store.exists():
        shutil.rmtree(store)


def result_to_dict(result: MlflowRunResult):
    return asdict(result)
