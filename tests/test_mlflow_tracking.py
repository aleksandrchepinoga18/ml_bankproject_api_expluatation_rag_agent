import json
from pathlib import Path

from src import mlflow_tracking


def test_mlflow_import_is_optional():
    mlflow, client = mlflow_tracking.import_mlflow()
    if mlflow is None:
        assert client is None
    else:
        assert client is not None


def test_result_to_dict_preserves_pending_status():
    result = mlflow_tracking.MlflowRunResult(
        status="pending_missing_mlflow",
        message="Install mlflow",
    )

    assert mlflow_tracking.result_to_dict(result) == {
        "status": "pending_missing_mlflow",
        "run_id": None,
        "model_name": None,
        "model_version": None,
        "tracking_uri": None,
        "registry_uri": None,
        "message": "Install mlflow",
    }


def test_data_manifest_tracks_mlflow_requirement():
    manifest_path = Path(__file__).resolve().parents[1] / "data" / "data_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["dependencies"]["requirements_txt"]["mlflow"] == "3.16.1"
