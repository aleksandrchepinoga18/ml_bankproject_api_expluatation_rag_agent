# MLflow integration audit

Date: 2026-10-12.

## Environment

- Python executable: `C:\Users\giper\AppData\Local\Programs\Python\Python311\python.exe`.
- Python version: `3.11.6`.
- MLflow version: `3.16.1`.
- Tests run: `python -m pytest tests/test_mlflow_tracking.py -q` -> `3 passed, 1 warning`.

## Main checked store

- Tracking URI: `sqlite:///mlflow/mlflow.db`.
- Registry URI: `sqlite:///mlflow/mlflow.db`.
- SQLite DB: `ml-project/mlflow/mlflow.db`.
- Artifact location: `file:///C:/workplace/GitHub/Projekt_bamk_scor VSCODE/ml_bankproject_api_expluatation_rag_agent/ml-project/mlruns/1`.
- Experiments found:
  - `Default` (`experiment_id=0`): 0 runs.
  - `wallet-risk-analyst` (`experiment_id=1`): 3 runs.

## Existing runs read from store

- `bd2460fb497a4aeaab8fdc1386c11347`, run name `registry_rehearsal_v1`, status `FAILED`.
- `2f2da3016d3d4fd794bc47d198ddaa24`, run name `registry_rehearsal_v1`, status `FINISHED`.
- `f38e6d75bf134180906650e69f4fb79e`, run name `registry_rehearsal_v2`, status `FINISHED`.

These runs were read from the MLflow SQLite store and artifact store. No training was executed during this audit. Their metrics should be treated as imported/logged results from existing `results/LightGBM_metrics.json`, not as a new training run.

Verified existing params include: `feature_count=65`, `threshold=0.3578026090255801`, `split_seed=42`, `split_strategy=group split by wallet_address; wallet target is max(target)`, `preprocessing=replace inf with nan; fill missing values with train medians`, and LightGBM params such as `learning_rate=0.04404205637217672`, `n_estimators=347`, `num_leaves=118`, `max_depth=9`.

Verified existing metrics include: `threshold=0.3578026090255801`, `validation_f1_at_threshold=0.7249339377759967`, `train_roc_auc=0.9567929500257917`, `val_roc_auc=0.8752606766052883`, `test_roc_auc=0.8722954225342738`, `test_accuracy=0.8006692746382618`, `test_class_1_precision=0.7226852205826133`, `test_class_1_recall=0.7342119572253437`, `test_class_1_f1=0.728402990058825`.

Verified existing artifacts on each run include: `data/data_manifest.json`, `environment/requirements.txt`, `evaluation/LightGBM_metrics.json`, `evaluation/LightGBM_classification_report.txt`, and `model_package/lightgbm_feature_names.pkl`, `lightgbm_preprocessing.pkl`, `lightgbm_best_threshold.pkl`, `lightgbm_best_params.pkl`, `lightgbm_param_search.pkl`.

Verified registered model artifacts exist under MLflow 3 logged model storage:

- Version 1: `WalletRiskLightGBMPackage`, source `models:/m-9351b3c37a13413198b747ba081a16da`, run `2f2da3016d3d4fd794bc47d198ddaa24`, status `READY`, files include `MLmodel`, `model.pkl`, `conda.yaml`, `python_env.yaml`, `requirements.txt`, `input_example.json`, `serving_input_example.json`.
- Version 2: `WalletRiskLightGBMPackage`, source `models:/m-e217bbe2b5cb4efaa604242fbd28a8c3`, run `f38e6d75bf134180906650e69f4fb79e`, status `READY`, same model artifact file set.

Registry: checked read-only. Registered model `WalletRiskLightGBMPackage` exists; alias `champion` points to version 1 and alias `candidate` points to version 2.

## Verification run

Separate local verification store was used to avoid modifying old experiments.

- Tracking URI: `sqlite:///mlflow_verification/mlflow.db`.
- Registry URI: `sqlite:///mlflow_verification/mlflow.db`.
- SQLite DB: `ml-project/mlflow_verification/mlflow.db`.
- Artifact root: `ml-project/mlflow_verification/artifacts`.
- Experiment: `mlflow-integration-verification` (`experiment_id=1`).
- Run: `ea55d814134e4d21b4392b2b2a87399f`, run name `verification_read_write_2026-10-12`.
- Params read back: `verification_param=round_trip`, `tracking_configured_by=src.mlflow_tracking.configure_mlflow`.
- Metrics read back: `verification_metric=1.0`.
- Artifacts read back: `verification/roundtrip.json`.

## UI

- MLflow UI is running at `http://127.0.0.1:13639`.
- Backend store: `sqlite:///mlflow/mlflow.db`.
- Open experiment `wallet-risk-analyst`.
- Open run `2f2da3016d3d4fd794bc47d198ddaa24` or `f38e6d75bf134180906650e69f4fb79e` to see the verified imported params, metrics, run artifacts, and registered model links.

## Limitations

- Docker, corpus, index, model files, commit, and push were not touched.
- No dependencies were installed or downloaded.
- No model training or model overwrite was performed.
- The verification run checks tracking write/read through the project integration, but does not register a model.
- The main store contains one failed rehearsal run with logged params/metrics/artifacts; registry versions point only to the two finished runs.
