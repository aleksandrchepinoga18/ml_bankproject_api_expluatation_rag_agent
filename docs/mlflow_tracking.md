# MLflow tracking and registry

Дата: 2026-09-29.

Цель: подключить MLflow Tracking к существующему train pipeline и показать локальную репетицию двух версий зарегистрированного пакета модели с возвратом alias `champion` на предыдущую версию.

## Configuration

- Tracking URI по умолчанию: `sqlite:///mlflow/mlflow.db`.
- Registry URI по умолчанию совпадает с Tracking URI.
- Registered model name: `WalletRiskLightGBMPackage`.
- Пакет включает модель, feature order, preprocessing, threshold, data manifest, метрики и requirements.

## Result

- Status: `completed`.
- Tracking URI: `sqlite:///mlflow/mlflow.db`.
- Registry URI: `sqlite:///mlflow/mlflow.db`.
- Version 1 run: `2f2da3016d3d4fd794bc47d198ddaa24`.
- Version 1 model version: `1`.
- Version 2 run: `f38e6d75bf134180906650e69f4fb79e`.
- Version 2 model version: `2`.
- Rollback: champion alias returned to version `1`.

## Notes

- Это локальная репетиция, не удаленный deployment.
- Автоматическое продвижение модели не включено.
- Переобучение на synthetic labels не выполняется.
