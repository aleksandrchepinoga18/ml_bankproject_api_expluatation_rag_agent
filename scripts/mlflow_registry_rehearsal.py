import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mlflow_tracking import (  # noqa: E402
    DEFAULT_REGISTERED_MODEL_NAME,
    log_model_package,
    result_to_dict,
    set_model_alias,
)


RESULT_PATH = PROJECT_ROOT / "results" / "mlflow_rehearsal.json"
REPORT_PATH = PROJECT_ROOT / "docs" / "mlflow_tracking.md"


def write_outputs(result):
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# MLflow tracking and registry",
        "",
        "Дата: 2026-09-29.",
        "",
        "Цель: подключить MLflow Tracking к существующему train pipeline и "
        "показать локальную репетицию двух версий зарегистрированного пакета "
        "модели с возвратом alias `champion` на предыдущую версию.",
        "",
        "## Configuration",
        "",
        "- Tracking URI по умолчанию: `sqlite:///mlflow/mlflow.db`.",
        "- Registry URI по умолчанию совпадает с Tracking URI.",
        f"- Registered model name: `{DEFAULT_REGISTERED_MODEL_NAME}`.",
        "- Пакет включает модель, feature order, preprocessing, threshold, "
        "data manifest, метрики и requirements.",
        "",
        "## Result",
        "",
        f"- Status: `{result['status']}`.",
    ]

    if result["status"] == "completed":
        lines.extend(
            [
                f"- Tracking URI: `{result['tracking_uri']}`.",
                f"- Registry URI: `{result['registry_uri']}`.",
                f"- Version 1 run: `{result['runs'][0]['run_id']}`.",
                f"- Version 1 model version: `{result['runs'][0]['model_version']}`.",
                f"- Version 2 run: `{result['runs'][1]['run_id']}`.",
                f"- Version 2 model version: `{result['runs'][1]['model_version']}`.",
                f"- Rollback: champion alias returned to version "
                f"`{result['rollback']['champion_version_after_rollback']}`.",
            ]
        )
    else:
        lines.extend(
            [
                f"- Reason: `{result.get('reason')}`.",
                "- `mlflow==3.16.1` добавлен в `requirements.txt`, но текущая "
                "локальная среда должна установить зависимость перед реальным "
                "Tracking/Registry запуском.",
            ]
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Это локальная репетиция, не удаленный deployment.",
            "- Автоматическое продвижение модели не включено.",
            "- Переобучение на synthetic labels не выполняется.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    first = log_model_package(
        run_name="registry_rehearsal_v1",
        tags={"registry_rehearsal": "true", "rehearsal_version": "1"},
    )
    if first.status != "logged":
        result = {
            "status": "pending_missing_mlflow",
            "reason": first.message,
            "runs": [result_to_dict(first)],
        }
        write_outputs(result)
        print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
        print(f"Wrote {RESULT_PATH.relative_to(PROJECT_ROOT)}")
        return

    first_alias = set_model_alias(first.model_name, "champion", first.model_version)

    second = log_model_package(
        run_name="registry_rehearsal_v2",
        tags={"registry_rehearsal": "true", "rehearsal_version": "2"},
    )
    if second.status != "logged":
        result = {
            "status": "partial",
            "runs": [result_to_dict(first), result_to_dict(second)],
            "reason": second.message,
        }
        write_outputs(result)
        print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
        print(f"Wrote {RESULT_PATH.relative_to(PROJECT_ROOT)}")
        return

    candidate_alias = set_model_alias(second.model_name, "candidate", second.model_version)
    rollback_alias = set_model_alias(first.model_name, "champion", first.model_version)

    result = {
        "status": "completed",
        "tracking_uri": first.tracking_uri,
        "registry_uri": first.registry_uri,
        "model_name": first.model_name,
        "runs": [result_to_dict(first), result_to_dict(second)],
        "aliases": {
            "initial_champion": result_to_dict(first_alias),
            "candidate": result_to_dict(candidate_alias),
        },
        "rollback": {
            "operation": result_to_dict(rollback_alias),
            "champion_version_after_rollback": first.model_version,
        },
    }
    write_outputs(result)
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {RESULT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
