import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.fastapi_app import version_payload  # noqa: E402
from monitoring.observability import (  # noqa: E402
    DASHBOARD_PATH,
    SNAPSHOT_PATH,
    collect_metrics,
    render_dashboard,
    write_monitoring_snapshot,
    write_test_alert,
)


def main():
    metrics = collect_metrics(versions=version_payload())
    write_monitoring_snapshot(metrics)
    render_dashboard(metrics)
    alert_path = write_test_alert(metrics)
    result = {
        "status": "ok",
        "snapshot": str(SNAPSHOT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "dashboard": str(DASHBOARD_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "test_alert": str(alert_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "external_alert_sent": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
