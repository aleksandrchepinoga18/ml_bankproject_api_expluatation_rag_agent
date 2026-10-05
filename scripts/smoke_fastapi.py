import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("AGENT_LLM_PROVIDER", "disabled")

from fastapi.testclient import TestClient  # noqa: E402

from app.fastapi_app import app, get_agent  # noqa: E402
from src.retrieval import load_manifest  # noqa: E402


def first_wallet_address():
    manifest = load_manifest()
    return next(doc["wallet_address"] for doc in manifest["documents"] if doc["wallet_address"])


def main():
    get_agent.cache_clear()
    client = TestClient(app)
    wallet_address = first_wallet_address()

    checks = []
    for name, method, path, payload in [
        ("live", "get", "/health/live", None),
        ("ready", "get", "/health/ready", None),
        ("version", "get", "/version", None),
        (
            "document_request",
            "post",
            "/analyze",
            {
                "question": f"Show document sources for wallet {wallet_address}",
                "wallet_address": wallet_address,
            },
        ),
        (
            "no_document_need",
            "post",
            "/chat",
            {"message": "What can this assistant do?"},
        ),
        (
            "address_without_features",
            "post",
            "/analyze",
            {
                "question": f"What is the risk score for {wallet_address}",
                "wallet_address": wallet_address,
            },
        ),
    ]:
        response = getattr(client, method)(path, json=payload) if payload is not None else getattr(client, method)(path)
        body = response.json()
        checks.append(
            {
                "name": name,
                "status_code": response.status_code,
                "ok": response.status_code == 200,
                "status": body.get("status") or body.get("data_status") or body.get("analysis", {}).get("data_status"),
                "intent": body.get("intent") or body.get("analysis", {}).get("intent"),
                "source_count": len(body.get("sources", []) or body.get("analysis", {}).get("sources", [])),
            }
        )

    result = {"status": "ok" if all(item["ok"] for item in checks) else "failed", "checks": checks}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
