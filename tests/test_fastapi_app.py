import builtins

from fastapi.testclient import TestClient

from app import fastapi_app
from src.retrieval import load_manifest


def _wallet_address():
    manifest = load_manifest()
    return next(doc["wallet_address"] for doc in manifest["documents"] if doc["wallet_address"])


def _client(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "disabled")
    fastapi_app.get_agent.cache_clear()
    return TestClient(fastapi_app.app)


def test_health_and_version_endpoints(monkeypatch):
    client = _client(monkeypatch)

    live = client.get("/health/live")
    ready = client.get("/health/ready")
    version = client.get("/version")

    assert live.status_code == 200
    assert live.json()["status"] == "live"
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert version.status_code == 200
    assert version.json()["status"] == "ok"
    assert version.json()["agent_version"]
    assert version.json()["model_version"]


def test_version_endpoint_reports_model_error_without_500(monkeypatch):
    def raise_model_error(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "app" and fromlist == ("api",):
            raise OSError("libgomp.so.1: cannot open shared object file")
        return original_import(name, globals, locals, fromlist, level)

    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", raise_model_error)
    client = _client(monkeypatch)

    response = client.get("/version")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "degraded"
    assert body["model_version"] is None
    assert body["schema_version"] is None
    assert "libgomp.so.1" in body["errors"]["model"]


def test_analyze_document_request_returns_sources(monkeypatch):
    client = _client(monkeypatch)
    wallet_address = _wallet_address()

    response = client.post(
        "/analyze",
        json={
            "question": f"Show document sources for wallet {wallet_address}",
            "wallet_address": wallet_address,
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["execution_mode"] == "langgraph"
    assert body["intent"] == "documents"
    assert body["risk"]["status"] == "not_requested"
    assert body["sources"]
    assert all(source["wallet_address"] == wallet_address for source in body["sources"])


def test_chat_without_document_need_can_skip_tools(monkeypatch):
    client = _client(monkeypatch)

    response = client.post("/chat", json={"message": "What can this assistant do?"})
    body = response.json()

    assert response.status_code == 200
    assert body["answer"]
    assert body["analysis"]["tool_selection_source"] == "none"
    assert body["analysis"]["tool_statuses"] == []


def test_chat_russian_capability_question_returns_capabilities_without_tools(monkeypatch):
    client = _client(monkeypatch)

    response = client.post("/chat", json={"message": "Что ты умеешь?"})
    body = response.json()

    assert response.status_code == 200
    assert body["analysis"]["data_status"] == "ok"
    assert body["analysis"]["tool_selection_source"] == "none"
    assert body["analysis"]["tool_statuses"] == []
    assert body["analysis"]["llm"]["status"] == "skipped"
    assert "анализом риска кошелька" in body["answer"]
    assert "не поддерживается" not in body["answer"].lower()


def test_analyze_address_without_features_returns_insufficient_data(monkeypatch):
    client = _client(monkeypatch)
    wallet_address = _wallet_address()

    response = client.post(
        "/analyze",
        json={
            "question": f"What is the risk score for {wallet_address}",
            "wallet_address": wallet_address,
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["data_status"] == "insufficient_data"
    assert body["risk"]["status"] == "insufficient_data"
    assert body["risk"]["score"] is None


def test_metrics_endpoint_reports_observability_shape(monkeypatch):
    client = _client(monkeypatch)

    response = client.get("/metrics")
    body = response.json()

    assert response.status_code == 200
    assert "requests" in body
    assert "latency_ms" in body
    assert "source_correctness" in body
    assert "versions" in body
    assert "definition" in body["source_correctness"]
