import json
import math
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "monitoring" / "logs"
ALERT_DIR = PROJECT_ROOT / "monitoring" / "alerts"
SNAPSHOT_PATH = PROJECT_ROOT / "results" / "monitoring_snapshot.json"
DASHBOARD_PATH = PROJECT_ROOT / "docs" / "local_monitoring_dashboard.html"

AGENT_PATHS = {"/analyze", "/chat"}
SECRET_KEYS = {"authorization", "cookie", "set-cookie", "x-api-key", "api_key", "token", "password", "secret"}


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _date_suffix() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def request_log_path() -> Path:
    return LOG_DIR / f"api_requests_{_date_suffix()}.jsonl"


def agent_log_path() -> Path:
    return LOG_DIR / f"agent_requests_{_date_suffix()}.jsonl"


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def sanitize_headers(headers: dict[str, Any]) -> dict[str, str]:
    sanitized = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in SECRET_KEYS or any(secret in lowered for secret in SECRET_KEYS):
            sanitized[lowered] = "[redacted]"
        elif lowered in {"user-agent", "content-type", "accept"}:
            sanitized[lowered] = str(value)[:160]
    return sanitized


def log_api_request(
    *,
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
    headers: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    payload = {
        "timestamp": utc_now(),
        "event": "api_request",
        "request_id": request_id,
        "method": method,
        "path": path,
        "status_code": int(status_code),
        "latency_ms": round(float(latency_ms), 3),
        "availability_ok": 200 <= int(status_code) < 500,
        "error": error,
        "headers": sanitize_headers(headers or {}),
    }
    _append_jsonl(request_log_path(), payload)


def _tool_elapsed(tool_statuses: list[Any], name: str) -> float | None:
    values = [
        float(getattr(status, "elapsed_ms", 0.0))
        for status in tool_statuses
        if getattr(status, "name", None) == name and getattr(status, "elapsed_ms", None) is not None
    ]
    return round(sum(values), 3) if values else None


def source_correctness(response: Any) -> dict[str, Any]:
    """Define source correctness for monitoring.

    Denominator eligibility: an Agent response is counted when its intent is documents,
    combined, or general, or when it returns at least one document finding/source.
    Correctness: all returned document findings are supported, every source has
    document_id and chunk_id, and no validation error reports a missing,
    unsupported, or foreign-wallet source.
    """
    intent = getattr(response, "intent", None)
    findings = list(getattr(response, "document_findings", []) or [])
    sources = list(getattr(response, "sources", []) or [])
    validation_errors = list(getattr(response, "validation_errors", []) or [])
    eligible = intent in {"documents", "combined", "general"} or bool(findings or sources)
    invalid_error_prefixes = ("foreign_wallet_source:", "missing_chunk:", "unsupported_claim:")
    invalid_errors = [err for err in validation_errors if str(err).startswith(invalid_error_prefixes)]
    missing_source_fields = [
        getattr(source, "chunk_id", None) or "<missing_chunk_id>"
        for source in sources
        if not getattr(source, "document_id", None) or not getattr(source, "chunk_id", None)
    ]
    unsupported_findings = [
        getattr(finding, "chunk_id", None) or "<missing_chunk_id>"
        for finding in findings
        if not getattr(finding, "supported", False)
    ]
    correct = eligible and not invalid_errors and not missing_source_fields and not unsupported_findings
    return {
        "eligible": bool(eligible),
        "correct": bool(correct),
        "definition": (
            "eligible=response intent is documents/combined/general or response has sources/findings; "
            "correct=all findings supported, all sources have document_id/chunk_id, and no "
            "foreign_wallet_source/missing_chunk/unsupported_claim validation errors"
        ),
        "invalid_errors": invalid_errors,
        "missing_source_fields": missing_source_fields,
        "unsupported_findings": unsupported_findings,
    }


def log_agent_response(response: Any, *, route: str, total_latency_ms: float | None = None) -> None:
    correctness = source_correctness(response)
    payload = {
        "timestamp": utc_now(),
        "event": "agent_response",
        "route": route,
        "request_id": getattr(response, "request_id", None),
        "agent_version": getattr(response, "agent_version", None),
        "prompt_version": getattr(response, "prompt_version", None),
        "intent": getattr(response, "intent", None),
        "data_status": getattr(response, "data_status", None),
        "wallet_address_present": bool(getattr(response, "wallet_address", None)),
        "risk_status": getattr(getattr(response, "risk", None), "status", None),
        "score": getattr(getattr(response, "risk", None), "score", None),
        "source_count": len(getattr(response, "sources", []) or []),
        "document_finding_count": len(getattr(response, "document_findings", []) or []),
        "tool_statuses": [
            {
                "name": getattr(status, "name", None),
                "status": getattr(status, "status", None),
                "elapsed_ms": getattr(status, "elapsed_ms", None),
                "error": getattr(status, "error", None),
            }
            for status in (getattr(response, "tool_statuses", []) or [])
        ],
        "tool_latency_ms": {
            "scoring": _tool_elapsed(getattr(response, "tool_statuses", []) or [], "scoring"),
            "shap": _tool_elapsed(getattr(response, "tool_statuses", []) or [], "shap"),
            "retrieval": _tool_elapsed(getattr(response, "tool_statuses", []) or [], "retrieval"),
            "llm": getattr(getattr(response, "llm", None), "eval_count", None),
        },
        "llm": {
            "provider": getattr(getattr(response, "llm", None), "provider", None),
            "model": getattr(getattr(response, "llm", None), "model", None),
            "status": getattr(getattr(response, "llm", None), "status", None),
            "done_reason": getattr(getattr(response, "llm", None), "done_reason", None),
        },
        "validation_errors": list(getattr(response, "validation_errors", []) or []),
        "limitations": list(getattr(response, "limitations", []) or []),
        "source_correctness": correctness,
        "total_latency_ms": None if total_latency_ms is None else round(float(total_latency_ms), 3),
    }
    _append_jsonl(agent_log_path(), payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _percentile(values: list[float], percentile: float) -> float | None:
    values = sorted(value for value in values if not math.isnan(value))
    if not values:
        return None
    index = (len(values) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return round(values[int(index)], 3)
    weighted = values[lower] * (upper - index) + values[upper] * (index - lower)
    return round(weighted, 3)


def collect_metrics(*, lookback_days: int = 7, versions: dict[str, Any] | None = None) -> dict[str, Any]:
    request_rows = []
    agent_rows = []
    for path in sorted(LOG_DIR.glob("api_requests_*.jsonl"))[-lookback_days:]:
        request_rows.extend(_read_jsonl(path))
    for path in sorted(LOG_DIR.glob("agent_requests_*.jsonl"))[-lookback_days:]:
        agent_rows.extend(_read_jsonl(path))

    latencies = [float(row["latency_ms"]) for row in request_rows if row.get("latency_ms") is not None]
    status_counts = Counter(str(row.get("status_code")) for row in request_rows)
    total_requests = len(request_rows)
    error_count = sum(1 for row in request_rows if int(row.get("status_code") or 0) >= 500)
    available_count = sum(1 for row in request_rows if row.get("availability_ok"))

    source_denominator = sum(1 for row in agent_rows if row.get("source_correctness", {}).get("eligible"))
    source_numerator = sum(
        1
        for row in agent_rows
        if row.get("source_correctness", {}).get("eligible") and row.get("source_correctness", {}).get("correct")
    )
    insufficient_data_count = sum(1 for row in agent_rows if row.get("data_status") == "insufficient_data")
    no_documents_count = sum(1 for row in agent_rows if "no_documents_found" in (row.get("limitations") or []))
    invalid_answer_count = sum(1 for row in agent_rows if row.get("validation_errors"))
    answer_replacement_count = sum(
        1
        for row in agent_rows
        if any(str(item).startswith("llm_answer_replaced_") for item in (row.get("limitations") or []))
    )

    metrics = {
        "generated_at": utc_now(),
        "lookback_days": lookback_days,
        "versions": versions or {},
        "requests": {
            "total": total_requests,
            "availability_rate": None if total_requests == 0 else round(available_count / total_requests, 6),
            "error_rate_5xx": None if total_requests == 0 else round(error_count / total_requests, 6),
            "status_counts": dict(status_counts),
        },
        "latency_ms": {
            "count": len(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "agent": {
            "total": len(agent_rows),
            "insufficient_data_rate": None
            if not agent_rows
            else round(insufficient_data_count / len(agent_rows), 6),
            "no_documents_rate": None if not agent_rows else round(no_documents_count / len(agent_rows), 6),
            "invalid_answer_rate": None if not agent_rows else round(invalid_answer_count / len(agent_rows), 6),
            "answer_replacement_count": answer_replacement_count,
            "answer_replacement_rate": None
            if not agent_rows
            else round(answer_replacement_count / len(agent_rows), 6),
            "answer_replacement_definition": (
                "Agent responses whose limitations include an llm_answer_replaced_* marker; "
                "these are cases where the original LLM answer was rejected and a deterministic "
                "fallback answer was returned."
            ),
        },
        "source_correctness": {
            "definition": (
                "denominator: Agent responses where intent is documents/combined/general or response has "
                "sources/findings; numerator: eligible responses with supported findings, document_id/chunk_id "
                "on every source, and no foreign_wallet_source/missing_chunk/unsupported_claim validation errors"
            ),
            "numerator": source_numerator,
            "denominator": source_denominator,
            "rate": None if source_denominator == 0 else round(source_numerator / source_denominator, 6),
        },
    }
    return metrics


def write_monitoring_snapshot(metrics: dict[str, Any]) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_test_alert(metrics: dict[str, Any], *, reason: str = "manual_test_alert") -> Path:
    payload = {
        "timestamp": utc_now(),
        "event": "test_alert",
        "status": "not_sent_external",
        "recipient": "local_file_only",
        "reason": reason,
        "summary": {
            "request_total": metrics.get("requests", {}).get("total"),
            "error_rate_5xx": metrics.get("requests", {}).get("error_rate_5xx"),
            "latency_p95_ms": metrics.get("latency_ms", {}).get("p95"),
        },
    }
    path = ALERT_DIR / f"test_alert_{_date_suffix()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def render_dashboard(metrics: dict[str, Any]) -> None:
    DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Wallet Risk Local Monitoring</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #202124; }}
    h1 {{ font-size: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #d0d7de; border-radius: 6px; padding: 12px; }}
    .label {{ color: #57606a; font-size: 12px; text-transform: uppercase; }}
    .value {{ font-size: 22px; margin-top: 6px; }}
    pre {{ white-space: pre-wrap; background: #f6f8fa; padding: 12px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Wallet Risk Local Monitoring</h1>
  <p>Generated at {metrics.get("generated_at")}. Local dashboard only; no external alert messages are sent.</p>
  <div class="grid">
    <div class="card"><div class="label">Requests</div><div class="value">{metrics["requests"]["total"]}</div></div>
    <div class="card"><div class="label">Availability</div><div class="value">{metrics["requests"]["availability_rate"]}</div></div>
    <div class="card"><div class="label">5xx Error Rate</div><div class="value">{metrics["requests"]["error_rate_5xx"]}</div></div>
    <div class="card"><div class="label">Latency p95 ms</div><div class="value">{metrics["latency_ms"]["p95"]}</div></div>
    <div class="card"><div class="label">Source Correctness</div><div class="value">{metrics["source_correctness"]["rate"]}</div></div>
    <div class="card"><div class="label">Insufficient Data Rate</div><div class="value">{metrics["agent"]["insufficient_data_rate"]}</div></div>
    <div class="card"><div class="label">Answer Replacement Rate</div><div class="value">{metrics["agent"]["answer_replacement_rate"]}</div></div>
  </div>
  <h2>Definitions</h2>
  <pre>{metrics["source_correctness"]["definition"]}</pre>
  <pre>{metrics["agent"]["answer_replacement_definition"]}</pre>
  <h2>Raw Snapshot</h2>
  <pre>{json.dumps(metrics, ensure_ascii=False, indent=2)}</pre>
</body>
</html>
"""
    DASHBOARD_PATH.write_text(html, encoding="utf-8")
