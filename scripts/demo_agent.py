import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8080"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Readable terminal demo client for the running wallet-risk Agent API."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"API base URL. Default: {DEFAULT_BASE_URL}")
    parser.add_argument("--endpoint", choices=["analyze", "chat"], default="analyze", help="API endpoint to call.")
    parser.add_argument("--question", required=True, help="Question or chat message for the Agent.")
    parser.add_argument("--wallet-address", help="Wallet address context for the request.")
    parser.add_argument("--features-file", type=Path, help="JSON file with a feature object or {'features': {...}}.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieval hits requested from the API.")
    parser.add_argument("--trace", action="store_true", help="Print detailed Agent trace.")
    parser.add_argument("--save-json", type=Path, help="Save raw API JSON response to this file.")
    parser.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout in seconds.")
    return parser.parse_args()


def load_features(path: Path | None) -> tuple[Any, str | None]:
    if path is None:
        return None, None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        raise SystemExit(f"Could not read features file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Features file is not valid JSON: {path}: {exc}") from exc

    if isinstance(data, dict) and "features" in data:
        note = data.get("note")
        return data["features"], str(note) if note else None
    return data, None


def build_payload(args: argparse.Namespace) -> tuple[dict[str, Any], str | None]:
    features, features_note = load_features(args.features_file)
    if args.endpoint == "chat":
        payload: dict[str, Any] = {"message": args.question, "top_k": args.top_k}
    else:
        payload = {"question": args.question, "top_k": args.top_k}
    if args.wallet_address:
        payload["wallet_address"] = args.wallet_address
    if features is not None:
        payload["features"] = features
    return payload, features_note


def request_agent(args: argparse.Namespace, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    url = f"{args.base_url.rstrip('/')}/{args.endpoint}"
    started = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=args.timeout)
    except requests.exceptions.ConnectionError as exc:
        raise SystemExit(f"Connection error: could not reach {url}. Is the Docker API running? Details: {exc}") from exc
    except requests.exceptions.Timeout as exc:
        raise SystemExit(f"Timeout: {url} did not respond within {args.timeout:.0f}s.") from exc
    except requests.exceptions.RequestException as exc:
        raise SystemExit(f"HTTP request failed for {url}: {exc}") from exc

    elapsed_ms = (time.perf_counter() - started) * 1000
    try:
        body = response.json()
    except ValueError as exc:
        raise SystemExit(f"HTTP {response.status_code}: response is not JSON:\n{response.text}") from exc

    if response.status_code >= 400:
        detail = body.get("detail") if isinstance(body, dict) else body
        raise SystemExit(f"HTTP {response.status_code} from {url}: {detail}")
    return body, elapsed_ms


def analysis_from_response(endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
    if endpoint == "chat":
        analysis = body.get("analysis")
        if not isinstance(analysis, dict):
            raise SystemExit("Malformed /chat response: missing analysis object.")
        return analysis
    return body


def answer_from_response(endpoint: str, body: dict[str, Any], analysis: dict[str, Any]) -> str:
    if endpoint == "chat":
        return str(body.get("answer") or analysis.get("answer") or "")
    return str(analysis.get("answer") or "")


def fmt_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_section(title: str) -> None:
    print()
    print(f"== {title} ==")


def summarize_state(analysis: dict[str, Any]) -> str:
    risk = analysis.get("risk") or {}
    llm = analysis.get("llm") or {}
    risk_status = risk.get("status")
    llm_status = llm.get("status")
    validation_errors = analysis.get("validation_errors") or []

    if llm_status in {"unavailable", "error"}:
        return f"LLM {llm_status}: structured API response was returned with explicit LLM status."
    if risk_status == "not_requested":
        return "Risk not requested: the Agent should not present a model score for this request."
    if risk_status == "insufficient_data":
        return "Insufficient data: scoring was requested, but required model features were not supplied."
    if validation_errors:
        return "Completed with validation errors: review the validation_errors section."
    if risk_status == "scored":
        return "Successful scoring response."
    return f"Completed with risk.status={risk_status or '-'}."


def print_question_answer(question: str, answer: str, features_file: Path | None, features_note: str | None) -> None:
    print_section("Question")
    print(question)
    if features_file:
        print_section("Feature Input")
        print(f"features_file: {features_file}")
        if features_note:
            print(f"origin: {features_note}")
        else:
            print("origin: user-supplied feature file; provenance not declared in the file")
    print_section("Agent Answer")
    print(answer or "-")


def print_agent_summary(analysis: dict[str, Any]) -> None:
    print_section("Agent Routing")
    print(f"state: {summarize_state(analysis)}")
    print(f"intent: {fmt_value(analysis.get('intent'))}")
    print(f"execution_mode: {fmt_value(analysis.get('execution_mode'))}")
    selected = analysis.get("selected_tools") or []
    print(f"selected_tools: {', '.join(selected) if selected else '-'}")


def print_risk(analysis: dict[str, Any]) -> None:
    risk = analysis.get("risk") or {}
    top_features = risk.get("top_features") or []
    if not risk:
        return

    print_section("LightGBM Result")
    print(f"risk.status: {fmt_value(risk.get('status'))}")
    print(f"score: {fmt_value(risk.get('score'))}")
    print(f"threshold: {fmt_value(risk.get('threshold'))}")
    print(f"prediction: {fmt_value(risk.get('prediction'))}")
    print(f"model_version: {fmt_value(risk.get('model_version'))}")
    print(f"schema_version: {fmt_value(risk.get('schema_version'))}")
    print(f"reason: {fmt_value(risk.get('reason'))}")

    if top_features:
        print()
        print("Top features:")
        for item in top_features[:8]:
            print(
                "- "
                f"{item.get('feature')}: value={fmt_value(item.get('value'))}, "
                f"impact={fmt_value(item.get('impact'))}"
            )


def print_sources(analysis: dict[str, Any]) -> None:
    sources = analysis.get("sources") or []
    findings = analysis.get("document_findings") or []
    print_section("Documents And Sources")
    if not sources:
        print("-")
    for source in sources:
        source_kind = source.get("source_kind")
        note = " (synthetic demo; not evidence)" if source_kind == "synthetic_demo" else ""
        wallet = source.get("wallet_address") or "general"
        print(
            "- "
            f"{source.get('document_id')}/{source.get('chunk_id')} "
            f"kind={source_kind}{note}; wallet={wallet}; rank={fmt_value(source.get('rank'))}"
        )
    if findings:
        print()
        print("Supported findings:")
        for finding in findings[:5]:
            note = " (synthetic demo; not evidence)" if finding.get("synthetic_demo") else ""
            print(f"- {finding.get('document_id')}/{finding.get('chunk_id')}{note}: {finding.get('claim')}")


def print_llm(analysis: dict[str, Any]) -> None:
    llm = analysis.get("llm") or {}
    print_section("LLM")
    print(f"provider: {fmt_value(llm.get('provider'))}")
    print(f"model: {fmt_value(llm.get('model'))}")
    print(f"status: {fmt_value(llm.get('status'))}")
    print(f"done_reason: {fmt_value(llm.get('done_reason'))}")
    print(f"error: {fmt_value(llm.get('error'))}")
    print(f"eval_count: {fmt_value(llm.get('eval_count'))}")
    print(f"prompt_eval_count: {fmt_value(llm.get('prompt_eval_count'))}")


def print_limitations_and_tools(analysis: dict[str, Any]) -> None:
    print_section("Limitations, Validation, Tool Statuses")
    limitations = analysis.get("limitations") or []
    validation_errors = analysis.get("validation_errors") or []
    tool_statuses = analysis.get("tool_statuses") or []
    print("limitations:")
    print("- " + "\n- ".join(limitations) if limitations else "-")
    print("validation_errors:")
    print("- " + "\n- ".join(validation_errors) if validation_errors else "-")
    print("tool_statuses:")
    if not tool_statuses:
        print("-")
    for item in tool_statuses:
        elapsed = fmt_value(item.get("elapsed_ms"))
        error = fmt_value(item.get("error"))
        print(f"- {item.get('name')}: {item.get('status')} ({elapsed} ms), error={error}")


def print_trace(analysis: dict[str, Any]) -> None:
    trace = analysis.get("trace") or []
    print_section("Trace")
    if not trace:
        print("-")
    for item in trace:
        detail = item.get("detail")
        if detail:
            print(
                f"- step={item.get('step')} node={item.get('node')} "
                f"action={item.get('action')} status={item.get('status')} detail={detail}"
            )
        else:
            print(
                f"- step={item.get('step')} node={item.get('node')} "
                f"action={item.get('action')} status={item.get('status')}"
            )


def save_json(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    payload, features_note = build_payload(args)
    body, elapsed_ms = request_agent(args, payload)
    analysis = analysis_from_response(args.endpoint, body)
    answer = answer_from_response(args.endpoint, body, analysis)

    if args.save_json:
        save_json(args.save_json, body)

    print_question_answer(args.question, answer, args.features_file, features_note)
    print_agent_summary(analysis)
    print_risk(analysis)
    print_sources(analysis)
    print_llm(analysis)
    print_limitations_and_tools(analysis)
    if args.trace:
        print_trace(analysis)
    print_section("Request Time")
    print(f"{elapsed_ms:.1f} ms")
    if args.save_json:
        print(f"raw_json: {args.save_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
