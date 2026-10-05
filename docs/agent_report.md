# Agent rehearsal

Дата: 2026-09-30.

Проверка команды 8: real LangGraph execution, local Ollama tool calling, typed Agent state, allow-listed scoring/SHAP/retrieval tools, deterministic fallback and answer validation.

- Agent version: `wallet_risk_agent_v1`.
- Prompt version: `wallet_risk_agent_prompt_v1`.
- Wallet address: `0x01da6f3b20d0540f24d390e28195ad7311516739`.
- Execution mode: `langgraph`.
- LangGraph app: `CompiledStateGraph`.
- Tool selection policy: `local_llm_tool_calling_with_deterministic_fallback`.
- Ollama model: `qwen2.5:7b-instruct`.
- Scenario count: `6`.

## Scenarios

| Scenario | Mode | Intent | Selection | Selected tools | Tool results | LLM status | Sources | Limitations |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| combined_score_shap_documents | langgraph | combined | llm_tool_calling | scoring, shap, retrieval | scoring:ok, shap:ok, retrieval:ok | ok | 3 | synthetic_demo_documents_are_not_independent_evidence |
| address_without_features | langgraph | score | llm_tool_calling | scoring | scoring:ok | ok | 0 | features_required |
| document_only_sources | langgraph | documents | deterministic_fallback | retrieval | retrieval:ok | ok | 3 | synthetic_demo_documents_are_not_independent_evidence, tool_selection_fallback |
| document_only_prompt_injection | langgraph | documents | deterministic_fallback | retrieval | retrieval:ok | ok | 3 | synthetic_demo_documents_are_not_independent_evidence, tool_selection_fallback |
| general_methodology | langgraph | general | llm_tool_calling | retrieval | retrieval:ok | ok | 3 | none |
| no_tool_capability_question | langgraph | unsupported | none | none | none | ok | 0 | unsupported_intent |

## Trace Evidence

LangGraph is the main execution path. After deterministic address validation, local Ollama selects tools through tool calling from the allow-list. Deterministic intent routing is used only as fallback when LLM tool selection is unavailable or empty.

- `combined_score_shap_documents`: parse_request[node]:ok -> llm_tool_decision[llm_tool_decision]:ok -> scoring[tool_call]:ok -> scoring[tool_result]:ok -> shap[tool_call]:ok -> shap[tool_result]:ok -> retrieval[tool_call]:ok -> retrieval[tool_result]:ok -> execute_tools[node]:ok -> build_answer[node]:ok -> validate_answer[node]:ok
- `address_without_features`: parse_request[node]:ok -> llm_tool_decision[llm_tool_decision]:ok -> scoring[tool_call]:ok -> scoring[tool_result]:ok -> execute_tools[node]:ok -> build_answer[node]:ok -> validate_answer[node]:ok
- `document_only_sources`: parse_request[node]:ok -> llm_tool_decision[llm_tool_decision]:ok -> retrieval[tool_call]:ok -> retrieval[tool_result]:ok -> execute_tools[node]:ok -> build_answer[node]:ok -> validate_answer[node]:ok
- `document_only_prompt_injection`: parse_request[node]:ok -> llm_tool_decision[llm_tool_decision]:ok -> retrieval[tool_call]:ok -> retrieval[tool_result]:ok -> execute_tools[node]:ok -> build_answer[node]:ok -> validate_answer[node]:ok
- `general_methodology`: parse_request[node]:ok -> llm_tool_decision[llm_tool_decision]:ok -> retrieval[tool_call]:ok -> retrieval[tool_result]:ok -> execute_tools[node]:ok -> build_answer[node]:ok -> validate_answer[node]:ok
- `no_tool_capability_question`: parse_request[node]:ok -> build_answer[node]:ok -> validate_answer[node]:ok

## Operational Checks

| Check | Result | Detail |
| --- | --- | --- |
| tool_timeout | timeout | tool_timeout |
| duplicate_tool_call | duplicate_blocked | repeated_tool_call |
| max_steps | ok | max_steps_reached |
| llm_fallback_unavailable_model | unavailable | 404 Client Error: Not Found for url: http://localhost:11434/api/generate |
| docker_analyze_real_llm | ok | execution_mode=langgraph; tool_selection_source=llm_tool_calling; selected_tools=retrieval; llm.status=ok |

## Requested Checks

| Check | Result | Evidence |
| --- | --- | --- |
| address_without_features_insufficient_data | ok | risk.status=insufficient_data; score=None; prediction=None; trace=parse_request[node]:ok -> llm_tool_decision[llm_tool_decision]:ok -> scoring[tool_call]:ok -> scoring[tool_result]:ok -> execute_tools[node]:ok -> build_answer[node]:ok -> validate_answer[node]:ok |
| trace_distinguishes_llm_and_fallback | ok | llm_trace=source=llm_tool_calling; selected=scoring,shap,retrieval; llm_status=ok; invalid=none; fallback_trace=source=deterministic_fallback; selected=retrieval; llm_status=ok; invalid=none |
| no_tool_query_without_tool_calls_or_forced_fallback | ok | selection=none; selected_tools=[]; tool_statuses=[]; trace=parse_request[node]:ok -> build_answer[node]:ok -> validate_answer[node]:ok |

## Notes

- Document claims are generated only from returned chunk text and validated against existing chunk IDs.
- Synthetic demo documents are marked via `source_kind=synthetic_demo` and are not treated as independent evidence.
- If local Ollama or the requested model is unavailable, the Agent returns structured tool output with `llm.status=unavailable`.
- Docker `/analyze` with real Ollama exposed a truncated answer at `AGENT_LLM_NUM_PREDICT=192`; answer generation now records Ollama `done_reason`, retries once with a concise prompt, and falls back to structured tool output with `validation_errors=["llm_answer_truncated"]` if the retry is still truncated.
- LangGraph and LangChain dependencies are installed in the current Python environment and pinned in `requirements.txt`.
