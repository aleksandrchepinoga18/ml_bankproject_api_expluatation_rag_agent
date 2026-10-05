# Agent composite answer audit

Command 8a audit: composite questions, per-part coverage, chunk-level grounding, partial responses, and stopped bad answers.

- Ollama model: `qwen2.5:7b-instruct`.
- Wallet address: `0x01da6f3b20d0540f24d390e28195ad7311516739`.
- Scenario count: `3`.

## Composite Scenarios

| Scenario | Result | Intent | Selection | Selected tools | Risk | Sources | Validation errors |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| wallet_score_factors_documents_boundary | ok | combined | llm_tool_calling | scoring, shap, retrieval | scored | 3 | none |
| general_methodology_multi_fragment | ok | general | llm_tool_calling | retrieval | not_requested | 3 | none |
| missing_features_plus_documents | ok | combined | llm_tool_calling | scoring, retrieval | insufficient_data | 3 | none |

## Per-Part Coverage

### wallet_score_factors_documents_boundary
- `risk_score`: `ok`
- `model_factors`: `ok`
- `document_sources`: `ok`
- `source_kind_boundary`: `ok`
- `document_claims_supported_by_chunk_text`: `ok`
- trace: `parse_request[node]:ok -> llm_tool_decision[llm_tool_decision]:ok -> scoring[tool_call]:ok -> scoring[tool_result]:ok -> shap[tool_call]:ok -> shap[tool_result]:ok -> retrieval[tool_call]:ok -> retrieval[tool_result]:ok -> execute_tools[node]:ok -> build_answer[node]:ok -> validate_answer[node]:ok`

### general_methodology_multi_fragment
- `retrieval_rules_chunk`: `ok`
- `scoring_limits_chunk`: `ok`
- `feature_dictionary_chunk`: `ok`
- `document_claims_supported_by_chunk_text`: `ok`
- trace: `parse_request[node]:ok -> llm_tool_decision[llm_tool_decision]:ok -> retrieval[tool_call]:ok -> retrieval[tool_result]:ok -> execute_tools[node]:ok -> build_answer[node]:ok -> validate_answer[node]:ok`

### missing_features_plus_documents
- `risk_reports_missing_features`: `ok`
- `model_factors_missing_data_is_explicit`: `ok`
- `document_sources`: `ok`
- `document_claims_supported_by_chunk_text`: `ok`
- trace: `parse_request[node]:ok -> llm_tool_decision[llm_tool_decision]:ok -> scoring[tool_call]:ok -> scoring[tool_result]:ok -> retrieval[tool_call]:ok -> retrieval[tool_result]:ok -> execute_tools[node]:ok -> build_answer[node]:ok -> validate_answer[node]:ok`

## Chunk Grounding

| Scenario | Chunk | Exists | Supported by text | Source kind |
| --- | --- | --- | --- | --- |
| wallet_score_factors_documents_boundary | synthetic_note_001::chunk_001 | True | True | synthetic_demo |
| wallet_score_factors_documents_boundary | derived_wallet_001::chunk_001 | True | True | derived_data |
| wallet_score_factors_documents_boundary | derived_wallet_001::chunk_002 | True | True | derived_data |
| general_methodology_multi_fragment | method_feature_dictionary::chunk_001 | True | True | derived_data |
| general_methodology_multi_fragment | method_scoring_limits::chunk_001 | True | True | derived_data |
| general_methodology_multi_fragment | method_retrieval_rules::chunk_001 | True | True | derived_data |
| missing_features_plus_documents | synthetic_note_001::chunk_001 | True | True | synthetic_demo |
| missing_features_plus_documents | derived_wallet_001::chunk_001 | True | True | derived_data |
| missing_features_plus_documents | derived_wallet_001::chunk_002 | True | True | derived_data |

## Failure And Fallback Checks

| Check | Result | Detail |
| --- | --- | --- |
| retrieval_error_partial_structured_response | ok | data_status=partial; llm=ok; trace=parse_request[node]:ok -> llm_tool_decision[llm_tool_decision]:ok -> retrieval[tool_call]:ok -> retrieval[tool_result]:error -> execute_tools[node]:ok -> build_answer[node]:ok -> validate_answer[node]:ok |
| llm_unavailable_returns_tool_output | ok | data_status=ok; llm=unavailable; trace=parse_request[node]:ok -> llm_tool_decision[llm_tool_decision]:ok -> scoring[tool_call]:ok -> scoring[tool_result]:ok -> shap[tool_call]:ok -> shap[tool_result]:ok -> retrieval[tool_call]:ok -> retrieval[tool_result]:ok -> execute_tools[node]:ok -> build_answer[node]:ok -> validate_answer[node]:ok |

## Stopped Bad Answers

| Example | Result | Validation errors |
| --- | --- | --- |
| existing_chunk_unsupported_claim | blocked | unsupported_claim:guard_demo::chunk_001 |
| forbidden_answer_phrase | blocked | forbidden_answer_phrase:proven fraud |
