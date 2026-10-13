# Docker API Review Smoke 2026-10-13

Scope:

- Checked the already rebuilt Docker API at `http://127.0.0.1:8080`.
- Docker was not rebuilt and the container was not stopped.
- Code, models, corpus, index, and MLflow were not changed.

Raw JSON saved:

- `results/docker_api_health_ready_review_2026-10-13.json`
- `results/docker_api_version_review_2026-10-13.json`
- `results/docker_api_metrics_review_2026-10-13.json`
- `results/demo_agent_documents_review_2026-10-13.json`
- `results/demo_agent_methodology_review_2026-10-13.json`
- `results/demo_agent_scoring_review_2026-10-13.json`
- `results/demo_agent_combined_review_2026-10-13.json`
- `results/demo_agent_multirow_review_2026-10-13.json`

Endpoint checks:

| Check | Result |
| --- | --- |
| `/health/ready` | `ready`; model, scoring adapter, retrieval index, and corpus all `true`. |
| `/version` | `llm_provider=ollama`, `ollama_model=qwen2.5:7b-instruct`, model `lightgbm_89ef90618da4df9d`, schema `scoring_explanation_v1`, corpus `demo_corpus_v1`, index `idx_422d98707f81`. |
| `/metrics` | availability `1.0`, 5xx error rate `0.0`, Agent total `140`, answer replacement rate `0.114286`, source correctness `64/64`. The answer replacement metric is a fallback/replacement count, not evidence of LLM answer quality. |

Scenario checks:

| Scenario | Result | Final answer and fallback status |
| --- | --- | --- |
| documents | `intent=documents`, `risk.status=not_requested`, retrieval `ok`, 3 wallet-linked sources, `llm.status=ok`, `done_reason=stop`, no validation errors. | Final answer came from successful LLM generation, not fallback. It did not report a model score. |
| methodology | `intent=general`, retrieval `ok`, 2 method sources, `llm.status=skipped`, `llm.error=answer_generation_skipped_for_explicit_methodology`. | Final answer is deterministic safe methodology output. This is intentionally distinct from successful LLM generation and is not evidence of LLM answer quality. |
| scoring | `intent=score`, scoring `ok`, score `0.35894730197831315`, threshold `0.34438552759270996`, prediction `1`, relation above threshold. | Original LLM output was rejected with `llm_answer_replaced_for_scored_risk_contradiction`; final answer is deterministic fallback with generic marker `llm_answer_fallback`. |
| combined | `intent=combined`, scoring/SHAP/retrieval all `ok`, score `0.35894730197831315`, threshold `0.34438552759270996`, prediction `1`, 3 wallet-linked sources, 3 top features. | Initial/retry LLM path ended truncated: `llm.status=ok`, `done_reason=length`, markers `llm_answer_retry_after_truncation`, `llm_answer_fallback_after_truncation`, `llm_answer_fallback`; final answer is deterministic structured fallback. |
| multi-row scoring | `intent=explain`, scoring `ok`, aggregated score `0.3615507526626647`, threshold `0.34438552759270996`, prediction `1`, `aggregation.records_count=2`, representative row `review-row-a`. | SHAP first-row explanation was not returned as aggregate explanation: `risk.top_features=[]`, tool `shap=skipped`, error `aggregated_score_shap_not_supported`, marker `shap_skipped_aggregated_score_not_supported`. |

Source and numeric consistency:

- Scoring and combined final fallback answers preserved the structured score, threshold, prediction, and above-threshold relation.
- Documents and combined sources were wallet-linked to `0x01da6f3b20d0540f24d390e28195ad7311516739`; the synthetic note was marked as demo-only/not independent evidence.
- Methodology sources were general method documents: `method_scoring_limits` and `method_retrieval_rules`.
- Multi-row aggregation preserved the aggregate score and did not expose first-row SHAP factors as aggregate factors.

Defect found:

- In `results/demo_agent_multirow_review_2026-10-13.json`, the multi-row final answer includes an LLM explanation sentence claiming: `Given the methodology documents retrieved...`
- The same raw response has `sources=[]`, no retrieval tool status, and selected tools only `scoring, shap`.
- The core multi-row scoring and SHAP-skip contract is correct, but this is a final-answer source consistency defect: the LLM-generated explanatory tail refers to retrieved methodology documents that were not retrieved in that scenario.

Docker/Ollama follow-up needed:

- After fixing the multi-row source-consistency defect, rerun only the multi-row scenario plus `/metrics` against the already rebuilt API or the next rebuilt API, depending on deployment state.

## Source-Consistency Rerun

Repeated on the already rebuilt Docker API. Docker was not rebuilt and code was not changed during this rerun.

Raw JSON saved:

- `results/docker_api_health_ready_source_consistency_2026-10-13.json`
- `results/demo_agent_multirow_source_consistency_2026-10-13.json`
- `results/demo_agent_documents_source_consistency_2026-10-13.json`

Results:

| Scenario | Result | Final answer type |
| --- | --- | --- |
| `/health/ready` | `ready`; model, scoring adapter, retrieval index, and corpus all `true`. | n/a |
| multi-row scoring | `intent=explain`, `risk.status=scored`, aggregate score `0.3615507526626647`, threshold `0.34438552759270996`, prediction `1`, `aggregation.records_count=2`, representative row `review-row-a`, `sources=[]`. SHAP stayed explicit skip: `shap=skipped(aggregated_score_shap_not_supported)`, `risk.top_features=[]`. | Accepted LLM text, not fallback. The final answer did not claim documents were retrieved/found/used, so source-consistency replacement was not triggered and no source-consistency fallback markers were expected in this run. |
| documents | `intent=documents`, retrieval `ok`, 3 wallet-linked sources, `llm.status=ok`, `done_reason=stop`, no validation errors. | Accepted LLM text, not fallback. Real source references were allowed because `sources` were present. |

Source-consistency conclusion:

- The prior multi-row defect shape was not reproduced: with `sources=[]`, the final answer no longer claimed retrieved methodology documents.
- Correct document links were not blocked when retrieval actually returned sources.
- No source-consistency fallback markers appeared because neither rerun required a source-consistency replacement. If a future LLM answer again claims actual retrieved sources while `sources=[]`, the expected markers are `llm_answer_replaced_for_source_consistency` and `llm_answer_fallback`.

## Final Review Docker Rerun

Repeated against the already rebuilt Docker API at `http://127.0.0.1:8080`. Docker was not rebuilt, the container was not stopped, code was not changed, and commit/push were not performed.

Raw JSON saved:

- `results/docker_api_health_ready_final_review_2026-10-13.json`
- `results/demo_agent_russian_explain_final_review_2026-10-13.json`
- `results/demo_agent_russian_explain_final_review_2026-10-13.json.client_context.json`
- `results/demo_agent_documents_final_review_2026-10-13.json`

Results:

| Scenario | Result | Final answer status |
| --- | --- | --- |
| `/health/ready` | `status=ready`; checks: model `true`, scoring adapter `true`, retrieval index `true`, corpus `true`. | n/a |
| Russian explain | `intent=explain`, selected tools `scoring, shap`, `risk.status=scored`, score `0.35894730197831315`, threshold `0.34438552759270996`, prediction `1`, relation above threshold. `llm.status=ok`, `done_reason=stop`, no validation errors. | Accepted LLM text, not safe fallback. The deterministic scored prefix is readable and has no `score=...`, `threshold=...`, `prediction=...`, `model_version`, `schema_version`, or `dictionary_note` key-value noise. Demo feature warning was preserved in the sidecar client context JSON. |
| documents | `intent=documents`, selected tool `retrieval`, `risk.status=not_requested`, retrieval `ok`, 3 wallet-linked sources, `llm.status=ok`, `done_reason=stop`, no validation errors. | Accepted LLM text, not safe fallback. Correct references to returned sources were allowed and no source-consistency fallback was triggered. |

Source and fallback checks:

- Russian explain had `sources=[]` and did not claim retrieved/found/used documents.
- Documents response included `synthetic_note_001/synthetic_note_001::chunk_001`, `derived_wallet_001/derived_wallet_001::chunk_001`, and `derived_wallet_001/derived_wallet_001::chunk_002`, all linked to wallet `0x01da6f3b20d0540f24d390e28195ad7311516739`.
- Both responses had `validation_errors=[]`; neither had `llm_answer_*` fallback markers nor `final_answer_violation`.
- Therefore this rerun exercised successful accepted LLM output, not safe deterministic fallback. No remaining final validation errors were present in these two Docker responses.

Defect found:

- In `results/demo_agent_russian_explain_final_review_2026-10-13.json`, the accepted LLM explanation tail describes SHAP impacts as percentages, e.g. `45.86%`, `25.13%`, and `19.99%`.
- The deterministic prefix is readable and keeps SHAP impacts as model contributions, but the accepted LLM tail violates the desired presentation rule that SHAP impact should not be called a percent or probability.
- This is not evidence of fallback quality: no fallback occurred in this scenario.

## SHAP Units Docker Follow-Up

Recorded current outcome after the SHAP-unit guard was available in the already rebuilt Docker API. Docker was not rebuilt, code was not changed during this documentation update, and commit/push were not performed.

Raw JSON saved:

- `results/demo_agent_explain_shap_units_2026-10-13.json`
- `results/demo_agent_explain_shap_units_2026-10-13.json.client_context.json`

Result:

| Scenario | Result | Final answer status |
| --- | --- | --- |
| Russian explain SHAP units | `intent=explain`, selected tools `scoring, shap`, `risk.status=scored`, score `0.35894730197831315`, threshold `0.34438552759270996`, prediction `1`, relation above threshold. `llm.status=ok`, `llm.error=null`, `done_reason=stop`. | Safe deterministic fallback, not accepted LLM text. Original LLM text was rejected with `llm_answer_replaced_for_scored_risk_contradiction`; limitations also include `llm_answer_fallback` and `tool_selection_coverage_guard`. |

Consistency checks:

- Score and threshold are consistent with the structured response and show above-threshold classification.
- Final answer uses SHAP units as `raw output/margin LightGBM`, not percentages, probabilities, or percentage points.
- `%` is absent from the final answer.
- Demo feature warning is preserved in the neighboring `.client_context.json` file.

Remaining product/readability gap:

- The safe fallback is technically correct, but the user-facing explanation remains too technical.
- Next work should check the feature dictionary and feature meaning, separate plain-language explanation from diagnostics, and investigate the exact rejected LLM text.
- Do not assume `rejected_text` was saved in the raw response. If diagnostics were not enabled before the run, a separate diagnostic reproduction is needed.

Operational status:

- Server deployment: `not_run`.
- Rollback verification: `not_run`.
