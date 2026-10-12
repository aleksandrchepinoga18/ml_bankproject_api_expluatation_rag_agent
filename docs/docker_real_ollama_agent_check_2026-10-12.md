# Docker API Real Ollama Agent Check

Date: 2026-10-12.

Scope: checked the already rebuilt Docker API at `http://127.0.0.1:8080` with real Ollama. No rebuild, code change, model change, MLflow change, commit, or push was performed.

## API Checks

- `/health/ready`: `ready`; model, scoring adapter, retrieval index, and corpus checks were `true`.
- `/version`: `llm_provider=ollama`, `ollama_model=qwen2.5:7b-instruct`, `model_version=lightgbm_89ef90618da4df9d`, `corpus_version=demo_corpus_v1`, `index_version=idx_422d98707f81`.
- `/metrics`: availability `1.0`, 5xx error rate `0.0`, Agent answer replacement rate `0.109375`, source correctness `59/59`.

Raw JSON:

- `results/docker_api_health_ready_real_ollama_2026-10-12.json`
- `results/docker_api_version_real_ollama_2026-10-12.json`
- `results/docker_api_metrics_real_ollama_2026-10-12.json`

## Demo Scenarios

| Scenario | Raw JSON | Result |
| --- | --- | --- |
| methodology | `results/demo_agent_methodology_real_ollama_2026-10-12.json` | Partial content issue: metadata and tools are correct, but answer text has one wrong sentence. |
| scoring | `results/demo_agent_scoring_real_ollama_2026-10-12.json` | Passed safe-answer behavior after rejecting original LLM text. |
| combined | `results/demo_agent_combined_real_ollama_2026-10-12.json` | Passed safe-answer behavior after rejecting original LLM text. |

## Methodology

- `intent=general`, `execution_mode=langgraph`, selected tool `retrieval`.
- `risk.status=not_requested`, `data_status=ok`.
- Sources were methodology documents only: `method_scoring_limits` and `method_retrieval_rules`; no scoring tool was run.
- The answer explains the methodology documents and does not claim `insufficient_data`.
- New content defect found: the first sentence says, "The wallet risk scoring methodology is not requested in this case", even though the user explicitly requested methodology. This is not an insufficient-data regression, but it is a factual inconsistency in the natural-language answer.
- No fallback marker was emitted because the current guard accepted the answer: `limitations=["tool_selection_intent_guard"]`, `validation_errors=[]`, trace ended with `validate_answer:ok`.

Important: `llm.status=ok` and `validate_answer:ok` only mean the request completed and structural checks passed; they do not prove every natural-language sentence is correct.

## Scoring

- `intent=score`, selected tool `scoring`.
- Structured risk: `score=0.35894730197831315`, `threshold=0.34438552759270996`, `prediction=1`, relation `score is above threshold`.
- Final safe answer reports `score=0.358947`, `threshold=0.344386`, `prediction=1`, and `relation=score is above threshold`.
- Original LLM text was rejected: `limitations` include `llm_answer_fallback` and `llm_answer_replaced_for_scored_risk_contradiction`.
- `validation_errors=["llm_answer_replaced_for_scored_risk_contradiction"]`, `data_status=partial`.
- Trace records the replacement at `build_answer`: `llm_answer_replaced:llm_answer_replaced_for_scored_risk_contradiction`.
- No retrieval/synthetic document was named as the basis for the model score.
- No unsupported claims such as "30 risky transactions", "threshold 20", or "wallet age 10 days" appeared in the final answer.

## Combined

- `intent=combined`, selected tools `scoring`, `shap`, and `retrieval`.
- Structured risk matches scoring: `score=0.35894730197831315`, `threshold=0.34438552759270996`, `prediction=1`, relation `score is above threshold`.
- Final safe answer reports the same score, threshold, prediction, relation, top features, and linked document refs.
- Original LLM text was rejected: `limitations` include `llm_answer_fallback` and `llm_answer_replaced_for_scored_risk_contradiction`.
- `validation_errors=["llm_answer_replaced_for_scored_risk_contradiction"]`, `data_status=partial`.
- Trace records the replacement at `build_answer`: `llm_answer_replaced:llm_answer_replaced_for_scored_risk_contradiction`.
- `synthetic_note_001` is present only as a retrieved synthetic demo source; the output and limitations mark synthetic/demo material as not independent evidence.
- The final answer does not claim the LightGBM score is based on the synthetic note or retrieval documents.
- No unsupported rule/fact claims such as "30 risky transactions", "threshold 20", or "wallet age 10 days" appeared in the final answer.

## Remaining Limitation

The current Docker/Ollama check confirms safe deterministic replacement for scored contradictions, but methodology natural-language validation still allows a contradictory sentence that says methodology was not requested. This should be fixed in a later code change; no code was changed during this check.

## Methodology Fix Rerun

After the methodology-only fix was applied and the Docker API had already been rebuilt by the operator, the explicit methodology scenario was rerun without another rebuild.

Raw JSON:

- `results/docker_api_health_ready_after_methodology_fix_2026-10-12.json`
- `results/demo_agent_methodology_after_fix_real_ollama_2026-10-12.json`

Result:

- `/health/ready`: `ready`; model, scoring adapter, retrieval index, and corpus checks were `true`.
- Scenario: `Explain the wallet risk scoring methodology and cite relevant document sources.`
- `intent=general`, `execution_mode=langgraph`, `data_status=ok`, `risk.status=not_requested`.
- Selected tools: `retrieval` only; scoring did not run.
- Sources: `method_scoring_limits/method_scoring_limits::chunk_001` and `method_retrieval_rules/method_retrieval_rules::chunk_001`.
- Final answer explains both found `method_*` documents with document/chunk references.
- The final answer does not say methodology was not requested and does not claim `insufficient_data`.
- Answer generation was skipped truthfully: `llm.provider=not_called`, `llm.model=none`, `llm.status=skipped`, `llm.error=answer_generation_skipped_for_explicit_methodology`.
- Marker present: `llm_answer_skipped_for_explicit_methodology`.
- Tool-selection trace is preserved: `llm_tool_decision` has `source=llm_tool_calling`, `selected=retrieval`, and `intent_removed=scoring`.
- Build-answer trace shows deterministic rendering: `action=deterministic_methodology_answer`, `detail=llm_generation_skipped_for_explicit_methodology`.
- `validation_errors=[]`.

This rerun resolves the methodology content defect recorded above.
