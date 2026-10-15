# User output examples

Дата: 2026-10-15. Примеры ожидаемого отображения, **не результаты нового
запуска**. Числа и статусы взяты из
`results/demo_agent_explain_shap_units_2026-10-13.json`; происхождение входа —
из соседнего `.client_context.json`. Текст соответствует новому резервному
шаблону. `<время>` обозначает фактическое время будущего запроса.

Обновление: scored-presentation gate удалён. Код добавляет точные исходные
числа, класс и сравнение, а также ограничения; LLM пишет свободное пояснение
без обязательных оборотов. Отсутствие чисел/предупреждений в пояснении допустимо.
Ниже показан fallback; принятый LLM-текст занимает место его поясняющего абзаца.
Принимаются имена признаков, технические термины, отрицания и упоминания нуля;
неподтверждённые определения отмечаются в контексте LLM и в `--details`.
Проверки явных противоречий — ограниченные эвристики, не полная проверка
истинности произвольного текста.

## Обычный вывод

```text
== Question ==
Объясни оценку риска кошелька.

== Feature Input ==
origin: Synthetic/demo feature row from the existing project acceptance example. It is for local demonstration only, not real wallet evidence.

== Agent Answer ==
Оценка модели — 0,35894730197831315 при пороге 0,34438552759270996: значение выше порога. Результат классификации — класс повышенного риска.
Показанные положительные SHAP-вклады направлены в сторону класса повышенного риска; это часть объяснения модели. Модельные факторы доступны в подробностях; их предметная интерпретация ограничена документацией исходных данных.
Это модельная оценка, не калиброванная вероятность мошенничества и не решение одобрить или отказать.
Текст LLM заменён резервным ответом; причины доступны с --details.

== Request Time ==
<время> ms
```

Предупреждение `origin` выводится только при наличии `note` в переданном файле.
Вход без такого объявления не получает метку demo/synthetic.
Сообщение о замене выводится только при наличии соответствующих маркеров.

## С --details

Вопрос, объявленное происхождение и ответ остаются теми же. Дополнительно до
времени запроса выводятся следующие разделы. Это сокращённый пример: полный
`structured risk` содержит все поля исходного risk, включая aggregation и
rule_triggers. Raw JSON сохраняется без преобразований.

```text
== Agent Routing ==
request_id: 7b3d55049c6841159078838b96926e4a
state: Original LLM text was rejected; safe deterministic fallback was returned.
intent: explain
execution_mode: langgraph
selected_tools: scoring, shap
original_llm_text_rejected: yes
safe_fallback_returned: yes
replacement_reasons: llm_answer_replaced_for_scored_risk_contradiction
final_answer_validation: passed_after_fallback

== LightGBM Result ==
structured risk:
... полный исходный объект risk ...
risk.status: scored
score: 0.358947
threshold: 0.344386
relation: above
prediction: 1
model_version: lightgbm_89ef90618da4df9d
schema_version: scoring_explanation_v1
agent_version: wallet_risk_agent_v1
prompt_version: wallet_risk_agent_prompt_v1
data_status: partial
reason: -

Selected positive SHAP contributions (not a complete explanation):
SHAP units: raw output/margin LightGBM; not percentages or probabilities
Feature semantics: current dictionary does not verify field definitions, feature units or zero meanings.
- time_since_last_liquidated: value=0, impact=0.458609
- time_since_first_deposit: value=0, impact=0.251331
- total_available_borrows_eth: value=0, impact=0.199978

== Documents And Sources ==
-

== LLM ==
provider: ollama
model: qwen2.5:7b-instruct
status: ok
done_reason: stop
error: -
eval_count: 260
prompt_eval_count: 765

== Limitations, Validation, Tool Statuses ==
limitations:
- llm_answer_fallback
- llm_answer_replaced_for_scored_risk_contradiction
- tool_selection_coverage_guard
validation_errors:
- llm_answer_replaced_for_scored_risk_contradiction
tool_statuses:
- scoring: ok (127.816754 ms), error=-
- shap: ok (70.672642 ms), error=-

== Request Time ==
<время> ms
```

`--trace` независимо добавляет раздел Trace; его можно сочетать с `--details`.
Отображение не восстанавливает потерянный LLM-текст старого запуска.

## Локальная диагностика

По умолчанию выключена. При заданном `AGENT_REJECTED_LLM_DIAGNOSTICS_DIR`
файл отклонения дополнительно содержит `rejection_detail` с `rule`, `field`,
`expected`, `detected`. Например, при неправильном score:

```json
{"rule": "reported_number", "field": "risk.score", "expected": 0.72, "detected": 0.9}
```

Ошибка записи выдаёт операторский warning с request ID, общим reason и типом
ошибки; LLM-текст и текст исключения в обычный лог не попадают. API-схема
не расширяется; точная диагностика остаётся в локальном файле.

Проверки: tests, Ruff, mypy, Docker — `not_run` по указанию пользователя.
