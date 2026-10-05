# Scoring contract

Дата: 2026-09-29.

Документ фиксирует контракт после команды 3 из
`TZ_Scoring_RAG_Agent_Production_v2.md`.

## Versioning

Текущие значения при аудите:

- `schema_version`: `scoring_explanation_v1`
- `model_version`: вычисляется при старте API

`model_version` строится как hash от связанного пакета:

- `models/lightgbm_model.pkl`;
- порядок `feature_names`;
- preprocessing description;
- imputation values;
- threshold;
- `schema_version`.

Текущее значение в локальном запуске: `lightgbm_4fc20542d0f4c2cb`.

## `/predict`

`POST /predict` по-прежнему принимает один объект или список объектов с полными
признаками строки. Поддерживается переходный формат:

```json
{
  "user_id": "demo-user",
  "wallet_address": "0x...",
  "row_id": "source-row-1",
  "features": {
    "...": 0
  }
}
```

Ответ содержит:

- `user_id` - внешний пользовательский/клиентский идентификатор;
- `wallet_address` - адрес кошелька, если передан;
- `row_id` - идентификатор строки наблюдения, отдельно от `user_id`;
- `request_id` - основной идентификатор прогноза и explanation JSON;
- `model_version`;
- `schema_version`;
- `prediction`;
- `risk_probability`;
- `threshold`.

Новый explanation сохраняется как
`monitoring/explanations/{request_id}.json`. Файл больше не перезаписывается по
одному `user_id`.

## Explanation JSON

Новый JSON содержит:

- `request_id`;
- `user_id`;
- `wallet_address`;
- `row_id`;
- `model_version`;
- `schema_version`;
- `score`;
- `threshold`;
- `decision`;
- `risk_class`;
- `model_class`;
- `feature_order`;
- `top_features`;
- `rule_triggers`;
- `timestamp`.

Переходное поле `decision` больше не использует кредитные формулировки
`отказ`/`одобрение`. Значения:

- `повышенный_риск`;
- `низкий_риск`.

Машиночитаемое поле:

- `risk_class`: `high_risk` или `lower_risk`.

## `/explain`

Предпочтительный вход:

```json
{"request_id": "..."}
```

Для обратной совместимости `/explain` также принимает:

```json
{"user_id": "..."}
```

Если найден текущий JSON с совпадающими `model_version` и `schema_version`,
возвращается `risk.status = current`.

Если JSON старого формата или версия не совпадает, возвращается
`risk.status = stale`. Такой ответ не считается текущим объяснением и содержит
`current_model_version` / `current_schema_version`.

Текст `/explain` описывает риск кошелька и больше не говорит об одобрении или
отказе заявки.
