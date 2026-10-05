# Baseline audit

Дата аудита: 2026-09-29.

Основание: команда 1 из `TZ_Scoring_RAG_Agent_Production_v2.md`.

Цель аудита: зафиксировать фактическое состояние текущего проекта скоринга до
добавления RAG/Agent/production-слоя. На этом этапе RAG и Agent не
реализовывались.

## Проверенный срез репозитория

- Git commit: `b230ad9`.
- Рабочее дерево на момент аудита уже было изменено: `git status --short`
  показывал изменения в `README.md`, `app/api.py`, `src/*`,
  `monitoring/*`, `requirements.txt`, `results/*`, а также untracked
  `project_review/`, `project_review.zip`, `TZ_Scoring_RAG_Agent_Production_v2.md`,
  `RAG_IMPLEMENTATION_PLAN.md`, `operator_mode.py`, `monitoring/logs/`,
  `monitoring/explanations/`, `monitoring/reference/` и другие файлы.
- Проверенные файлы по команде 1:
  - `train_pipeline.py`
  - `app/api.py`
  - `src/data_preparation.py`
  - `src/evaluate.py`
  - `src/inference.py`
  - `src/train.py`
  - `monitoring/check_data_drift.py`
  - `monitoring/check_model_quality.py`
  - `monitoring/check_score_drift.py`
  - `monitoring/retrain_if_needed.py`
  - `monitoring/simulate_labels.py`
  - `monitoring/log_predictions.py`
  - `models/`
  - `results/`
  - `data/dataset.parquet`

## Данные

В текущей рабочей папке есть исходный файл `data/dataset.parquet`.

Фактическая сводка файла:

- строк: `442961`
- столбцов: `78`
- `wallet_address`: есть
- `target`: есть
- уникальных `wallet_address`: `74476`
- распределение `target`: `0 = 276761`, `1 = 166200`
- SHA256 `data/dataset.parquet`:
  `7BCBE3A162BE9AC2D1F7EE89A0241875E200ECA717B728A52F009AD9851D3DC4`

Первые признаки в файле включают `borrow_block_number`, `borrow_timestamp`,
`wallet_address`, `first_tx_timestamp`, `last_tx_timestamp`, `wallet_age`,
`incoming_tx_count`, `outgoing_tx_count`, `risky_tx_count` и другие признаки
транзакционной/рыночной истории.

Ограничение: в репозитории пока нет отдельного data manifest с происхождением
датасета, версией источника, временем среза, правилами доступности признаков и
проверкой временной утечки. Сам файл для повторного обучения локально есть, но
его происхождение и внешний источник не задокументированы.

## Unit of Prediction и Split

Текущий unit of prediction - одна строка наблюдения кошелька. У одного
`wallet_address` может быть несколько строк.

`src/data_preparation.py` делит данные через `split_data(..., group_col="wallet_address")`:

- группировка выполняется по `wallet_address`
- стратификация выполняется на уровне кошелька через `max(target)`
- строки кошельков из разных split не пересекаются

Сохраненный `models/lightgbm_preprocessing.pkl` содержит следующий
`split_summary`:

- train: `226306` строк, `37238` уникальных кошельков
- validation: `110570` строк, `18619` уникальных кошельков
- test: `106085` строк, `18619` уникальных кошельков
- пересечения кошельков train/validation, train/test, validation/test: `0`

Важно: split по кошелькам не доказывает отсутствие временной утечки. В текущем
аудите не найден документированный смысл времени среза и доступности каждого
признака на момент прогноза.

## Training Pipeline

`train_pipeline.py` выполняет следующий поток:

1. Загружает `data/dataset.parquet`.
2. Удаляет заранее заданный список высококоррелированных признаков.
3. Запускает EDA.
4. Делит данные на train/validation/test по `wallet_address`.
5. Формирует признаки как все столбцы, кроме `target` и `wallet_address`.
6. Заменяет `inf/-inf` на `NaN`; медианы для заполнения fit-ятся только на train.
7. Обучает LightGBM через `src/train.py`.
8. Сохраняет модель, список признаков, параметры, preprocessing и порог.
9. Считает SHAP summary plots и top-50 risky wallets.
10. Сохраняет reference features/scores для мониторинга.

`src/train.py`:

- использует `LGBMClassifier`
- стартует с `DEFAULT_PARAMS`
- дополнительно перебирает `ParameterSampler`
- выбирает лучшую модель по validation ROC-AUC
- сохраняет `models/lightgbm_model.pkl` и `models/lightgbm_param_search.pkl`

Сохраненный поиск параметров:

- selection metric: `validation_roc_auc`
- best validation ROC-AUC: `0.8752606766052883`
- кандидатов: `9`

## Сохраненные ML-артефакты

В `models/` есть:

- `lightgbm_model.pkl`
- `lightgbm_feature_names.pkl`
- `lightgbm_best_threshold.pkl`
- `lightgbm_best_params.pkl`
- `lightgbm_param_search.pkl`
- `lightgbm_preprocessing.pkl`

Фактическое состояние:

- feature count: `65`
- threshold: `0.3578026090255801`
- preprocessing keys: `feature_names`, `imputation_values`, `preprocessing`,
  `split_summary`
- imputation values count: `0`

Пока нет единого явного `model_version`, который связывает модель, порядок
признаков, preprocessing, threshold и версию схемы.

## Метрики и порог

`src/evaluate.py` выбирает operating threshold только на validation по F1, затем
один раз оценивает test с этим фиксированным порогом.

Сохраненный `results/LightGBM_metrics.json`:

- threshold source: `validation_f1`
- best threshold: `0.3578026090255801`
- validation F1 at threshold: `0.7249339377759967`
- train ROC-AUC: `0.9567929500257917`
- validation ROC-AUC: `0.8752606766052883`
- test ROC-AUC: `0.8722954225342738`
- test accuracy: `0.8006692746382618`
- class 1 test precision/recall/F1:
  `0.7226852205826133 / 0.7342119572253437 / 0.728402990058825`

Эти метрики относятся к строкам наблюдений в сохраненном запуске. Их нельзя
выдавать за качество решения по целому кошельку, качество RAG/Agent или
калиброванную вероятность мошенничества в реальном мире.

## Top-50 Risky Wallets

`src/evaluate.py::get_risky_wallets` считает вероятность для каждой строки, затем
агрегирует кошелек как строку с максимальной `probability`.

`results/LightGBM_top50_risky_wallets.csv`:

- строк: `50`
- уникальных кошельков: `50`
- столбцы: `wallet_address`, `probability`, `records_count`, `source_row_index`
- min/max probability в top-50: `0.9984649174115724` /
  `0.9995443150686776`
- min/max `records_count`: `57` / `2759`

Ограничение: правило `max(row_probability)` может отдавать преимущество адресам
с большим числом наблюдений. Альтернативные правила агрегации пока не
сравнивались.

## Текущий Inference/API Contract

Текущий сервис - Flask в `app/api.py`.

Безопасная проверка импорта API показала маршруты:

- `/predict`
- `/explain`
- `/static/<path:filename>`

`/predict`:

- принимает JSON-объект или список объектов
- поддерживает вложенный формат `{"features": {...}}`
- ожидает признаки строки, а не один `wallet_address`
- `wallet_address` и `user_id` извлекаются из верхнего уровня или из `features`
- если `user_id` не передан, используется `wallet_address` или сгенерированный
  технический идентификатор
- строит `X` через `src/inference.py::build_model_input`
- отсутствующие признаки добавляются как `NaN`
- заполнение пропусков берется из `lightgbm_preprocessing.pkl`
- возвращает `user_id`, `wallet_address`, `prediction`, `risk_probability`
- пишет prediction log и JSON explanation как побочный эффект

`build_model_input` не возвращает `insufficient_data` при запросе только по
адресу. Он формирует строку признаков с пропусками, если признаки не переданы.
Feature store или проверенный lookup признаков по одному адресу сейчас не
реализованы.

`/explain`:

- принимает `user_id`
- читает `monitoring/explanations/{user_id}.json`
- возвращает `user_id`, `wallet_address`, `decision`, `score`, `threshold`,
  `model_class`, `explanation`
- не проверяет `model_version`, потому что текущие explanation JSON его не
  содержат

## SHAP и объяснения

`src/inference.py` строит `shap.TreeExplainer(model)` и сохраняет top positive
SHAP factors через `get_top_risk_factors`.

Текущий `save_explanation` сохраняет:

- `user_id`
- `wallet_address`
- `request_id`
- `score`
- `threshold`
- `decision`
- `model_class`
- `feature_order`
- `top_features`
- `rule_triggers`
- `timestamp`

Предметная ошибка: `decision` сейчас хранит значения `отказ` / `одобрение`, а
`/explain` формирует текст вроде `Мы не можем одобрить заявку...` и `Заявка
может быть одобрена...`. Это не соответствует домену риска криптокошельков.

Старые explanation JSON неоднородны:

- `OLD_FORMAT_CHECK.json` содержит только `user_id`, `score`, `decision`,
  `top_features`
- ряд старых файлов не содержит `threshold`, `request_id`, `wallet_address`
- часть новых файлов содержит `threshold = 0.35360315435065043`, что не совпадает
  с текущим сохраненным threshold `0.3578026090255801`

Следствие: старые explanation JSON нельзя выдавать за текущие объяснения без
проверки версии/порога и статуса `stale`.

## Мониторинг

Текущий мониторинг диагностический:

- `monitoring/log_predictions.py` пишет JSONL с `timestamp`, `model_version`,
  `score`, `features`
- значение `model_usage` по умолчанию: `lightgbm_v1`
- `monitoring/check_score_drift.py` сравнивает текущие score logs с
  `monitoring/reference/reference_scores.parquet`
- `monitoring/check_data_drift.py` сравнивает feature logs с
  `monitoring/reference/reference_features.parquet`
- PSI считается по quantile bins reference; результат помечается как
  `verified: False`
- `monitoring/retrain_if_needed.py` не запускает переобучение автоматически,
  а только возвращает `retrain_candidate` и `auto_retrain_started: False`

Reference-файлы есть:

- `monitoring/reference/reference_features.parquet`
- `monitoring/reference/reference_scores.parquet`
- CSV-дубликаты тех же reference данных

`monitoring/simulate_labels.py` создает синтетические метки с
`label_source = synthetic_inverse_score` и предупреждением, что они только для
demo/plumbing. Текущий `monitoring/check_model_quality.py` требует доверенный
`label_source` (`observed`, `verified`, `production`) и отказывается считать
качество на непроверенных метках.

Фактический `monitoring/logs/predictions_with_labels.csv` на момент аудита
содержит 3 строки и не содержит `label_source`, поэтому он не является
доказательством качества модели.

## Источник признаков по адресу

В текущем API нет feature store и нет безопасного сервиса, который по одному
`wallet_address` достает актуальную строку признаков с проверенным временем
среза.

Локальный `data/dataset.parquet` содержит строки с `wallet_address`, поэтому
технически может использоваться для анализа и воспроизводимого обучения. Но
использовать test-строки как скрытый lookup для production/demo-запросов нельзя
без явного контракта, версии, происхождения, времени доступности и защиты от
утечки.

## Датасет для повторного обучения

Локальный датасет для повторного запуска пайплайна есть:

- `data/dataset.parquet`
- `442961` строк
- `78` столбцов
- `74476` уникальных кошельков

Ограничения:

- нет data manifest
- нет documented provenance
- нет сохраненных хешей split-групп train/validation/test отдельно от
  `split_summary`
- нет проверки временной доступности признаков
- нет подтвержденных новых production labels для оценки качества после запуска

## Безопасные проверки, выполненные в рамках аудита

Выполнены только проверки чтения/импорта и статуса контрактов:

- `git status --short` - зафиксировано уже грязное рабочее дерево
- `rg --files` - зафиксирована структура проекта
- чтение файлов из команды 1
- чтение `models/*.pkl` через `joblib.load`
- чтение `data/dataset.parquet` через `pandas.read_parquet`
- чтение `results/LightGBM_metrics.json`
- импорт Flask API и просмотр `app.url_map`
- `python -m pytest -q`

Результат `pytest`:

```text
no tests ran in 0.05s
```

Команда завершилась с ненулевым кодом, потому что тесты не найдены. Отдельной
папки `tests/` и конфигурации pytest в корне проекта не обнаружено.

## Основные расхождения и риски перед следующими командами

1. Нет единого `model_version` для модели, feature order, preprocessing,
   threshold и версии схемы.
2. `/predict` принимает строки признаков, но не умеет корректно отвечать
   `insufficient_data` на запрос только по адресу.
3. Нет feature store или другого проверенного источника признаков по адресу.
4. Старые explanation JSON не имеют единого контракта и могут содержать старый
   threshold.
5. `/explain` использует терминологию кредитной заявки (`одобрение`/`отказ`),
   которая не соответствует задаче риска криптокошельков.
6. Метрики сохраненного запуска относятся к строкам наблюдений, а не к
   агрегированному решению по кошельку.
7. Top-50 агрегируется через `max(row_probability)`; влияние `records_count` на
   рейтинг пока не исследовано.
8. Временная доступность признаков и возможная temporal leakage пока не
   проверены.
9. Автотесты в текущем проекте не обнаружены.
10. RAG, Agent, FastAPI, Qdrant, MLflow registry и CI/CD в рамках команды 1 не
    реализовывались и не должны считаться текущей частью baseline.
