📝 Техническое задание: Добавление RAG и режима оператора
1. Контекст
Проект уже содержит рабочие компоненты: API, мониторинг дрейфа, инференс, переобучение.
Цель: добавить объяснимость без LLM.
2. Что нужно добавить
2.1. RAG-компонент (без LLM)
При вызове /predict сохранять SHAP-объяснение в monitoring/explanations/{user_id}.json
Формат:
{
  "user_id": "U123",
  "score": 0.12,
  "decision": "отказ",
  "top_features": [
    {"feature": "risky_tx_count", "value": 28, "impact": 0.35},
    {"feature": "wallet_age", "value": 259200, "impact": 0.25}
  ]
}

Объяснения генерируются динамически при первом /predict на основе:
model.predict_proba
shap.TreeExplainer
Бизнес-правил в коде

2.2. Эндпоинт /explain
Принимает: {"user_id": "U123"}
Возвращает: текстовое объяснение на русском
Правила:
risky_tx_count > 20 → «обнаружено N рисковых транзакций»
wallet_age < 604800 (7 дней) → «кошелёк создан менее недели назад»

Файл: app/api.py

Место: в конец файла, после всех существующих @app.route(...) и функций, но внутри того же файла, где определён Flask-приложение (app = Flask(__name__)).
Что вставить:
@app.route("/explain", methods=["POST"])
def explain():
    user_id = request.json.get("user_id")
    path = f"monitoring/explanations/{user_id}.json"
    if not os.path.exists(path):
        return jsonify({"error": "Объяснение не найдено"}), 404

    with open(path, "r", encoding="utf-8") as f:
        exp = json.load(f)

    reasons = []
    for feat in exp["top_features"]:
        name, val = feat["feature"], feat["value"]
        if name == "risky_tx_count":
            reasons.append(f"обнаружено {int(val)} рисковых транзакций")
        elif name == "wallet_age":
            days = int(val // 86400)
            reasons.append(f"кошелёк создан всего {days} дней назад")
        # Добавьте другие правила под ваши фичи

    text = "Мы не можем одобрить заявку, так как " + ", ".join(reasons) + "." if reasons else "Решение основано на комплексной оценке."
    return jsonify({"explanation": text})

 Убедитесь, что в начале файла уже есть импорты:
from flask import ..., jsonify
import os
import json


2.3. Режим оператора
Файл: создайте новый файл operator_mode.py в корне проекта

В цикле запрашивает user_id → делает POST к /explain → выводит ответ
Альтернатива curl для поддержки

import requests

print("🔍 Режим оператора: введите user_id для получения объяснения")
while True:
    user_id = input("\nuser_id (или 'exit'): ").strip()
    if user_id.lower() == "exit":
        break
    try:
        resp = requests.post("http://localhost:5000/explain", json={"user_id": user_id}, timeout=5)
        data = resp.json()
        print(f"\n💬 {data.get('explanation', 'Объяснение не найдено')}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")


3. Как сохраняются данные для RAG?
Ничего не предварительно не создаётся
Объяснение генерируется on-the-fly при первом /predict
Кэшируется в JSON для повторных запросов
Источники данных:
Модель (predict_proba)
SHAP-эксплейнер
Жёсткие правила в коде

4. Пошаговые задачи для агента

Задача 1: Обновить train_pipeline.py
Добавить после сохранения модели:

joblib.dump(feature_cols, "models/lightgbm_feature_names.pkl")

Задача 2: Обновить src/inference.py

Что сделать:
Оставить существующие функции load_model() и predict() без изменений
Добавить в конец файла следующие три функции:

import shap
import os
import json
from datetime import datetime

def load_explainer_and_metadata():
    model = load_model()
    explainer = shap.TreeExplainer(model)
    feature_names = joblib.load("models/lightgbm_feature_names.pkl")
    best_threshold = joblib.load("models/lightgbm_best_threshold.pkl")
    return explainer, feature_names, model, best_threshold

def get_top_risk_factors(shap_values, feature_names, features, top_k=2):
    impacts = list(zip(feature_names, features, shap_values[1]))
    risk_factors = sorted(
        [item for item in impacts if item[2] > 0],
        key=lambda x: x[2], reverse=True
    )
    return [{"feature": f, "value": v, "impact": float(i)} for f, v, i in risk_factors[:top_k]]

def save_explanation(user_id, score, top_features, best_threshold, output_dir="monitoring/explanations"):
    decision = "отказ" if score < best_threshold else "одобрение"
    explanation = {
        "user_id": user_id,
        "score": float(score),
        "decision": decision,
        "top_features": top_features,
        "timestamp": datetime.utcnow().isoformat()
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{user_id}.json", "w", encoding="utf-8") as f:
        json.dump(explanation, f, ensure_ascii=False, indent=2)


Оставить старые функции, добавить в конец:

load_explainer_and_metadata()
get_top_risk_factors() — только положительный вклад в класс 1
save_explanation() — сохраняет JSON в monitoring/explanations/

Задача 3: Обновить app/api.py
Импортировать новые функции из inference
Загрузить компоненты при старте
В /predict — вызвать SHAP и сохранить объяснение
Добавить эндпоинт /explain в конец файла

Уточнение для /predict:
1. Получить user_id из request.json["user_id"], затем из wallet_address; если идентификатор не передан — сгенерировать безопасный user_id.
2. Получить признаки из request.json["features"], если поле features передано, иначе использовать текущую логику проекта: признаки лежат на верхнем уровне JSON.
3. Привести признаки к порядку feature_names.
4. Посчитать score = model.predict_proba(X)[:, 1].
5. Посчитать shap_values = explainer.shap_values(X).
6. Получить top_features через get_top_risk_factors().
7. Получить rule_triggers по бизнес-правилам.
8. Сохранить explanation JSON в monitoring/explanations/{user_id}.json.
9. Вернуть прежний ответ /predict без поломки существующего API, дополнительно включая user_id.

Задача 4: Создать operator_mode.py
CLI-скрипт с циклом input() → POST → print

5. Последовательность команд
Установить зависимости: pip install shap requests
Обновить train_pipeline.py
Обновить src/inference.py
Обновить app/api.py
Создать operator_mode.py
Протестировать:

bash
1. Запустить API в терминале 1:

python app/api.py

2. Сначала вызвать /predict, чтобы создался JSON:

curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id":"U123","features":{"wallet_age":259200,"risky_tx_count":28}}'

Важно: в реальном запросе в features нужно передать все признаки модели. Короткий curl выше показывает форму payload.

3. Проверить, что появился файл:

ls monitoring/explanations/

Для Windows PowerShell:

Get-ChildItem monitoring/explanations

4. Проверить /explain:

curl -X POST http://127.0.0.1:5000/explain \
  -H "Content-Type: application/json" \
  -d '{"user_id":"U123"}'

5. Проверить CLI:

python operator_mode.py

6. Правки перед реализацией

   Перед реализацией ТЗ исправь следующее:

1. src/inference.py — добавить импорт:
 import joblib
В блоке новых импортов он отсутствует, но используется в

load_explainer_and_metadata().

2. app/api.py — санитизация user_id в /explain:

import re
if not re.match(r'^[\w\-]+$', user_id):
    return jsonify({"error": "Недопустимый user_id"}), 400

Добавить сразу после user_id = request.json.get("user_id").

3. app/api.py — загрузка компонентов при старте, сразу после импортов:
   explainer, feature_names, model, best_threshold = load_explainer_and_metadata()
4. src/inference.py — совместимость с shap >= 0.42, в get_top_risk_factors() заменить:

# было
shap_values[1]
# стало
vals = shap_values.values[:, 1] if hasattr(shap_values, 'values') else shap_values[1]

И дальше использовать vals вместо shap_values[1].

5. operator_mode.py — обработка 404:

if resp.status_code == 404:
    print("⚠️ Сначала вызовите /predict для этого user_id")
else:
    print(f"\n💬 {data.get('explanation', '—')}")

7. Уточнения и требования к реализации

1. score — это вероятность риска из model.predict_proba(X)[:, 1], то есть вероятность класса 1.
2. Если score >= best_threshold, decision = "отказ".
3. Если score < best_threshold, decision = "одобрение".
4. /predict не должен ломать старый формат ответа, если он уже используется клиентами.
5. user_id обязателен для осмысленного сохранения и последующего поиска объяснения. Для обратной совместимости, если user_id не передан старым клиентом, API генерирует безопасный user_id и возвращает его в ответе.
6. SHAP должен считаться по признакам в том же порядке, что использовался при обучении модели.
7. feature_names загружаются из models/lightgbm_feature_names.pkl.
8. Если файл объяснения отсутствует, /explain возвращает 404.
9. Если explanation найден, /explain возвращает текстовое объяснение на русском.
10. Бизнес-правила должны проверяться по значениям признаков, а не только по top_features.
