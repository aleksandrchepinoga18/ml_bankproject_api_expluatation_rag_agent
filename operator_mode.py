import requests


print("Режим оператора: введите user_id для получения объяснения")

while True:
    user_id = input("\nuser_id (или 'exit'): ").strip()
    if user_id.lower() == "exit":
        break

    try:
        resp = requests.post(
            "http://localhost:5000/explain",
            json={"user_id": user_id},
            timeout=5,
        )
        data = resp.json()
        if resp.status_code == 200:
            print(f"\n{data.get('explanation', '-')}")
        elif resp.status_code == 404:
            print("Сначала вызовите /predict для этого user_id")
        else:
            print(f"Ошибка: {data.get('error', resp.text)}")
    except Exception as e:
        print(f"Ошибка: {e}")
