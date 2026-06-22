import requests

CHAT_ID = 6167417325

r = requests.post(
    f"https://api.telegram.org/bot8866238336:AAHKcPy9JyWsyXT6sxWLhWO81HFtxaHiEqo/sendMessage",
    json={
        "chat_id": CHAT_ID,
        "text": "✅ Test từ Kaggle"
    }
)

print(r.json())