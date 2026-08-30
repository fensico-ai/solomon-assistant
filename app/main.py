import json
from pathlib import Path

import ollama


BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_FILE = BASE_DIR / "data" / "chat_history.json"


def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError:
            return []
    return []


def save_history(messages):
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(messages, file, indent=2, ensure_ascii=False)


messages = load_history()

print("Solomon Assistant")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Assistant: Goodbye!")
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    print("\nAssistant: ", end="", flush=True)

    full_response = ""

    stream = ollama.chat(
        model="qwen3:8b",
        messages=messages,
        stream=True
    )

    for chunk in stream:
        text = chunk["message"]["content"]
        print(text, end="", flush=True)
        full_response += text

    print("\n")

    messages.append({
        "role": "assistant",
        "content": full_response
    })

    save_history(messages)