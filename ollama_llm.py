# # One way
# import requests
#
# url = "http://localhost:11434/v1/chat/completions"
#
# data = {
#     "model": "smollm2:135m",
#     "messages": [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Please tell me what you are doing."}
#     ],
#     "stream": False
# }
#
# r = requests.post(url, json=data)
# r.raise_for_status()
# print(r.json()["choices"][0]["message"]["content"])

# Alternative way
import ollama

model = "smollm2:135m"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Please tell me what you are doing."}
]

response = ollama.chat(model=model, messages=messages)

print(response["message"]["content"])