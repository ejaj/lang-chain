# api_tool_example.py
# pip install ollama requests
# ollama serve
# ollama pull smollm2:135m
import json
import requests
import ollama

def get_json(url: str) -> str:
    """Fetch JSON from an HTTP API and return as text."""
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.text  # keep as text for simplicity

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_json",
            "description": "GET a URL and return the response body (JSON as text).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"}
                },
                "required": ["url"]
            }
        }
    }
]

messages = [
    {"role": "system", "content": "You are helpful. If user asks for live data, call the tool."},
    {"role": "user", "content": "Call this API and tell me what keys exist: http://localhost:11434/api/tags"}
]

# 1) First call: model decides whether to use tool
resp1 = ollama.chat(model="smollm2:135m", messages=messages, tools=tools)
assistant_msg = resp1["message"]

# 2) If tool requested, run it and send tool result back
if assistant_msg.get("tool_calls"):
    messages.append(assistant_msg)

    for call in assistant_msg["tool_calls"]:
        args = call["arguments"]
        if isinstance(args, str):
            args = json.loads(args)

        if call["name"] == "get_json":
            tool_result = get_json(args["url"])
        else:
            tool_result = "Unknown tool"

        messages.append({
            "role": "tool",
            "tool_name": call["name"],
            "content": tool_result
        })

# 3) Second call: model reads tool result and answers in normal text
resp2 = ollama.chat(model="smollm2:135m", messages=messages)
print(resp2["message"]["content"])
