# TYPE: AIMessage
# DESCRIPTION: AIMessage is what the AI replies with.
# You never write it for NEW replies — the model generates it automatically.
# You only write it manually when you want the AI to REMEMBER past replies (history).
# Also contains metadata: token usage, tool calls, reasoning steps.

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage

model = init_chat_model("gpt-4.1", model_provider="openai")

# --- AI generates it automatically ---
response = model.invoke([HumanMessage("What is 2+2?")])
print(type(response))       # <class 'AIMessage'>
print(response.content)     # → "4"

# --- Check metadata inside AIMessage ---
print(response.usage_metadata)
# {
#   "input_tokens": 10,   ← tokens you sent
#   "output_tokens": 5,   ← tokens AI replied with
#   "total_tokens": 15    ← total billed
# }

# --- Manually create AIMessage for conversation history (memory) ---
# Use this when you want the AI to remember what it said before
messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("My name is Alice."),
    AIMessage("Nice to meet you, Alice!"),      # ← manually add past reply
    HumanMessage("What is my name?"),           # ← AI will answer using history
]
response = model.invoke(messages)
print(response.content)   # → "Your name is Alice."

# --- WHY manually create AIMessage? ---
# Because the model has NO memory between calls.
# You must pass the full conversation history every time.
# AIMessage lets you reconstruct that history accurately.