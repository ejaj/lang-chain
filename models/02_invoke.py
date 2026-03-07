# TYPE: Invoke (Basic Call)
# DESCRIPTION: invoke() sends a message and waits for the FULL reply.
# Use this when you don't need to show progress — just want the final answer.
# Accepts a single string, a list of dicts, or a list of message objects.

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage

model = init_chat_model("gpt-4.1", model_provider="openai")

# --- Option A: single string (simplest) ---
response = model.invoke("Why do parrots have colorful feathers?")
print(response.content)

# --- Option B: dict format (easy multi-turn) ---
response = model.invoke([
    {"role": "system",    "content": "You translate English to French."},
    {"role": "user",      "content": "I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},  # past reply = memory
    {"role": "user",      "content": "I love building apps."},      # ← AI answers this
])
print(response.content)
# → "J'adore créer des applications."

# --- Option C: message objects (same as B, more explicit) ---
response = model.invoke([
    SystemMessage("You translate English to French."),
    HumanMessage("I love programming."),
    AIMessage("J'adore la programmation."),   # past reply = memory
    HumanMessage("I love building apps."),    # ← AI answers this
])
print(response.content)
# → "J'adore créer des applications."

# NOTE: Options B and C produce identical results — use whichever feels cleaner