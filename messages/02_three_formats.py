# TYPE: Three Ways to Pass Messages
# DESCRIPTION: You can talk to the model in 3 formats — all do the same thing.
# Pick whichever feels easiest for your situation.

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage

model = init_chat_model("gpt-4.1", model_provider="openai")

# ============================================================
# WAY 1: Plain string — simplest, one question only
# Use when: single question, no history, no system rules needed
# ============================================================
response = model.invoke("Write a haiku about spring")
print(response.content)

# ============================================================
# WAY 2: Message objects — most explicit and readable
# Use when: multi-turn conversation, system rules, adding history
# ============================================================
response = model.invoke([
    SystemMessage("You are a poetry expert"),
    HumanMessage("Write a haiku about spring"),
    AIMessage("Cherry blossoms bloom..."),   # past AI reply = memory
    HumanMessage("Now write one about winter"),  # ← AI answers this
])
print(response.content)

# ============================================================
# WAY 3: Dict format — same as Way 2, less imports needed
# Use when: you prefer less typing, coming from OpenAI API background
# ============================================================
response = model.invoke([
    {"role": "system",    "content": "You are a poetry expert"},
    {"role": "user",      "content": "Write a haiku about spring"},
    {"role": "assistant", "content": "Cherry blossoms bloom..."},  # memory
    {"role": "user",      "content": "Now write one about winter"},
])
print(response.content)

# ALL THREE produce the same result — pick what feels cleanest to you
# role: "system"    = SystemMessage
# role: "user"      = HumanMessage
# role: "assistant" = AIMessage