# TYPE: Multi-Turn Conversation (Memory)
# DESCRIPTION: Models have NO memory between calls by default.
# To give them memory, pass the FULL conversation history every time.
# Build the history yourself by appending messages after each reply.

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage

model = init_chat_model("gpt-4.1", model_provider="openai")

# --- Start with system rules ---
history = [
    SystemMessage("You are a helpful assistant. Remember everything the user tells you."),
]

# --- Turn 1 ---
history.append(HumanMessage("My name is Alice and I love cats."))
response = model.invoke(history)
history.append(AIMessage(response.content))   # save AI reply to history
print("AI:", response.content)
# → "Nice to meet you Alice! Cats are wonderful!"

# --- Turn 2 ---
history.append(HumanMessage("What do I love?"))
response = model.invoke(history)
history.append(AIMessage(response.content))   # save again
print("AI:", response.content)
# → "You love cats!"   ← AI remembers because history was passed

# --- Turn 3 ---
history.append(HumanMessage("And what is my name?"))
response = model.invoke(history)
history.append(AIMessage(response.content))
print("AI:", response.content)
# → "Your name is Alice!"

# --- What history looks like after 3 turns ---
for msg in history:
    role = type(msg).__name__
    print(f"{role}: {msg.content}")
# SystemMessage: You are a helpful assistant...
# HumanMessage:  My name is Alice and I love cats.
# AIMessage:     Nice to meet you Alice!
# HumanMessage:  What do I love?
# AIMessage:     You love cats!
# HumanMessage:  And what is my name?
# AIMessage:     Your name is Alice!

# RULE: Always pass the full history — the model only knows what you send it