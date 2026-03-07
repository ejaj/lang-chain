# TYPE: Basic Message Usage
# DESCRIPTION: Messages are how you talk to the model.
# Every conversation is just a list of messages sent to the model.
# Three main types: SystemMessage (rules), HumanMessage (you), AIMessage (AI reply)

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model("gpt-4.1", model_provider="openai")

# --- Build a message list ---
messages = [
    SystemMessage("You are a helpful assistant."),  # rules for AI
    HumanMessage("Hello, how are you?"),            # your message
]

# --- Send to model → get AIMessage back ---
response = model.invoke(messages)

print(type(response))     # <class 'AIMessage'>
print(response.content)   # → "I'm doing great! How can I help you?"

# REMEMBER:
# SystemMessage  → hidden rules, always first
# HumanMessage   → what you say
# AIMessage      → what AI replies (returned automatically, never write it for new replies)