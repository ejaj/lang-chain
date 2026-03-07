# TYPE: SystemMessage
# DESCRIPTION: SystemMessage sets the rules and personality of the AI.
# The user never sees it — it's hidden instructions sent before the conversation.
# Always put it FIRST in the messages list. Use it ONCE per conversation.

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage

model = init_chat_model("gpt-4.1", model_provider="openai")

# --- Simple one-line instruction ---
response = model.invoke([
    SystemMessage("You are a helpful coding assistant."),
    HumanMessage("How do I create a REST API?"),
])
print(response.content)

# --- Detailed persona with multiple rules ---
response = model.invoke([
    SystemMessage("""
    You are a senior Python developer with 10 years of experience.
    Always provide working code examples.
    Explain WHY, not just HOW.
    Keep answers under 200 words.
    """),
    HumanMessage("How do I create a REST API?"),
])
print(response.content)

# --- Different personas, same question → different answers ---
child_tutor = SystemMessage("You teach 8-year-olds. Use very simple words and fun examples.")
expert_tutor = SystemMessage("You teach PhD students. Use technical terms and be precise.")

q = [HumanMessage("What is gravity?")]

print(model.invoke([child_tutor]  + q).content)  # simple answer
print(model.invoke([expert_tutor] + q).content)  # technical answer

# KEY POINT: Same question, totally different answer based on SystemMessage
# This is how you customize AI behavior without changing your tools or model