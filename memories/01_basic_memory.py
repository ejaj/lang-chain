# TYPE: Basic Short-Term Memory (Checkpointer)
# DESCRIPTION: By default agents have NO memory between invoke() calls.
# Add a checkpointer to give the agent memory within a conversation thread.
# thread_id groups messages together — same thread_id = same conversation.
# InMemorySaver = memory lives in RAM (lost when program stops).
# Use for: development, testing, simple apps.

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

@tool
def get_time() -> str:
    """Get the current time."""
    return "It is 3:00 PM."

# --- Add checkpointer → agent now remembers the conversation ---
agent = create_agent(
    "openai:gpt-4.1",
    tools=[get_time],
    checkpointer=InMemorySaver(),   # ← this enables memory
)

config = {"configurable": {"thread_id": "conversation-1"}}  # ← ties messages together

# Turn 1
agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    config,   # ← always pass same config for same conversation
)

# Turn 2 — agent remembers "Bob" from turn 1
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]},
    config,   # ← same thread_id = agent remembers
)
print(result["messages"][-1].content)
# → "Your name is Bob!"

# Different thread = fresh conversation, no memory of Bob
config2 = {"configurable": {"thread_id": "conversation-2"}}
result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]},
    config2,  # ← different thread_id
)
print(result2["messages"][-1].content)
# → "I don't know your name. Could you tell me?"

# RULE:
# Same thread_id   → agent remembers everything from that conversation
# Different thread → fresh start, no memory