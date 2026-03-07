# TYPE: Memory / Custom State
# The agent remembers conversation history automatically.
# You can also add your own extra memory fields.

from langchain.agents import create_agent, AgentState
from langchain.tools import tool
from typing import TypedDict

# --- Define extra fields to remember ---
class MyState(AgentState):
    user_name: str         # remember who we're talking to
    preferred_language: str  # remember their language preference

@tool
def greet(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[greet],
    state_schema=MyState,
)

# Pass initial memory values when invoking
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's my name?"}],
    "user_name": "Alice",
    "preferred_language": "English",
})
# Agent can now read user_name from state and answer: "Your name is Alice."

# --- Multi-turn: history is automatic ---
result2 = agent.invoke({
    "messages": [
        {"role": "user",      "content": "My favourite colour is blue."},
        {"role": "assistant", "content": "Got it! Blue is a great colour."},
        {"role": "user",      "content": "What's my favourite colour?"},
    ],
    "user_name": "Alice",
    "preferred_language": "English",
})
# → "Your favourite colour is blue."