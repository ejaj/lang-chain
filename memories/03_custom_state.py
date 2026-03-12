# TYPE: Custom State Schema
# DESCRIPTION: By default the agent only remembers messages.
# Extend AgentState to remember extra things like user_id, preferences, scores.
# Pass the custom state via state_schema in create_agent.
# Pass the values via invoke() alongside messages.

from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

# --- Define extra fields to remember ---
class CustomAgentState(AgentState):
    user_id: str           # who is logged in
    preferences: dict      # their saved settings
    message_count: int     # how many messages sent

# --- Tool that reads custom state ---
@tool
def get_my_preferences(runtime: ToolRuntime) -> str:
    """Get the current user's saved preferences."""
    prefs = runtime.state.get("preferences", {})
    user_id = runtime.state.get("user_id", "unknown")
    return f"User {user_id} preferences: {prefs}"

agent = create_agent(
    "openai:gpt-4.1",
    tools=[get_my_preferences],
    state_schema=CustomAgentState,   # ← tell agent about extra fields
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "1"}}

# Pass extra state fields in invoke()
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "What are my preferences?"}],
        "user_id": "user_123",                              # ← extra field
        "preferences": {"theme": "dark", "language": "EN"},  # ← extra field
        "message_count": 1,                                 # ← extra field
    },
    config,
)
print(result["messages"][-1].content)
# → "User user_123 preferences: theme=dark, language=EN"

# All custom fields are also saved in the checkpoint
# and available in the next invoke() for the same thread_id