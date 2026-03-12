# TYPE: Read Memory Inside a Tool
# DESCRIPTION: Tools can read the current conversation state (short-term memory)
# using ToolRuntime. This lets tools personalize their behaviour based on
# who is asking, what was said earlier, or any custom state fields.
# The runtime parameter is hidden from the AI — it's injected automatically.

from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime

# --- Custom state with extra fields ---
class CustomState(AgentState):
    user_id: str

# --- Tool reads user_id from state ---
@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """Look up the current user's information."""
    user_id = runtime.state["user_id"]   # read from state

    # pretend database lookup
    users = {
        "user_123": "John Smith, Premium plan",
        "user_456": "Jane Doe, Standard plan",
    }
    return users.get(user_id, "Unknown user")

# --- Tool reads message history from state ---
@tool
def count_messages(runtime: ToolRuntime) -> str:
    """Count how many messages are in this conversation."""
    messages = runtime.state["messages"]
    return f"This conversation has {len(messages)} messages so far."

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[get_user_info, count_messages],
    state_schema=CustomState,
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Look up my user information."}],
    "user_id": "user_123",   # ← passed into state
})
print(result["messages"][-1].content)
# → "Your info: John Smith, Premium plan"

# NOTE: The AI only sees get_user_info() with no parameters.
# It has no idea user_id is being read from state secretly.
# This is safer than exposing user_id as a tool argument.