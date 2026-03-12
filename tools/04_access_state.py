# TYPE: Access State (Short-term Memory)
# DESCRIPTION: Tools can read the current conversation state using ToolRuntime.
# State = short-term memory (messages, counters, custom fields).
# Add "runtime: ToolRuntime" to your tool — it is injected automatically.
# The AI NEVER sees the runtime parameter — it is hidden from the tool schema.

from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage
from langchain.agents import create_agent, AgentState

# --- Read from state: get last user message ---
@tool
def get_last_message(runtime: ToolRuntime) -> str:
    """Get the most recent message the user sent."""
    messages = runtime.state["messages"]

    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return f"Last user message: {msg.content}"

    return "No user messages found"

# --- Read from custom state field ---
@tool
def get_user_preference(pref_name: str, runtime: ToolRuntime) -> str:
    """Get a saved user preference by name."""
    prefs = runtime.state.get("user_preferences", {})
    value = prefs.get(pref_name, "not set")
    return f"{pref_name} = {value}"

# --- Define custom state schema ---
class MyState(AgentState):
    user_preferences: dict   # extra field beyond just messages

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[get_last_message, get_user_preference],
    state_schema=MyState,
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What language do I prefer?"}],
    "user_preferences": {"language": "French", "theme": "dark"},
    # ↑ passed into state — tools can read this
})
print(result["messages"][-1].content)
# → "Your preferred language is French"

# NOTE: runtime parameter is INVISIBLE to the AI
# AI only sees: get_user_preference(pref_name: str)
# runtime is secretly injected by LangChain at runtime