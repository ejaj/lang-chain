# TYPE: Update State from a Tool
# DESCRIPTION: Tools can not only READ state but also WRITE to it using Command.
# Return Command(update={...}) instead of a plain string to update state fields.
# This is how a tool "remembers" something for the rest of the conversation.

from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command

# --- Define custom state to store user name ---
class MyState(AgentState):
    user_name: str

# --- Tool that WRITES to state ---
@tool
def set_user_name(new_name: str, runtime: ToolRuntime) -> Command:
    """Save the user's name so we can remember it for the rest of the conversation."""
    return Command(
        update={
            "user_name": new_name,          # ← update the state field
            "messages": [
                ToolMessage(
                    content=f"Got it! I'll remember your name is {new_name}.",
                    tool_call_id=runtime.tool_call_id,  # required — links back to tool call
                )
            ],
        }
    )

# --- Tool that READS the state written above ---
@tool
def get_user_name(runtime: ToolRuntime) -> str:
    """Get the user's name that was saved earlier."""
    name = runtime.state.get("user_name", "unknown")
    return f"Your name is {name}"

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[set_user_name, get_user_name],
    state_schema=MyState,
)

# Turn 1: save name
result = agent.invoke({
    "messages": [{"role": "user", "content": "My name is Alice, please remember it."}],
    "user_name": "",
})
print(result["messages"][-1].content)
# → "Got it! I'll remember your name is Alice."

# Turn 2: read it back
result = agent.invoke({
    "messages": result["messages"] + [{"role": "user", "content": "What is my name?"}],
    "user_name": result["user_name"],  # carry state forward
})
print(result["messages"][-1].content)
# → "Your name is Alice."