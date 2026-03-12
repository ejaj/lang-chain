# TYPE: Write Memory from a Tool (Command)
# DESCRIPTION: Tools can also WRITE to memory using Command(update={...}).
# One tool saves data → next tool reads it from state.
# This is how tools pass information to each other during a conversation.
# tool_call_id is required when including a ToolMessage in the update.

from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel

# --- Custom state and context ---
class CustomState(AgentState):
    user_name: str   # extra field to store looked-up name

class CustomContext(BaseModel):
    user_id: str     # who is logged in — passed at invoke time

# --- Tool 1: looks up name and SAVES it to state ---
@tool
def update_user_info(runtime: ToolRuntime[CustomContext, CustomState]) -> Command:
    """Look up user info from database and save name to state."""
    user_id = runtime.context.user_id
    name = "John Smith" if user_id == "user_123" else "Unknown user"

    return Command(update={
        "user_name": name,   # ← writes to state
        "messages": [
            ToolMessage(
                content="Successfully looked up user information.",
                tool_call_id=runtime.tool_call_id,
            )
        ],
    })

# --- Tool 2: reads the name that Tool 1 saved ---
@tool
def greet(runtime: ToolRuntime[CustomContext, CustomState]) -> str | Command:
    """Greet the user by name."""
    user_name = runtime.state.get("user_name", None)

    if user_name is None:
        # Name not saved yet — tell AI to call update_user_info first
        return Command(update={
            "messages": [
                ToolMessage(
                    content="Please call 'update_user_info' first to get the user's name.",
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        })

    return f"Hello {user_name}! How can I help you today?"

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[update_user_info, greet],
    state_schema=CustomState,
    context_schema=CustomContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Please greet me."}]},
    context=CustomContext(user_id="user_123"),
)
print(result["messages"][-1].content)
# Agent calls update_user_info → saves "John Smith" → calls greet → "Hello John Smith!"