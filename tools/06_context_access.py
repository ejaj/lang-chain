# TYPE: Access Context (Immutable Config)
# DESCRIPTION: Context is read-only data passed at invoke() time — things like
# user ID, session ID, or permissions that don't change during the conversation.
# Use runtime.context to read it inside tools.
# Unlike state, context cannot be updated by tools.

from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

# --- A fake user database ---
USER_DB = {
    "user123": {"name": "Alice Johnson", "plan": "Premium", "balance": 5000},
    "user456": {"name": "Bob Smith",     "plan": "Standard", "balance": 1200},
}

# --- Define what context looks like ---
@dataclass
class UserContext:
    user_id: str      # who is logged in — passed at invoke time, never changes

# --- Tool reads user_id from context ---
@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the logged-in user's account information."""
    user_id = runtime.context.user_id   # read from context

    user = USER_DB.get(user_id)
    if user:
        return f"Name: {user['name']}\nPlan: {user['plan']}\nBalance: ${user['balance']}"
    return "User not found"

@tool
def check_balance(runtime: ToolRuntime[UserContext]) -> str:
    """Check the current user's balance."""
    user_id = runtime.context.user_id
    user = USER_DB.get(user_id)
    return f"Your balance is ${user['balance']}" if user else "User not found"

agent = create_agent(
    model=ChatOpenAI(model="gpt-4.1"),
    tools=[get_account_info, check_balance],
    context_schema=UserContext,
    system_prompt="You are a financial assistant.",
)

# Alice logs in → her user_id passed as context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my balance?"}]},
    context=UserContext(user_id="user123"),   # ← Alice
)
print(result["messages"][-1].content)
# → "Your balance is $5000"

# Bob logs in → different user_id, different result, same agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my balance?"}]},
    context=UserContext(user_id="user456"),   # ← Bob
)
print(result["messages"][-1].content)
# → "Your balance is $1200"