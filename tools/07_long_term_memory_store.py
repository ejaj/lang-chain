# TYPE: Long-term Memory (Store)
# DESCRIPTION: Store = memory that survives across sessions.
# Unlike state (lives only during one conversation), store data is PERSISTENT.
# Use it to remember user preferences, history, or facts across many conversations.
# Access via runtime.store inside tools.

from typing import Any
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore  # swap for PostgresStore in production

# --- Tool: save data to store ---
@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """Save user information to long-term memory."""
    runtime.store.put(
        ("users",),   # namespace — like a folder name
        user_id,      # key — like a filename
        user_info     # value — the data to save
    )
    return f"Saved info for user {user_id}"

# --- Tool: read data from store ---
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up saved user information from long-term memory."""
    result = runtime.store.get(("users",), user_id)
    if result:
        return str(result.value)
    return f"No info found for user {user_id}"

store = InMemoryStore()   # use PostgresStore in production

agent = create_agent(
    model=ChatOpenAI(model="gpt-4.1"),
    tools=[save_user_info, get_user_info],
    store=store,
)

# --- Session 1: save user ---
agent.invoke({
    "messages": [{"role": "user", "content": "Save this user: id=abc123, name=Alice, age=30, email=alice@example.com"}]
})

# --- Session 2: retrieve user (new conversation, store still has data) ---
result = agent.invoke({
    "messages": [{"role": "user", "content": "Get info for user abc123"}]
})
print(result["messages"][-1].content)
# → "Name: Alice, Age: 30, Email: alice@example.com"

# KEY DIFFERENCE:
# State  → memory only during ONE conversation (gone after invoke)
# Store  → memory FOREVER across ALL conversations (persists)