"""
TOPIC: Tool Context — Reads

WHAT IT MEANS:
    Tools need more than just the arguments the LLM passes them.
    A database query tool needs a connection string.
    A user-data tool needs a user ID.
    An API tool needs an API key.

    Instead of hardcoding these or using globals, tools READ them
    from three data sources at runtime:

THREE READ SOURCES:
    STATE           → session data that changes during the conversation
                      e.g. authenticated, uploaded_files, current cart
    STORE           → long-term memory that persists across sessions
                      e.g. user preferences, saved facts
    RUNTIME CONTEXT → static config injected at invoke time
                      e.g. user_id, api_key, db_connection

HOW TO READ:
    Add ToolRuntime[YourContext] as a parameter to your tool.
    LangChain injects it automatically — the model never sees it.

    runtime.state              → current session state dict
    runtime.store              → BaseStore instance
    runtime.context.field      → your context dataclass fields
"""

from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.store.memory import InMemoryStore


# ─────────────────────────────────────────────────────────────────────────────
# READ 1: From STATE
#
# WHY STATE:
#   Auth status, uploaded files, and other session data live in state.
#   They are specific to this conversation and change as it progresses.
#
# USE WHEN:
#   Checking if the user is logged in before doing something sensitive
#   Reading what files are available this session
#   Any data that was set earlier in this same conversation
# ─────────────────────────────────────────────────────────────────────────────

@tool
def check_authentication(runtime: ToolRuntime) -> str:
    """Check if the current user is authenticated in this session."""
    # Read from STATE
    is_authenticated = runtime.state.get("authenticated", False)
    user_id          = runtime.state.get("auth_user_id", "unknown")

    if is_authenticated:
        return f"User {user_id} is authenticated. Full access granted."
    else:
        return "User is NOT authenticated. Please log in first."


@tool
def list_uploaded_files(runtime: ToolRuntime) -> str:
    """List all files the user has uploaded in this session."""
    # Read from STATE
    files = runtime.state.get("uploaded_files", [])

    if not files:
        return "No files uploaded in this session."

    lines = [f"- {f['name']} ({f['type']}): {f['summary']}" for f in files]
    return "Uploaded files:\n" + "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# READ 2: From STORE
#
# WHY STORE:
#   User preferences, learned facts, and history persist across sessions.
#   The user set their preference last week — it should still be there today.
#   State would forget it; store keeps it forever.
#
# USE WHEN:
#   Loading saved user preferences (tone, language, format)
#   Recalling facts learned in previous conversations
#   Any data that should survive beyond one conversation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserContext:
    user_id: str


@tool
def get_preference(
    preference_key: str,
    runtime: ToolRuntime[UserContext],
) -> str:
    """Get a saved user preference from long-term memory."""
    user_id = runtime.context.user_id   # get user_id from context
    store   = runtime.store             # access the store

    item = store.get(("preferences",), user_id)

    if not item:
        return f"No preferences saved for this user yet."

    value = item.value.get(preference_key)
    if value:
        return f"{preference_key}: {value}"
    else:
        return f"No preference set for '{preference_key}'."


@tool
def get_all_preferences(runtime: ToolRuntime[UserContext]) -> str:
    """Get all saved preferences for the current user."""
    user_id = runtime.context.user_id
    store   = runtime.store

    item = store.get(("preferences",), user_id)

    if not item or not item.value:
        return "No preferences saved yet."

    lines = [f"  {k}: {v}" for k, v in item.value.items()]
    return "Your saved preferences:\n" + "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# READ 3: From RUNTIME CONTEXT
#
# WHY RUNTIME CONTEXT:
#   API keys, database URLs, and user identity are per-request config.
#   They should never be hardcoded in tools — they change per tenant/user.
#   Runtime context injects them at invoke time.
#
# USE WHEN:
#   Database connections (different per environment or tenant)
#   API keys (never hardcode secrets in tool code)
#   User ID for scoping queries to the right user
#   Any config that varies per request but is fixed during the run
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AppContext:
    user_id:       str
    api_key:       str
    db_connection: str


@tool
def fetch_user_data(
    query: str,
    runtime: ToolRuntime[AppContext],
) -> str:
    """Fetch data for the current user using injected config."""
    # Read from RUNTIME CONTEXT — never hardcoded
    user_id       = runtime.context.user_id
    api_key       = runtime.context.api_key
    db_connection = runtime.context.db_connection

    # In a real app: use db_connection + api_key to run the query
    print(f"  [tool] Querying {db_connection} for user {user_id}")
    print(f"  [tool] Using API key: {api_key[:8]}...")

    return (
        f"Results for user {user_id}:\n"
        f"  Query: {query}\n"
        f"  DB: {db_connection}\n"
        f"  Found: 42 matching records"
    )


@tool
def call_external_service(
    endpoint: str,
    runtime: ToolRuntime[AppContext],
) -> str:
    """Call an external service using the injected API key."""
    api_key = runtime.context.api_key   # injected — never hardcoded

    # In a real app: requests.get(endpoint, headers={"Authorization": f"Bearer {api_key}"})
    print(f"  [tool] Calling {endpoint} with key {api_key[:8]}...")
    return f"Response from {endpoint}: {{status: 200, data: 'success'}}"


# ─────────────────────────────────────────────────────────────────────────────
# Agents and tests
# ─────────────────────────────────────────────────────────────────────────────

# Agent 1: reads from state
agent_state = create_agent(
    model="gpt-4.1",
    tools=[check_authentication, list_uploaded_files],
)

print("=" * 60)
print("READ 1: From STATE")
print("=" * 60)

print("\n─── Not authenticated, no files ───")
r = agent_state.invoke({
    "messages":      [HumanMessage("Am I logged in? What files do I have?")],
    "authenticated": False,
    "uploaded_files": [],
})
print(f"Response: {r['messages'][-1].content[:250]}")

print("\n─── Authenticated with files ───")
r = agent_state.invoke({
    "messages":      [HumanMessage("Am I logged in? What files do I have?")],
    "authenticated": True,
    "auth_user_id":  "u-alice",
    "uploaded_files": [
        {"name": "report.pdf",    "type": "PDF",  "summary": "Q3 financial report"},
        {"name": "contacts.csv",  "type": "CSV",  "summary": "Client contact list"},
    ],
})
print(f"Response: {r['messages'][-1].content[:250]}")


# Agent 2: reads from store
store = InMemoryStore()
store.put(("preferences",), "u-alice", {
    "tone":     "casual",
    "language": "English",
    "format":   "bullet points",
})

agent_store = create_agent(
    model="gpt-4.1",
    tools=[get_preference, get_all_preferences],
    context_schema=UserContext,
    store=store,
)

print("\n" + "=" * 60)
print("READ 2: From STORE")
print("=" * 60)

print("\n─── Alice reads her saved preferences ───")
r = agent_store.invoke(
    {"messages": [HumanMessage("What is my saved tone preference? Show me all my preferences too.")]},
    context=UserContext(user_id="u-alice"),
)
print(f"Response: {r['messages'][-1].content[:300]}")

print("\n─── New user — nothing saved yet ───")
r = agent_store.invoke(
    {"messages": [HumanMessage("What are my saved preferences?")]},
    context=UserContext(user_id="u-newuser"),
)
print(f"Response: {r['messages'][-1].content[:200]}")


# Agent 3: reads from runtime context
agent_context = create_agent(
    model="gpt-4.1",
    tools=[fetch_user_data, call_external_service],
    context_schema=AppContext,
)

print("\n" + "=" * 60)
print("READ 3: From RUNTIME CONTEXT")
print("=" * 60)

r = agent_context.invoke(
    {"messages": [HumanMessage("Fetch my recent orders and call the /status endpoint")]},
    context=AppContext(
        user_id="user_123",
        api_key="sk-super-secret-key-abc123",
        db_connection="postgresql://prod-db:5432/orders",
    ),
)
print(f"Response: {r['messages'][-1].content[:300]}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Quick Reference — tool reads")
print("=" * 60)
print("""
  Source           │ How to read in a tool              │ Use for
  ─────────────────┼────────────────────────────────────┼──────────────────────────
  STATE            │ runtime.state.get("key", default)  │ Auth, uploads, session data
  STORE            │ runtime.store.get(("ns",), key)    │ Preferences, cross-session facts
  RUNTIME CONTEXT  │ runtime.context.field_name         │ user_id, api_key, db_url

  Tool signature:
    @tool
    def my_tool(arg: str, runtime: ToolRuntime[MyContext]) -> str:
        runtime.state.get(...)          # state
        runtime.store.get(...)          # store
        runtime.context.my_field        # context
        ...

  Note: the runtime parameter is hidden from the model.
        The model only sees arg: str.
""")