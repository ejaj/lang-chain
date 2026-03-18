"""
tool_selection_examples.py
============================
TOPIC: Dynamic Tool Selection

WHAT IT IS:
    Filtering the available tools before every model call so the model
    only sees the tools that are appropriate for the current situation.

WHY IT MATTERS:
    Too many tools  → model gets confused, picks the wrong one, makes errors
    Too few tools   → model can't complete the task
    Right tools     → model focuses, picks correctly, costs less tokens

HOW IT WORKS:
    Use wrap_model_call + request.override(tools=filtered_list)
    All tools must be registered at create_agent() time.
    override(tools=...) picks a SUBSET — you cannot add new tools dynamically.

THREE SELECTION PATTERNS:
    1. From STATE          → auth status, conversation stage
    2. Tool definition     → how to write tools the model uses correctly
    3. From RUNTIME CONTEXT → user role / permissions
"""

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool
from langchain.messages import HumanMessage
from typing import Callable


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def public_search(query: str) -> str:
    """Search public product catalog and documentation. No login required."""
    return f"Public results for '{query}': [doc_1, doc_2, doc_3]"


@tool
def private_search(query: str) -> str:
    """Search private account data and order history. Requires authentication."""
    return f"Private results for '{query}': [order_001, order_042]"


@tool
def advanced_search(query: str, filters: str) -> str:
    """Advanced search with filters. Only available after several messages."""
    return f"Advanced results for '{query}' with filters '{filters}': [result_1]"


@tool
def read_data(resource: str) -> str:
    """Read a resource. Available to all roles."""
    return f"Data for '{resource}': {{id: 1, value: 'example'}}"


@tool
def write_data(resource: str, value: str) -> str:
    """Write to a resource. Requires editor or admin role."""
    return f"Written '{value}' to '{resource}' successfully."


@tool
def delete_data(resource: str) -> str:
    """Delete a resource. Requires admin role only."""
    return f"Deleted '{resource}' permanently."


@tool(parse_docstring=True)
def search_orders(
    user_id: str,
    status: str,
    limit: int = 10,
) -> str:
    """Search for user orders by status.

    Use this when the user asks about order history or wants to check
    order status. Always filter by the provided status.

    Args:
        user_id: Unique identifier for the user making the request
        status: Order status to filter by: 'pending', 'shipped', or 'delivered'
        limit: Maximum number of orders to return (default 10, max 50)
    """
    return f"Orders for {user_id} with status '{status}': [order_1, order_2] (limit={limit})"


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1: Select tools from STATE
#
# WHY STATE:
#   Auth status and conversation length live in state — they change during
#   the conversation and are specific to this session.
#
# LOGIC:
#   Not authenticated  → public tools only (hide sensitive data)
#   Authenticated, early conversation (<5 msgs) → no advanced_search yet
#   Authenticated, established conversation → all tools unlocked
#
# USE WHEN:
#   You want to gate tools behind a login step
#   You want to introduce tools gradually as conversation progresses
#   Feature flags stored in session state
# ─────────────────────────────────────────────────────────────────────────────

@wrap_model_call
def state_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Filter tools based on conversation state."""
    state            = request.state
    is_authenticated = state.get("authenticated", False)
    message_count    = len(state["messages"])

    if not is_authenticated:
        # Not logged in — public tools only
        tools = [t for t in request.tools if t.name.startswith("public_")]
        print(f"  [tools] Not authenticated → {[t.name for t in tools]}")

    elif message_count < 5:
        # Logged in but early in conversation — hide advanced_search
        tools = [t for t in request.tools if t.name != "advanced_search"]
        print(f"  [tools] Auth, early ({message_count} msgs) → {[t.name for t in tools]}")

    else:
        # Logged in, established conversation — all tools available
        tools = request.tools
        print(f"  [tools] Auth, established → all {len(tools)} tools")

    return handler(request.override(tools=tools))


agent_state = create_agent(
    model="gpt-4.1",
    tools=[public_search, private_search, advanced_search],
    middleware=[state_based_tools],
)

print("=" * 60)
print("EXAMPLE 1: Tool selection from STATE")
print("=" * 60)

print("\n─── Not authenticated ───")
r = agent_state.invoke({
    "messages":      [HumanMessage("Search for my recent orders")],
    "authenticated": False,
})
print(f"Response: {r['messages'][-1].content[:200]}")

print("\n─── Authenticated, early conversation ───")
r = agent_state.invoke({
    "messages":      [HumanMessage("Search for my recent orders")],
    "authenticated": True,
})
print(f"Response: {r['messages'][-1].content[:200]}")

print("\n─── Authenticated, established conversation ───")
r = agent_state.invoke({
    "messages": [
        {"role": "user",      "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user",      "content": "What can you do?"},
        {"role": "assistant", "content": "I can search..."},
        {"role": "user",      "content": "Search with filters for electronics under $50"},
    ],
    "authenticated": True,
})
print(f"Response: {r['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 2: How to define tools well
#
# WHY THIS MATTERS:
#   The tool name, description, and argument descriptions ARE the prompt
#   the model uses to decide when and how to call your tool.
#   Vague descriptions = model calls the wrong tool or passes wrong args.
#
# GOOD TOOL DEFINITION:
#   - Name says exactly what it does
#   - Description says WHEN to use it (not just what it does)
#   - Each arg has a clear description with valid values listed
#   - Use parse_docstring=True to pull Args section automatically
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EXAMPLE 2: Tool definition best practices")
print("=" * 60)

agent_orders = create_agent(
    model="gpt-4.1",
    tools=[search_orders],
)

r = agent_orders.invoke({
    "messages": [HumanMessage("Show me all my shipped orders, user ID is u-42")]
})
print(f"Response: {r['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 3: Select tools from RUNTIME CONTEXT
#
# WHY RUNTIME CONTEXT:
#   User role is set at login and doesn't change during a session.
#   It's per-request configuration — the right fit for runtime context.
#
# LOGIC:
#   admin  → all tools (read + write + delete)
#   editor → read + write only (no delete)
#   viewer → read only
#
# USE WHEN:
#   Role-based access control (RBAC)
#   Multi-tenant apps where different users have different permissions
#   Any permission that is fixed for the duration of a request
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserContext:
    user_role: str   # "admin" | "editor" | "viewer"


@wrap_model_call
def context_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Filter tools based on user role from runtime context."""
    user_role = request.runtime.context.user_role

    if user_role == "admin":
        # Admins get everything
        tools = request.tools
        print(f"  [tools] Admin → all {len(tools)} tools")

    elif user_role == "editor":
        # Editors can read and write but not delete
        tools = [t for t in request.tools if t.name != "delete_data"]
        print(f"  [tools] Editor → {[t.name for t in tools]}")

    else:
        # Viewers get read-only tools only
        tools = [t for t in request.tools if t.name.startswith("read_")]
        print(f"  [tools] Viewer → {[t.name for t in tools]}")

    return handler(request.override(tools=tools))


agent_context = create_agent(
    model="gpt-4.1",
    tools=[read_data, write_data, delete_data],
    middleware=[context_based_tools],
    context_schema=UserContext,
)

print("\n" + "=" * 60)
print("EXAMPLE 3: Tool selection from RUNTIME CONTEXT")
print("=" * 60)

print("\n─── Admin ───")
r = agent_context.invoke(
    {"messages": [HumanMessage("Delete the old test records and read the users table")]},
    context=UserContext(user_role="admin"),
)
print(f"Response: {r['messages'][-1].content[:200]}")

print("\n─── Editor ───")
r = agent_context.invoke(
    {"messages": [HumanMessage("Delete the old test records and read the users table")]},
    context=UserContext(user_role="editor"),
)
print(f"Response: {r['messages'][-1].content[:200]}")

print("\n─── Viewer ───")
r = agent_context.invoke(
    {"messages": [HumanMessage("Delete the old test records and read the users table")]},
    context=UserContext(user_role="viewer"),
)
print(f"Response: {r['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Quick Reference — dynamic tool selection")
print("=" * 60)
print("""
  Source           │ Read pattern                        │ Use for
  ─────────────────┼─────────────────────────────────────┼──────────────────────────
  STATE            │ request.state.get("key", default)   │ Auth status, conv length,
                   │ len(request.messages)               │ feature flags this session
  ─────────────────┼─────────────────────────────────────┼──────────────────────────
  RUNTIME CONTEXT  │ request.runtime.context.field       │ User role, plan tier,
                   │                                     │ tenant permissions

  Override pattern:
    tools   = [t for t in request.tools if <condition>]
    request = request.override(tools=tools)
    return handler(request)

  Rule: ALL tools must be registered at create_agent(tools=[...]) time.
        override(tools=...) can only pick a SUBSET — not add new ones.
""")