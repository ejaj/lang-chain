"""
TOPIC: MCP Interceptors — Accessing Runtime Context

WHAT ARE INTERCEPTORS:
    MCP servers run as separate processes — they can't access LangGraph's
    runtime information (store, context, agent state).
    Interceptors bridge this gap: they run inside the LangChain process
    and have access to all runtime data.

    Interceptor = middleware for MCP tool calls.

    async def my_interceptor(request: MCPToolCallRequest, handler):
        # modify request, access runtime, short-circuit, etc.
        result = await handler(request)   # call the actual tool
        # post-process result
        return result

FOUR RUNTIME SOURCES IN INTERCEPTORS:
    request.runtime.context      → your context dataclass (user_id, api_key, etc.)
    request.runtime.store        → BaseStore for long-term memory
    request.runtime.state        → current agent state dict
    request.runtime.tool_call_id → the specific tool call ID for logging

MODIFYING REQUESTS:
    Use request.override(args={...}) to change the tool arguments before
    the tool executes. Immutable — original request unchanged.
"""

import asyncio
from dataclasses import dataclass
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain.agents import create_agent
from langchain.messages import ToolMessage
from langgraph.store.memory import InMemoryStore


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1: runtime.context — inject user credentials into tool calls
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AppContext:
    user_id: str
    api_key: str


async def inject_user_context(
    request: MCPToolCallRequest,
    handler,
):
    """
    Read user_id and api_key from runtime context.
    Inject them into MCP tool arguments so the server knows who is calling.
    The MCP server itself cannot access runtime context — this interceptor bridges the gap.
    """
    runtime = request.runtime
    user_id = runtime.context.user_id   # from runtime context
    api_key = runtime.context.api_key   # from runtime context

    print(f"  [interceptor] inject_user_context → user={user_id}")

    # Add user context to the tool's arguments
    modified_request = request.override(
        args={**request.args, "user_id": user_id, "api_key": api_key}
    )
    return await handler(modified_request)


async def context_example():
    client = MultiServerMCPClient(
        {
            "orders": {
                "transport": "http",
                "url":       "http://localhost:8001/mcp",
            }
        },
        tool_interceptors=[inject_user_context],
    )

    tools  = await client.get_tools()
    agent  = create_agent("gpt-4.1", tools, context_schema=AppContext)

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Search my orders"}]},
        context=AppContext(user_id="user_123", api_key="sk-secret"),
    )
    print(f"Result: {result['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2: runtime.store — personalize tool calls from saved preferences
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserContext:
    user_id: str


async def personalize_from_store(
    request: MCPToolCallRequest,
    handler,
):
    """
    Read user preferences from the store and apply them to tool arguments.
    The store has data from previous sessions — the user's saved settings.
    """
    runtime = request.runtime
    user_id = runtime.context.user_id
    store   = runtime.store   # long-term memory store

    if store and request.name == "search":
        prefs = store.get(("preferences",), user_id)
        if prefs:
            # Apply saved preferences to the search tool arguments
            modified_args = {
                **request.args,
                "language": prefs.value.get("language", "en"),
                "limit":    prefs.value.get("result_limit", 10),
            }
            print(f"  [interceptor] personalize_from_store → applied prefs for {user_id}")
            request = request.override(args=modified_args)

    return await handler(request)


async def store_example():
    store = InMemoryStore()
    store.put(("preferences",), "user_123", {
        "language":     "fr",
        "result_limit": 5,
    })

    client = MultiServerMCPClient(
        {
            "search": {
                "transport": "http",
                "url":       "http://localhost:8001/mcp",
            }
        },
        tool_interceptors=[personalize_from_store],
    )

    tools  = await client.get_tools()
    agent  = create_agent("gpt-4.1", tools, context_schema=UserContext, store=store)

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Search for Python tutorials"}]},
        context=UserContext(user_id="user_123"),
    )
    print(f"Result: {result['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3: runtime.state — gate tools based on session state
# ─────────────────────────────────────────────────────────────────────────────

async def require_authentication(
    request: MCPToolCallRequest,
    handler,
):
    """
    Block sensitive MCP tools if the user is not authenticated in state.
    State holds auth status set earlier in this conversation.
    """
    runtime           = request.runtime
    state             = runtime.state   # current agent state
    is_authenticated  = state.get("authenticated", False)

    sensitive_tools = {"delete_file", "update_settings", "export_data"}

    if request.name in sensitive_tools and not is_authenticated:
        print(f"  [interceptor] BLOCKED — {request.name} requires authentication")
        # Return an error ToolMessage instead of calling the tool
        return ToolMessage(
            content="Authentication required. Please log in first.",
            tool_call_id=runtime.tool_call_id,
        )

    return await handler(request)


async def state_example():
    client = MultiServerMCPClient(
        {
            "files": {
                "transport": "http",
                "url":       "http://localhost:8001/mcp",
            }
        },
        tool_interceptors=[require_authentication],
    )

    tools = await client.get_tools()
    agent = create_agent("gpt-4.1", tools)

    # Not authenticated — should be blocked
    print("─── Not authenticated ───")
    r1 = await agent.ainvoke({
        "messages":      [{"role": "user", "content": "Delete the old log files"}],
        "authenticated": False,
    })
    print(f"Result: {r1['messages'][-1].content[:200]}")

    # Authenticated — should proceed
    print("\n─── Authenticated ───")
    r2 = await agent.ainvoke({
        "messages":      [{"role": "user", "content": "Delete the old log files"}],
        "authenticated": True,
    })
    print(f"Result: {r2['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 4: runtime.tool_call_id — logging and rate limiting
# ─────────────────────────────────────────────────────────────────────────────

_rate_limited_tools: dict[str, int] = {}

async def rate_limit_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """
    Use tool_call_id for logging. Check rate limits before execution.
    tool_call_id uniquely identifies this specific invocation.
    """
    runtime      = request.runtime
    tool_call_id = runtime.tool_call_id   # unique ID for this call

    # Check rate limit (simplified)
    call_count = _rate_limited_tools.get(request.name, 0)
    if call_count >= 3:
        print(f"  [interceptor] RATE LIMITED — {request.name} (call_id={tool_call_id[:8]})")
        return ToolMessage(
            content="Rate limit exceeded. Please try again later.",
            tool_call_id=tool_call_id,
        )

    _rate_limited_tools[request.name] = call_count + 1
    print(f"  [interceptor] Executing {request.name} (call_id={tool_call_id[:8]}...)")
    result = await handler(request)
    print(f"  [interceptor] {request.name} completed")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("""
Interceptor Runtime Context Quick Reference:

  Source              │ Access pattern                       │ Use for
  ────────────────────┼──────────────────────────────────────┼─────────────────────────
  Runtime context     │ request.runtime.context.field        │ user_id, api_key, config
  Store               │ request.runtime.store.get(...)       │ Saved user preferences
  State               │ request.runtime.state.get(...)       │ Auth status, session data
  Tool call ID        │ request.runtime.tool_call_id         │ Logging, tracing

  Modify request args:
    modified = request.override(args={**request.args, "new_key": value})
    return await handler(modified)

  Short-circuit (block tool):
    return ToolMessage(
        content="Error message for the model",
        tool_call_id=request.runtime.tool_call_id,
    )
""")

if __name__ == "__main__":
    asyncio.run(context_example())