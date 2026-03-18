"""
TOPIC: MCP Interceptors — Custom Patterns

FOUR PATTERNS:
    1. Basic     → log, inspect, observe tool calls
    2. Modify    → change request args or headers before tool runs
    3. Compose   → chain multiple interceptors (onion model)
    4. Error     → retry on failure, fallback on error

INTERCEPTOR SIGNATURE:
    async def my_interceptor(
        request: MCPToolCallRequest,
        handler,              # calling handler runs the actual tool
    ):
        # BEFORE: runs before tool
        result = await handler(request)   # ← calls the tool
        # AFTER:  runs after tool
        return result

YOU CONTROL WHEN handler IS CALLED:
    Never  → short-circuit, return cached/fallback result (0 calls)
    Once   → normal execution (1 call)
    Many   → retry logic (N calls)

ONION ORDER — [outer, inner]:
    outer.before → inner.before → TOOL → inner.after → outer.after
"""

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 1: Basic — logging and observation
# The simplest interceptor: observe without modifying.
# ─────────────────────────────────────────────────────────────────────────────

async def logging_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """Log every tool call — what was called, with what args, what came back."""
    print(f"  [log] → {request.name}({request.args})")

    result = await handler(request)   # call the tool normally

    print(f"  [log] ← {request.name} returned: {str(result)[:80]}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 2: Modify — change request before tool runs
# Use request.override() — immutable, original unchanged.
# ─────────────────────────────────────────────────────────────────────────────

async def double_args_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """Double all numeric arguments before the tool executes."""
    modified_args    = {k: v * 2 if isinstance(v, (int, float)) else v
                        for k, v in request.args.items()}
    modified_request = request.override(args=modified_args)

    print(f"  [modify] Original: {request.args} → Modified: {modified_args}")
    return await handler(modified_request)


async def add_auth_header_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """Add auth headers dynamically based on which tool is being called."""
    token            = f"token-for-{request.name}"   # per-tool token logic
    modified_request = request.override(
        headers={"Authorization": f"Bearer {token}"}
    )

    print(f"  [headers] Added auth header for {request.name}")
    return await handler(modified_request)


async def filter_sensitive_args(
    request: MCPToolCallRequest,
    handler,
):
    """
    Validate and sanitize args before passing to tool.
    Example: clamp a 'limit' argument to a maximum value.
    """
    args = dict(request.args)

    if "limit" in args:
        original = args["limit"]
        args["limit"] = min(args["limit"], 100)   # clamp to 100 max
        if args["limit"] != original:
            print(f"  [sanitize] Clamped limit {original} → {args['limit']}")

    return await handler(request.override(args=args))


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 3: Compose — multiple interceptors, onion order
# First interceptor in the list = outermost layer.
# ─────────────────────────────────────────────────────────────────────────────

async def outer_interceptor(request, handler):
    print("  [outer] BEFORE tool")
    result = await handler(request)
    print("  [outer] AFTER  tool")
    return result


async def inner_interceptor(request, handler):
    print("  [inner] BEFORE tool")
    result = await handler(request)
    print("  [inner] AFTER  tool")
    return result


# Execution order: outer.before → inner.before → TOOL → inner.after → outer.after
# tool_interceptors=[outer_interceptor, inner_interceptor]


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 4a: Error handling — retry with exponential backoff
# Call handler multiple times if it fails.
# ─────────────────────────────────────────────────────────────────────────────

import asyncio as _asyncio


async def retry_interceptor(
    request: MCPToolCallRequest,
    handler,
    max_retries: int = 3,
    delay: float = 1.0,
):
    """Retry failed tool calls with exponential backoff."""
    last_error = None

    for attempt in range(max_retries):
        try:
            return await handler(request)   # try calling the tool

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = delay * (2 ** attempt)   # 1s → 2s → 4s
                print(f"  [retry] {request.name} failed (attempt {attempt + 1}/{max_retries}), "
                      f"retrying in {wait}s: {e}")
                await _asyncio.sleep(wait)
            else:
                print(f"  [retry] {request.name} failed all {max_retries} attempts")

    raise last_error


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 4b: Error handling — fallback value on specific errors
# Return a safe default instead of propagating the error to the model.
# ─────────────────────────────────────────────────────────────────────────────

async def fallback_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """Return a fallback value when specific errors occur."""
    try:
        return await handler(request)

    except TimeoutError:
        print(f"  [fallback] {request.name} timed out — using fallback")
        return f"Tool '{request.name}' timed out. Please try again later."

    except ConnectionError:
        print(f"  [fallback] {request.name} connection failed — using cached data")
        return f"Could not connect to '{request.name}' service. Using cached data."

    except Exception as e:
        print(f"  [fallback] {request.name} unexpected error: {e}")
        return f"An error occurred running '{request.name}'."


# ─────────────────────────────────────────────────────────────────────────────
# FULL EXAMPLE: Combine multiple interceptors
# ─────────────────────────────────────────────────────────────────────────────

async def full_example():
    """Production-grade interceptor stack: logging + validation + retry."""
    from langchain.agents import create_agent

    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command":   "python",
                "args":      ["/path/to/math_server.py"],
            }
        },
        tool_interceptors=[
            logging_interceptor,      # outermost: log everything
            filter_sensitive_args,    # middle: sanitize args
            retry_interceptor,        # innermost: retry on failure
        ],
    )

    tools  = await client.get_tools()
    agent  = create_agent("gpt-4.1", tools)

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is 6 + 7?"}]
    })
    print(f"Result: {result['messages'][-1].content}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("""
Custom Interceptor Quick Reference:

  Basic (observe only):
    async def my_interceptor(request, handler):
        result = await handler(request)
        return result

  Modify request (immutable):
    modified = request.override(args={**request.args, "key": value})
    modified = request.override(headers={"Authorization": "Bearer token"})
    return await handler(modified)

  Short-circuit (skip tool):
    return "fallback string"   # or ToolMessage(...)

  Retry:
    for attempt in range(max_retries):
        try:    return await handler(request)
        except: await asyncio.sleep(delay * 2**attempt)

  Compose — onion order:
    tool_interceptors=[outer, middle, inner]
    # outer.before → middle.before → inner.before → TOOL
    # → inner.after → middle.after → outer.after

  Attach to client:
    client = MultiServerMCPClient({...}, tool_interceptors=[a, b, c])
""")

if __name__ == "__main__":
    asyncio.run(full_example())