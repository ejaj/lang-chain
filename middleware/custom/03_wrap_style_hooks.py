"""
03_wrap_style_hooks.py
========================
TOPIC: Wrap-Style Hooks (@wrap_model_call and @wrap_tool_call)

WHAT ARE WRAP-STYLE HOOKS:
    You wrap around the actual model or tool call and decide HOW it runs.
    Unlike node-style hooks (which just observe), wrap-style hooks CONTROL
    execution — you can call the handler 0 times (cache), 1 time (normal),
    or many times (retry).

TWO WRAP HOOKS:
    @wrap_model_call  → wraps each model API call
    @wrap_tool_call   → wraps each tool execution

SIGNATURE:
    @wrap_model_call
    def my_hook(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # YOUR code here
        result = handler(request)   # ← actually calls the model
        # MORE code here
        return result

    @wrap_tool_call
    def my_hook(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        # YOUR code here
        result = handler(request)   # ← actually executes the tool
        return result

KEY INSIGHT — you control when/if handler is called:
    - Skip handler entirely → return a cached result (0 calls)
    - Call handler once → normal behavior (1 call)
    - Call handler in a loop → retry logic (N calls)

WHEN TO USE WRAP-STYLE:
    Retry logic with backoff
    Response caching
    Request/response transformation
    Timing and cost measurement
    Tool call monitoring and logging
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    wrap_model_call,
    wrap_tool_call,
    ModelRequest,
    ModelResponse,
)
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command
from typing import Callable
import time


# ---------------------------------------------------------------------------
# EXAMPLE 1: @wrap_model_call — retry with backoff
# The handler is called multiple times if it fails.
# ---------------------------------------------------------------------------
@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Retry model calls up to 3 times with exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return handler(request)   # ← try calling the model
        except Exception as e:
            last_error = e
            if attempt < 2:   # don't wait after the final attempt
                wait = 2 ** attempt   # 1s, 2s
                print(f"⚠️  [retry_model] Attempt {attempt + 1} failed: {e}. "
                      f"Retrying in {wait}s...")
                time.sleep(wait)
    raise last_error   # all 3 attempts failed


# ---------------------------------------------------------------------------
# EXAMPLE 2: @wrap_model_call — response caching
# If we've seen this exact message history before, return the cached response.
# handler is called 0 times on cache hit, 1 time on miss.
# ---------------------------------------------------------------------------
_cache: dict[str, ModelResponse] = {}

@wrap_model_call
def cache_responses(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Cache model responses by message fingerprint."""
    # Build a simple cache key from the last user message
    last_user = next(
        (m for m in reversed(request.messages) if m.type == "human"),
        None,
    )
    cache_key = str(last_user.content) if last_user else ""

    if cache_key in _cache:
        print(f"[cache] HIT for: {cache_key[:60]}")
        return _cache[cache_key]   # handler NOT called — return cached value

    print(f"[cache] MISS for: {cache_key[:60]}")
    response = handler(request)   # handler called once
    _cache[cache_key] = response
    return response


# ---------------------------------------------------------------------------
# EXAMPLE 3: @wrap_model_call — timing measurement
# ---------------------------------------------------------------------------
@wrap_model_call
def time_model_calls(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Measure how long each model call takes."""
    start = time.time()
    response = handler(request)
    elapsed = time.time() - start
    print(f"[timer] Model responded in {elapsed:.2f}s")
    return response


# ---------------------------------------------------------------------------
# EXAMPLE 4: @wrap_tool_call — tool call monitoring
# Logs every tool execution with its args and result.
# ---------------------------------------------------------------------------
@wrap_tool_call
def monitor_tools(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """Log every tool call with args and result."""
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]
    print(f"[tool_monitor] CALLING: {tool_name}({tool_args})")
    try:
        result = handler(request)   # ← actually execute the tool
        # Extract content from ToolMessage
        content = getattr(result, "content", str(result))
        print(f"[tool_monitor] RESULT : {str(content)[:100]}")
        return result
    except Exception as e:
        print(f"[tool_monitor] ERROR  : {tool_name} failed — {e}")
        raise


# ---------------------------------------------------------------------------
# EXAMPLE 5: @wrap_tool_call — tool result transformation
# Adds a timestamp prefix to every tool result.
# ---------------------------------------------------------------------------
@wrap_tool_call
def stamp_tool_results(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """Prepend a timestamp to every tool result."""
    result = handler(request)
    if isinstance(result, ToolMessage):
        timestamp = time.strftime("%H:%M:%S")
        result.content = f"[{timestamp}] {result.content}"
    return result


# ---------------------------------------------------------------------------
# Wire into agent
# ---------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"22°C and sunny in {city}."


agent = create_agent(
    model="gpt-4.1",
    tools=[get_weather],
    middleware=[
        time_model_calls,   # measure model latency
        monitor_tools,      # log tool calls
        stamp_tool_results, # add timestamp to tool results
        retry_model,        # retry on model failure
    ],
)

print("=" * 60)
print("Wrap-style hooks — wrap_model_call + wrap_tool_call")
print("=" * 60)
print()

result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]
})
print()
print("Final answer:")
print(result["messages"][-1].content[:300])

# ---------------------------------------------------------------------------
# EXPECTED OUTPUT:
#
# [cache] MISS for: What's the weather in Tokyo?
# [timer] Model responded in 1.45s
# [tool_monitor] CALLING: get_weather({'city': 'Tokyo'})
# [tool_monitor] RESULT : 22°C and sunny in Tokyo.
# [timer] Model responded in 0.88s
#
# Final answer:
# [14:30:22] The weather in Tokyo is 22°C and sunny.
# ---------------------------------------------------------------------------