"""
TOPIC: Custom State Schema in Middleware

WHAT IT DOES:
    Extends the agent's built-in state with your own custom fields.
    Custom state lets middleware track counters, flags, user info, and
    other values that persist across the entire agent execution.

WHY THIS MATTERS:
    Hooks from different decorators/classes run at different times.
    Without custom state, there's no safe way to share data between them.
    With custom state, before_model can write a value that after_agent reads.

HOW IT WORKS:
    1. Define a class that extends AgentState
    2. Add your fields with NotRequired[type] (so they're optional)
    3. Pass state_schema=YourClass to the decorator
    4. Access your fields via state.get("field_name", default)
    5. Return {"field_name": new_value} to update them

BUILT-IN AgentState FIELDS:
    messages: list[BaseMessage]   ← the conversation history

YOUR CUSTOM FIELDS:
    Any key you add will be merged into the agent's LangGraph state.
    Use NotRequired so the agent still works without them being set.

NODE-STYLE STATE UPDATES:
    Return a dict → merged into state via graph reducers

WRAP-STYLE STATE UPDATES:
    Return ExtendedModelResponse(model_response=..., command=Command(update=...))
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    before_model,
    after_model,
    before_agent,
    after_agent,
    AgentState,
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    ExtendedModelResponse,
)
from langchain.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing import Any, Callable
from typing_extensions import NotRequired
import time


# ---------------------------------------------------------------------------
# 1. Define custom state — extend AgentState
# ---------------------------------------------------------------------------
class MonitoringState(AgentState):
    """
    Custom state that tracks:
      - model_call_count : how many times the model was called
      - total_tokens     : accumulated token count
      - session_start    : Unix timestamp when agent started
      - user_id          : the requesting user (set at invoke time)
    """
    model_call_count: NotRequired[int]
    total_tokens:     NotRequired[int]
    session_start:    NotRequired[float]
    user_id:          NotRequired[str]


# ---------------------------------------------------------------------------
# 2. Hooks that USE the custom state
# ---------------------------------------------------------------------------

@before_agent(state_schema=MonitoringState)
def initialize_session(state: MonitoringState, runtime: Runtime) -> dict[str, Any] | None:
    """Record session start time (runs once)."""
    uid = state.get("user_id", "anonymous")
    print(f"Session started for user: {uid}")
    return {"session_start": time.time()}   # write to custom state


@before_model(state_schema=MonitoringState, can_jump_to=["end"])
def enforce_call_limit(state: MonitoringState, runtime: Runtime) -> dict[str, Any] | None:
    """Block agents that exceed their call quota."""
    count = state.get("model_call_count", 0)
    uid   = state.get("user_id", "anonymous")

    if count >= 10:
        print(f"User {uid} hit call limit ({count} calls)")
        return {
            "messages": [AIMessage(content="Request limit exceeded.")],
            "jump_to": "end",
        }
    return None   # no state update needed here — count updated in after_model


@after_model(state_schema=MonitoringState)
def increment_call_count(state: MonitoringState, runtime: Runtime) -> dict[str, Any] | None:
    """Increment model call counter after each successful model call."""
    new_count = state.get("model_call_count", 0) + 1
    print(f"Model calls so far: {new_count}")
    return {"model_call_count": new_count}   # write back to custom state


@after_agent(state_schema=MonitoringState)
def log_session_summary(state: MonitoringState, runtime: Runtime) -> dict[str, Any] | None:
    """Print a summary when the agent finishes."""
    start = state.get("session_start", time.time())
    duration = time.time() - start
    calls  = state.get("model_call_count", 0)
    uid    = state.get("user_id", "anonymous")
    print(f"Session summary for {uid}: {calls} model calls in {duration:.2f}s")
    return None


# ---------------------------------------------------------------------------
# 3. Wrap-style hook that updates custom state
#    Uses ExtendedModelResponse + Command for state updates from wrap layer
# ---------------------------------------------------------------------------
@wrap_model_call(state_schema=MonitoringState)
def track_token_usage(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ExtendedModelResponse:
    """Track token usage in custom state via Command."""
    response = handler(request)

    # Extract token count from response metadata if available
    tokens_used = 0
    for msg in response.messages:
        usage = getattr(msg, "usage_metadata", None)
        if usage:
            tokens_used = usage.get("total_tokens", 0)

    # Use Command to update custom state from the wrap layer
    return ExtendedModelResponse(
        model_response=response,
        command=Command(update={
            "total_tokens": tokens_used,   # overwrites previous value
        }),
    )


# ---------------------------------------------------------------------------
# 4. Create agent with custom state middleware
# ---------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"22°C and sunny in {city}."


agent = create_agent(
    model="gpt-4.1",
    tools=[get_weather],
    middleware=[
        initialize_session,    # before_agent
        enforce_call_limit,    # before_model  (reads model_call_count)
        track_token_usage,     # wrap_model_call (writes total_tokens via Command)
        increment_call_count,  # after_model   (writes model_call_count)
        log_session_summary,   # after_agent
    ],
)


# ---------------------------------------------------------------------------
# 5. Invoke — pass custom state fields in the initial dict
# ---------------------------------------------------------------------------
print("=" * 60)
print("Custom State Schema")
print("=" * 60)
print()

result = agent.invoke({
    "messages":          [HumanMessage(content="Weather in London?")],
    "model_call_count":  0,        # initialize custom field
    "total_tokens":      0,        # initialize custom field
    "user_id":           "user-42",  # pass user context
})

print()
print(f"Final model_call_count : {result['model_call_count']}")
print(f"Final total_tokens     : {result['total_tokens']}")
print(f"Final answer           : {result['messages'][-1].content[:200]}")

# ---------------------------------------------------------------------------
# EXPECTED OUTPUT:
#
# Session started for user: user-42
# Model calls so far: 1
# Model calls so far: 2
# Session summary for user-42: 2 model calls in 2.31s
#
# Final model_call_count : 2
# Final total_tokens     : 342
# ---------------------------------------------------------------------------