"""
01_node_style_hooks_decorator.py
==================================
TOPIC: Node-Style Hooks (Decorator API)

WHAT ARE NODE-STYLE HOOKS:
    Functions that run at fixed, sequential execution points in the agent loop.
    Think of them as event listeners attached to the agent's lifecycle.

FOUR AVAILABLE HOOKS:
    @before_agent  → runs ONCE when the agent starts (before any model call)
    @before_model  → runs before EACH model call
    @after_model   → runs after EACH model response
    @after_agent   → runs ONCE when the agent finishes (after last model call)

EXECUTION ORDER (per turn):
    before_agent
        ↓
    before_model  ←─────────┐
        ↓                   │
      [model call]          │ (loops if tool calls)
        ↓                   │
    after_model             │
        ↓                   │
      [tool calls] ─────────┘
        ↓
    after_agent

RETURN VALUES:
    return None          → no state change, continue normally
    return {"key": val}  → merge this dict into agent state
    return {"jump_to": "end", ...} → early exit (requires can_jump_to=["end"])

WHEN TO USE NODE-STYLE:
    Logging / auditing
    Input validation
    Rate limiting / call counting
    State tracking (incrementing counters, timestamps)
    Anything sequential that doesn't need to control retry
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    before_agent,
    before_model,
    after_model,
    after_agent,
    AgentState,
)
from langchain.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from typing import Any
import time


# ---------------------------------------------------------------------------
# 1. @before_agent — runs once at the very start
#    Great for: session logging, request ID assignment, input sanitization
# ---------------------------------------------------------------------------
@before_agent
def log_session_start(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    first_msg = state["messages"][0] if state["messages"] else None
    user_text = first_msg.content if hasattr(first_msg, "content") else ""
    print(f"[before_agent] Session started")
    print(f"First user message: {user_text[:80]}")
    return None   # no state change needed


# ---------------------------------------------------------------------------
# 2. @before_model — runs before EACH model call
#    Great for: message count checks, prompt injection detection, logging
# ---------------------------------------------------------------------------
@before_model(can_jump_to=["end"])   # can_jump_to enables early exit
def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    msg_count = len(state["messages"])
    print(f"⬆️  [before_model] Calling model. Messages in context: {msg_count}")

    # Early exit if conversation is too long
    if msg_count >= 50:
        print("[before_model] Message limit reached — stopping agent")
        return {
            "messages": [AIMessage(content="Conversation limit reached. Please start a new session.")],
            "jump_to": "end",  # immediately jump past the model call
        }

    return None   # continue normally


# ---------------------------------------------------------------------------
# 3. @after_model — runs after EACH model response
#    Great for: response logging, content filtering, response transformation
# ---------------------------------------------------------------------------
@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last_msg = state["messages"][-1]
    content = getattr(last_msg, "content", "")
    tool_calls = getattr(last_msg, "tool_calls", [])

    if tool_calls:
        print(f"⬇️  [after_model]  Model made {len(tool_calls)} tool call(s): "
              f"{[tc['name'] for tc in tool_calls]}")
    else:
        preview = str(content)[:100].replace("\n", " ")
        print(f"⬇️  [after_model]  Model responded: {preview}")
    return None


# ---------------------------------------------------------------------------
# 4. @after_agent — runs once at the very end
#    Great for: session summary logging, cleanup, cost reporting
# ---------------------------------------------------------------------------
@after_agent
def log_session_end(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    total_msgs = len(state["messages"])
    # Count AI messages as a proxy for model calls
    ai_msg_count = sum(1 for m in state["messages"]
                       if type(m).__name__ == "AIMessage")
    print(f" [after_agent]  Session ended")
    print(f"   Total messages : {total_msgs}")
    print(f"   Model calls    : {ai_msg_count}")
    return None


# ---------------------------------------------------------------------------
# 5. Wire all hooks into an agent
# ---------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"It's 22°C and sunny in {city}."


agent = create_agent(
    model="gpt-4.1",
    tools=[get_weather],
    middleware=[
        log_session_start,     # @before_agent
        check_message_limit,   # @before_model
        log_response,          # @after_model
        log_session_end,       # @after_agent
    ],
)


# ---------------------------------------------------------------------------
# 6. Run it
# ---------------------------------------------------------------------------
print("=" * 60)
print("Node-style hooks — decorator API")
print("=" * 60)
print()

result = agent.invoke({
    "messages": [HumanMessage(content="What's the weather in Tokyo and Paris?")]
})

print()
print("Final answer:")
print(result["messages"][-1].content[:300])

# ---------------------------------------------------------------------------
# EXPECTED OUTPUT:
#
# [before_agent] Session started
#    First user message: What's the weather in Tokyo and Paris?
# ⬆️  [before_model] Calling model. Messages in context: 1
# ⬇️  [after_model]  Model made 2 tool call(s): ['get_weather', 'get_weather']
# ⬆️  [before_model] Calling model. Messages in context: 4
# ⬇️  [after_model]  Model responded: The weather in Tokyo is 22°C and sunny...
# [after_agent]  Session ended
#    Total messages : 5
#    Model calls    : 2
# ---------------------------------------------------------------------------