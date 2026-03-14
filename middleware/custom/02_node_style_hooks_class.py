"""
TOPIC: Node-Style Hooks (Class API)

WHEN TO USE CLASSES OVER DECORATORS:
    You need MULTIPLE hooks in one middleware unit
    You need configuration at __init__ time (thresholds, models, API keys)
    You want to share state between before_model and after_model
    You need both sync AND async implementations
    You want reusable middleware to distribute across projects

CLASS STRUCTURE:
    class MyMiddleware(AgentMiddleware):
        def before_agent(self, state, runtime) -> dict | None: ...
        def before_model(self, state, runtime) -> dict | None: ...
        def after_model(self, state, runtime) -> dict | None: ...
        def after_agent(self, state, runtime) -> dict | None: ...

    # Async variants (for async agents):
        async def async_before_model(self, state, runtime) -> dict | None: ...
        async def async_after_model(self, state, runtime) -> dict | None: ...

NOTE ON can_jump_to WITH CLASSES:
    Use the @hook_config decorator to declare jump targets on class methods.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from typing import Any
import time


# ---------------------------------------------------------------------------
# Example 1: Simple audit logger — logs every execution point with timestamps
# ---------------------------------------------------------------------------
class AuditLoggerMiddleware(AgentMiddleware):
    """Logs every execution stage with timing info."""

    def __init__(self, log_prefix: str = "AUDIT"):
        self.log_prefix = log_prefix
        self._start_time: float = 0.0
        self._model_call_times: list[float] = []

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        self._start_time = time.time()
        print(f"[{self.log_prefix}] Agent started at {time.strftime('%H:%M:%S')}")
        return None

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        self._model_call_times.append(time.time())
        call_n = len(self._model_call_times)
        print(f"[{self.log_prefix}] Model call #{call_n} — {len(state['messages'])} messages")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        elapsed = time.time() - self._model_call_times[-1]
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", [])
        print(f"[{self.log_prefix}] Model responded in {elapsed:.2f}s"
              + (f" — called tools: {[tc['name'] for tc in tool_calls]}"
                 if tool_calls else f" — '{str(last.content)[:60]}'"))
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        total = time.time() - self._start_time
        calls = len(self._model_call_times)
        print(f"[{self.log_prefix}] Agent finished — {calls} model call(s) in {total:.2f}s")
        return None


# ---------------------------------------------------------------------------
# Example 2: Call limit guard — configurable via __init__
# ---------------------------------------------------------------------------
class CallLimitMiddleware(AgentMiddleware):
    """
    Stops the agent after a configurable number of model calls.
    Shows how to:
      - configure via __init__
      - share state between before_model and after_model
      - jump to "end" from a class method
    """

    def __init__(self, max_calls: int = 5, warning_at: int = 3):
        self.max_calls = max_calls
        self.warning_at = warning_at
        self._call_count = 0

    @hook_config(can_jump_to=["end"])   # declare allowed jump targets
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        self._call_count += 1

        if self._call_count > self.max_calls:
            print(f"[LIMIT] Reached {self.max_calls} model calls — stopping agent")
            return {
                "messages": [AIMessage(
                    content=f"Agent stopped: exceeded {self.max_calls} model call limit."
                )],
                "jump_to": "end",
            }

        if self._call_count >= self.warning_at:
            remaining = self.max_calls - self._call_count + 1
            print(f"[LIMIT] ⚠️  Call {self._call_count}/{self.max_calls} "
                  f"({remaining} remaining)")
        return None


# ---------------------------------------------------------------------------
# Example 3: Content inspector — checks model output for policy violations
# ---------------------------------------------------------------------------
class ContentPolicyMiddleware(AgentMiddleware):
    """
    Inspects model output and blocks responses containing forbidden words.
    Demonstrates class middleware with configurable rules.
    """

    def __init__(self, forbidden_words: list[str] | None = None):
        self.forbidden_words = [w.lower() for w in (forbidden_words or [])]

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last = state["messages"][-1]
        content = str(getattr(last, "content", "")).lower()

        for word in self.forbidden_words:
            if word in content:
                print(f"[POLICY] Blocked response containing '{word}'")
                # Replace the last message with a safe one
                return {
                    "messages": [AIMessage(content="I can't provide that information.")],
                    "jump_to": "end",
                }
        return None


# ---------------------------------------------------------------------------
# Wire into an agent
# ---------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"22°C and sunny in {city}."


agent = create_agent(
    model="gpt-4.1",
    tools=[get_weather],
    middleware=[
        AuditLoggerMiddleware(log_prefix="DEMO"),    # timing logs
        CallLimitMiddleware(max_calls=5, warning_at=3),  # call cap
        ContentPolicyMiddleware(forbidden_words=["classified", "secret"]),
    ],
)

print("=" * 60)
print("Node-style hooks — class API")
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
# [DEMO] Agent started at 14:22:01
# [DEMO] Model call #1 — 1 messages
# [DEMO] Model responded in 1.23s — called tools: ['get_weather', 'get_weather']
# [DEMO] Model call #2 — 4 messages
# [DEMO] Model responded in 0.89s — 'The weather in Tokyo is 22°C...'
# [DEMO] Agent finished — 2 model call(s) in 2.14s
# ---------------------------------------------------------------------------