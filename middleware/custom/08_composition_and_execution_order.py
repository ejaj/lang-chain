"""
TOPIC: Middleware Composition + Execution Order

WHAT IT DOES:
    Shows exactly how multiple middleware layers interact, compose, and
    in what order they execute. Covers both hook ordering rules and how
    ExtendedModelResponse Commands compose when multiple layers use them.

EXECUTION ORDER RULES:
    ┌─────────────────────────────────────────────────────┐
    │ middleware=[A, B, C]                                │
    │                                                     │
    │ before_* hooks:   A → B → C  (first to last)       │
    │ after_*  hooks:   C → B → A  (last to first/reverse)│
    │ wrap_*   hooks:   A wraps (B wraps (C wraps model)) │
    └─────────────────────────────────────────────────────┘

WRAP NESTING VISUALIZED:
    A.wrap_model_call(
        B.wrap_model_call(
            C.wrap_model_call(
                → actual model call ←
            )
        )
    )
    Outer (A) executes first before handler,
    then inner layers, then actual model,
    then inner layers' post-handler code,
    then outer (A) post-handler code.

COMMAND COMPOSITION (ExtendedModelResponse):
    When multiple wrap layers return Commands:
    - messages field → ADDITIVE (all messages appended)
    - other fields   → OUTER WINS (outermost value takes precedence)
    - Retry-safe     → if outer retries, inner commands from failed attempts are discarded

WHEN TO UNDERSTAND THIS:
    Debugging unexpected state values
    Designing middleware that interacts with other middleware
    Understanding which middleware "wins" on key conflicts
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ExtendedModelResponse,
)
from langchain.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing import Any, Callable
from typing_extensions import NotRequired, Annotated


# ---------------------------------------------------------------------------
# Helper: a reducer that last writer wins (for demonstrating outer-wins)
# ---------------------------------------------------------------------------
def _last_wins(_a: str, b: str) -> str:
    return b


# ---------------------------------------------------------------------------
# Custom state showing both additive (messages) and outer-wins (trace_layer)
# ---------------------------------------------------------------------------
class TraceState(AgentState):
    """trace_layer uses last-wins reducer so outer middleware wins."""
    trace_layer: NotRequired[Annotated[str, _last_wins]]
    execution_log: NotRequired[list[str]]


# ---------------------------------------------------------------------------
# Three middleware classes — each logs when they run
# ---------------------------------------------------------------------------
class MiddlewareA(AgentMiddleware):
    """Outermost middleware (index 0 in middleware list)."""

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("  [A] before_agent")
        return None

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("  [A] before_model  ← runs FIRST (first in list)")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("  [A] after_model   ← runs LAST (first in list = last in reverse)")
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("  [A] after_agent")
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ExtendedModelResponse:
        print("  [A] wrap_model_call BEFORE handler  ← outermost, runs first")
        response = handler(request)
        print("  [A] wrap_model_call AFTER  handler  ← outermost, runs last")
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={
                "trace_layer": "outer_A",          # outer wins on conflict
                "messages": [SystemMessage("[A ran]")],  # additive
            }),
        )


class MiddlewareB(AgentMiddleware):
    """Middle middleware (index 1)."""

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("  [B] before_model  ← runs SECOND")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("  [B] after_model   ← runs SECOND (reversed)")
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ExtendedModelResponse:
        print("  [B] wrap_model_call BEFORE handler")
        response = handler(request)
        print("  [B] wrap_model_call AFTER  handler")
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={
                "trace_layer": "inner_B",          # will be OVERWRITTEN by A
                "messages": [SystemMessage("[B ran]")],  # KEPT (additive)
            }),
        )


class MiddlewareC(AgentMiddleware):
    """Innermost middleware (index 2, closest to the model)."""

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("  [C] before_model  ← runs THIRD (last)")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("  [C] after_model   ← runs FIRST (reversed = last in list first)")
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ExtendedModelResponse:
        print("  [C] wrap_model_call BEFORE handler  ← innermost, runs last before model")
        response = handler(request)
        print("  [C] wrap_model_call AFTER  handler  ← innermost, runs first after model")
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={
                "trace_layer": "innermost_C",      # will be OVERWRITTEN by A and B
                "messages": [SystemMessage("[C ran]")],  # KEPT (additive)
            }),
        )


# ---------------------------------------------------------------------------
# Run and observe execution order
# ---------------------------------------------------------------------------
agent = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[MiddlewareA(), MiddlewareB(), MiddlewareC()],   # A=outer, C=inner
)

print("=" * 60)
print("Middleware composition execution order")
print("middleware=[A, B, C]")
print("=" * 60)
print()

print("─── Execution trace ───")
result = agent.invoke({
    "messages": [HumanMessage("Hello")],
    "trace_layer": "",
})

print()
print("─── State results (showing composition) ───")
print(f"trace_layer = '{result.get('trace_layer', '?')}'")
print("  → Expected: 'outer_A'  (outer middleware wins on conflict)")
print()

# Show that ALL [A ran], [B ran], [C ran] messages were added (additive)
injected = [m for m in result["messages"]
            if type(m).__name__ == "SystemMessage"
            and m.content.startswith("[")]
print(f"Injected messages (additive): {[m.content for m in injected]}")
print("  → Expected: ['[C ran]', '[B ran]', '[A ran]']  (inner first, outer last)")


# ---------------------------------------------------------------------------
# Summary diagram printed at end
# ---------------------------------------------------------------------------
print("""
─── Execution Order Summary ───

middleware = [A, B, C]

before_agent:    A → B → C   (first to last)
before_model:    A → B → C   (first to last)

wrap_model_call nesting:
  A.before_handler
    B.before_handler
      C.before_handler
        ← MODEL CALL →
      C.after_handler  (C's command applied first)
    B.after_handler    (B's command applied second)
  A.after_handler      (A's command applied last = A WINS on conflicts)

after_model:     C → B → A   (last to first)
after_agent:     C → B → A   (last to first)

Command composition:
  messages  → ADDITIVE: all messages from all layers are kept
  other keys → OUTER WINS: A overwrites B overwrites C on same key
""")