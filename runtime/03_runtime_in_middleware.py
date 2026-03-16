"""
TOPIC: Accessing Runtime Inside Middleware

HOW RUNTIME REACHES MIDDLEWARE:
    The mechanism depends on hook style:

    NODE-STYLE HOOKS (before_model, after_model, etc.):
        The `runtime` parameter is passed directly as the second argument.
        def before_model(state, runtime: Runtime[Context]) -> dict | None:
            runtime.context.user_id   # ← access context here

    WRAP-STYLE HOOKS (wrap_model_call, wrap_tool_call):
        The `Runtime` lives INSIDE the `ModelRequest` object.
        def wrap_model_call(request, handler):
            request.runtime.context.user_id   # ← access via request.runtime

    DYNAMIC PROMPT (@dynamic_prompt):
        Also comes via ModelRequest:
        def dynamic_system_prompt(request: ModelRequest) -> str:
            request.runtime.context.user_name

WHAT YOU CAN DO IN MIDDLEWARE:
    - Read context to personalize behavior (log user, adjust prompt)
    - Read/write the store for session tracking or preferences
    - Use stream_writer to emit monitoring events
    - Enforce per-user rate limits, quotas, or access controls
    - Build dynamic system prompts from user data

WHEN TO USE CONTEXT IN MIDDLEWARE vs TOOLS:
    Tools     → when the agent DECIDES to use a capability
    Middleware → when you want it to ALWAYS happen regardless of the model
                 (logging, rate limits, dynamic prompts, guardrails)
"""

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    before_agent,
    before_model,
    after_model,
    after_agent,
    wrap_model_call,
    dynamic_prompt,
    ModelRequest,
    ModelResponse,
    hook_config,
)
from langchain.messages import HumanMessage
from langgraph.runtime import Runtime
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Context schema
# ---------------------------------------------------------------------------
@dataclass
class RequestContext:
    user_id:   str
    user_name: str
    locale:    str        # e.g. "en-US", "fr-FR"
    is_admin:  bool = False
    max_calls: int  = 5   # per-user model call limit


# ---------------------------------------------------------------------------
# EXAMPLE 1: @dynamic_prompt — personalize system prompt from context
#    This is the cleanest way to inject per-user personality/instructions.
#    Runs before every model call. Returns a string → becomes the system prompt.
# ---------------------------------------------------------------------------
@dynamic_prompt
def personalized_system_prompt(request: ModelRequest) -> str:
    """Build a personalized system prompt from context."""
    ctx: RequestContext = request.runtime.context

    # Locale → language instruction
    lang_map = {
        "en-US": "Respond in English.",
        "fr-FR": "Réponds en français.",
        "es-ES": "Responde en español.",
        "de-DE": "Antworte auf Deutsch.",
    }
    lang_instruction = lang_map.get(ctx.locale, "Respond in English.")

    # Admin gets extra capabilities in the prompt
    admin_note = (
        "\nYou have full admin access. You may access all data."
        if ctx.is_admin else
        "\nYou have standard user access."
    )

    return (
        f"You are a helpful assistant. "
        f"The user's name is {ctx.user_name} (ID: {ctx.user_id}). "
        f"{lang_instruction}"
        f"{admin_note}"
    )


# ---------------------------------------------------------------------------
# EXAMPLE 2: @before_agent — access context for session-level logging
# ---------------------------------------------------------------------------
@before_agent
def log_session_start(state: AgentState, runtime: Runtime[RequestContext]) -> dict | None:
    """Log the start of each agent session with user context."""
    ctx = runtime.context
    print(f" Session start — user={ctx.user_name} ({ctx.user_id}), "
          f"locale={ctx.locale}, admin={ctx.is_admin}")
    return None


# ---------------------------------------------------------------------------
# EXAMPLE 3: @before_model with context — per-user call limit enforcement
# ---------------------------------------------------------------------------
_user_call_counts: dict[str, int] = {}   # in-memory counter (use Redis in prod)


@before_model(can_jump_to=["end"])
def per_user_call_limit(
    state: AgentState,
    runtime: Runtime[RequestContext],
) -> dict | None:
    """Enforce per-user model call limits using context."""
    from langchain.messages import AIMessage

    ctx = runtime.context
    count = _user_call_counts.get(ctx.user_id, 0) + 1
    _user_call_counts[ctx.user_id] = count

    print(f"   before_model — user={ctx.user_name}, call={count}/{ctx.max_calls}")

    if count > ctx.max_calls:
        print(f" Limit exceeded for {ctx.user_name}")
        return {
            "messages": [AIMessage(
                content=f"You've exceeded your limit of {ctx.max_calls} model calls. "
                        "Please try again later."
            )],
            "jump_to": "end",
        }
    return None


# ---------------------------------------------------------------------------
# EXAMPLE 4: @after_model — log context + response metadata
# ---------------------------------------------------------------------------
@after_model
def log_model_response(
    state: AgentState,
    runtime: Runtime[RequestContext],
) -> dict | None:
    """Log which user received which model response."""
    ctx = runtime.context
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", [])

    if tool_calls:
        print(f"   after_model  — user={ctx.user_name} → "
              f"tools planned: {[tc['name'] for tc in tool_calls]}")
    else:
        preview = str(getattr(last, "content", ""))[:80].replace("\n", " ")
        print(f"   after_model  — user={ctx.user_name} → '{preview}'")
    return None


# ---------------------------------------------------------------------------
# EXAMPLE 5: @after_agent — session summary with context
# ---------------------------------------------------------------------------
@after_agent
def log_session_end(
    state: AgentState,
    runtime: Runtime[RequestContext],
) -> dict | None:
    """Log session end with call count for this user."""
    ctx = runtime.context
    calls = _user_call_counts.get(ctx.user_id, 0)
    print(f"⏹  Session end   — user={ctx.user_name}, total calls this session: {calls}")
    return None


# ---------------------------------------------------------------------------
# EXAMPLE 6: wrap_model_call — access context via request.runtime
# ---------------------------------------------------------------------------
def wrap_model_call(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Log context info from inside a wrap-style hook."""
    ctx: RequestContext = request.runtime.context   # ← via request.runtime
    print(f"   wrap_model    — user={ctx.user_name}, "
          f"messages={len(request.messages)}")
    return handler(request)

from langchain.agents.middleware import wrap_model_call as wrap_decorator
wrap_monitor = wrap_decorator(wrap_model_call)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
from langchain.tools import tool, ToolRuntime

@tool
def get_data(category: str, runtime: ToolRuntime[RequestContext]) -> str:
    """Get data for a category. Admins get more data."""
    ctx = runtime.context
    if ctx.is_admin:
        return f"[ADMIN] Full data for '{category}': all records, all fields"
    return f"[USER] Public data for '{category}': summary only"


# ---------------------------------------------------------------------------
# Create agent with all context-aware middleware
# ---------------------------------------------------------------------------
agent = create_agent(
    model="gpt-5-nano",
    tools=[get_data],
    context_schema=RequestContext,
    middleware=[
        personalized_system_prompt,   # @dynamic_prompt — personalizes every call
        log_session_start,            # @before_agent — session logging
        per_user_call_limit,          # @before_model — per-user call limit
        wrap_monitor,                 # wrap_model_call — request-level logging
        log_model_response,           # @after_model — response logging
        log_session_end,              # @after_agent — session summary
    ],
)


# ---------------------------------------------------------------------------
# Run: two different users, different contexts
# ---------------------------------------------------------------------------
print("=" * 60)
print("Runtime in Middleware — per-user context throughout")
print("=" * 60)

# Standard user (English, no admin)
print("\n─── Standard user (en-US) ───")
_user_call_counts.clear()  # reset for clean demo
r = agent.invoke(
    {"messages": [HumanMessage("Get me data for the products category")]},
    context=RequestContext(
        user_id="u-001",
        user_name="Alice",
        locale="en-US",
        is_admin=False,
        max_calls=5,
    ),
)
print(f"\nFinal: {r['messages'][-1].content[:200]}")

# Admin user (French)
print("\n─── Admin user (fr-FR) ───")
r = agent.invoke(
    {"messages": [HumanMessage("Récupère les données pour la catégorie utilisateurs")]},
    context=RequestContext(
        user_id="u-002",
        user_name="Bernard",
        locale="fr-FR",
        is_admin=True,
        max_calls=5,
    ),
)
print(f"\nFinal: {r['messages'][-1].content[:200]}")