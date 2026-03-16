"""
01_runtime_context_basics.py
==============================
TOPIC: Runtime Context — What It Is and Why It Exists

WHAT IS THE RUNTIME:
    When an agent runs, LangGraph creates a `Runtime` object that carries
    three things throughout the entire execution:

        runtime.context      → your custom dependency injection data
        runtime.store        → a BaseStore for long-term memory
        runtime.stream_writer → writes to the "custom" stream mode

    Think of it as a "request-scoped container" — like a Flask/FastAPI
    request object, but for agent execution.

WHAT IS CONTEXT:
    Context is static, read-only configuration you pass at invoke time.
    It travels to EVERY tool and EVERY middleware hook during that run.

    Use it for:
        User identity (user_id, user_name, roles)
        Tenant/org info (org_id, plan, feature flags)
        Database connections or API clients
        Per-request configuration (language, timezone, preferences)

WHY NOT GLOBAL VARIABLES:
    Global state breaks when:
      - Two requests run concurrently (race conditions)
      - You want to test a tool in isolation (can't mock global state easily)
      - You deploy serverless / multi-process (globals don't survive)

    Context solves this: each invoke() gets its own isolated context.

DEFINING CONTEXT:
    Use @dataclass or Pydantic BaseModel.
    Pass context_schema=YourClass to create_agent().
    Pass context=YourClass(...) to agent.invoke() or agent.stream().

HOW IT FLOWS:
    agent.invoke(..., context=Context(user_id="u1"))
         ↓
    every tool call:       runtime.context.user_id == "u1"
    every middleware hook: runtime.context.user_id == "u1"
"""

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage


# ---------------------------------------------------------------------------
# 1. Define your context schema
#    Use @dataclass (or Pydantic BaseModel) — both work
# ---------------------------------------------------------------------------
@dataclass
class UserContext:
    """
    Per-request context for a multi-tenant agent.
    Everything here is available in every tool and middleware hook.
    """
    user_id:   str
    user_name: str
    org_id:    str
    plan:      str   # "free" | "pro" | "enterprise"
    language:  str = "en"


# ---------------------------------------------------------------------------
# 2. Define tools that READ from context via ToolRuntime[Context]
# ---------------------------------------------------------------------------
@tool
def greet_user(runtime: ToolRuntime[UserContext]) -> str:
    """Greet the current user by name."""
    # Access context via runtime.context
    name = runtime.context.user_name
    lang = runtime.context.language
    greeting = {
        "en": f"Hello, {name}!",
        "es": f"¡Hola, {name}!",
        "fr": f"Bonjour, {name}!",
    }.get(lang, f"Hello, {name}!")
    return greeting


@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Return account information for the current user."""
    ctx = runtime.context
    return (
        f"Account info:\n"
        f"  User ID   : {ctx.user_id}\n"
        f"  Name      : {ctx.user_name}\n"
        f"  Org       : {ctx.org_id}\n"
        f"  Plan      : {ctx.plan}\n"
        f"  Language  : {ctx.language}"
    )


@tool
def check_feature_access(feature: str, runtime: ToolRuntime[UserContext]) -> str:
    """Check if the current user has access to a feature based on their plan."""
    ctx = runtime.context
    # Simple plan-based access control
    pro_features = {"advanced_analytics", "export_csv", "api_access"}
    enterprise_features = {"sso", "audit_logs", "custom_models"}

    if feature in enterprise_features:
        has_access = ctx.plan == "enterprise"
    elif feature in pro_features:
        has_access = ctx.plan in ("pro", "enterprise")
    else:
        has_access = True   # free features

    status = "GRANTED" if has_access else "DENIED (upgrade required)"
    return f"Feature '{feature}' for plan '{ctx.plan}': {status}"


# ---------------------------------------------------------------------------
# 3. Create agent — register context_schema
# ---------------------------------------------------------------------------
agent = create_agent(
    model="gpt-5-nano",
    tools=[greet_user, get_account_info, check_feature_access],
    context_schema=UserContext,   # 👈 tells create_agent what context shape to expect
)


# ---------------------------------------------------------------------------
# 4. Invoke — pass context per request
#    Each .invoke() call is fully isolated — no shared state between users
# ---------------------------------------------------------------------------
print("=" * 60)
print("Runtime Context Basics")
print("=" * 60)

# Request from User A (free plan, English)
print("\n─── User A: Alice, free plan ───")
result_alice = agent.invoke(
    {"messages": [HumanMessage("Greet me, show my account, and check if I can use api_access")]},
    context=UserContext(
        user_id="u001",
        user_name="Alice",
        org_id="org-acme",
        plan="free",
        language="en",
    ),
)
print(result_alice["messages"][-1].content[:400])

# Request from User B (enterprise plan, French)
print("\n─── User B: Bob, enterprise plan, French ───")
result_bob = agent.invoke(
    {"messages": [HumanMessage("Greet me, then check if I can use sso and api_access")]},
    context=UserContext(
        user_id="u002",
        user_name="Bob",
        org_id="org-globex",
        plan="enterprise",
        language="fr",
    ),
)
print(result_bob["messages"][-1].content[:400])

# ---------------------------------------------------------------------------
# 5. Key point: same agent code, fully isolated contexts
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Both users used the SAME agent object.")
print("Context was isolated per .invoke() call — no shared state.")
print("Alice never saw Bob's user_id, and vice versa.")
print("=" * 60)

# ---------------------------------------------------------------------------
# EXPECTED OUTPUT:
#
# User A: Alice, free plan
#   Hello, Alice! Your account... plan: free
#   api_access:  DENIED (upgrade required)
#
# User B: Bob, enterprise plan, French
#   Bonjour, Bob! ...
#   sso:  GRANTED    api_access: GRANTED
# ---------------------------------------------------------------------------