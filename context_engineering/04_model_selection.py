"""
TOPIC: Dynamic Model Selection

WHAT IT IS:
    Swapping which LLM handles each model call based on the current situation —
    without changing any agent logic. The same agent routes to different models
    depending on conversation length, user preferences, or cost tier.

WHY IT MATTERS:
    Not every request needs the same model.
    Using the best model for every call is expensive and slow.
    Using the cheapest model for every call produces poor results.
    Dynamic selection gives you the right model at the right moment.

HOW IT WORKS:
    Use wrap_model_call + request.override(model=chosen_model)
    Initialize all models ONCE at module level — never inside the middleware.

    @wrap_model_call
    def select_model(request, handler):
        model = pick_based_on_something(request)
        return handler(request.override(model=model))

THREE SELECTION PATTERNS:
    1. From STATE          → conversation length drives model choice
    2. From STORE          → user's saved model preference
    3. From RUNTIME CONTEXT → cost tier and environment
"""

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langgraph.store.memory import InMemoryStore
from typing import Callable


# ─────────────────────────────────────────────────────────────────────────────
# Initialize models ONCE at module level — never inside middleware
# Creating a model instance is expensive. Do it once, reuse forever.
# ─────────────────────────────────────────────────────────────────────────────
large_model     = init_chat_model("claude-sonnet-4-6")   # large context window
standard_model  = init_chat_model("gpt-4.1")             # balanced
efficient_model = init_chat_model("gpt-4.1-mini")        # fast and cheap

MODEL_MAP = {
    "gpt-4.1":       standard_model,
    "gpt-4.1-mini":  efficient_model,
    "claude-sonnet": large_model,
}


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1: Select model from STATE — conversation length
#
# WHY STATE:
#   As a conversation grows longer, two things happen:
#     1. Cheap models start struggling — they forget earlier context
#     2. You need a model with a larger context window
#   Message count lives in state — it changes every turn.
#
# LOGIC:
#   < 10 messages  → gpt-4.1-mini  (fast, cheap, short context is fine)
#   10-20 messages → gpt-4.1       (more capable, handles medium context)
#   > 20 messages  → claude-sonnet (large context window, handles long conversations)
#
# USE WHEN:
#   You want to save cost on short conversations
#   Long conversations need a model that won't forget early messages
#   You want quality to scale automatically with conversation complexity
# ─────────────────────────────────────────────────────────────────────────────

@wrap_model_call
def state_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Select model based on conversation length from state."""
    # request.messages is a shortcut for request.state["messages"]
    message_count = len(request.messages)

    if message_count > 20:
        model = large_model
        label = "claude-sonnet-4-6 (long conversation)"
    elif message_count > 10:
        model = standard_model
        label = "gpt-4.1 (medium conversation)"
    else:
        model = efficient_model
        label = "gpt-4.1-mini (short conversation)"

    print(f"  [model] {message_count} messages → {label}")
    return handler(request.override(model=model))


agent_state = create_agent(
    model="gpt-4.1-mini",    # default — overridden by middleware every call
    tools=[],
    middleware=[state_based_model],
)

print("=" * 60)
print("EXAMPLE 1: Model selection from STATE")
print("=" * 60)

print("\n─── Short conversation (2 messages) → gpt-4.1-mini ───")
r = agent_state.invoke({
    "messages": [HumanMessage("What is Python?")]
})
print(f"Response: {r['messages'][-1].content[:200]}")

print("\n─── Medium conversation (12 messages) → gpt-4.1 ───")
r = agent_state.invoke({
    "messages": (
        [{"role": "user", "content": f"Message {i}"} for i in range(11)]
        + [HumanMessage("Summarize what we discussed")]
    )
})
print(f"Response: {r['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 2: Select model from STORE — user preference
#
# WHY STORE:
#   Some users have a preferred model — they set it once in their account
#   settings and expect it to be used in every future conversation.
#   This preference should survive across sessions, so store is correct.
#   State would forget it when the session ends.
#
# LOGIC:
#   Load the user's preferred model from store.
#   If they have a saved preference and it's a valid model, use it.
#   Otherwise fall through to the default.
#
# USE WHEN:
#   You want to let users choose their preferred model
#   Power users who always want the best model
#   Budget-conscious users who always want the cheapest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserContext:
    user_id: str


@wrap_model_call
def store_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Select model based on user's saved preference from store."""
    user_id = request.runtime.context.user_id
    store   = request.runtime.store

    if store:
        item = store.get(("preferences",), user_id)
        if item:
            preferred = item.value.get("preferred_model")
            if preferred and preferred in MODEL_MAP:
                print(f"  [model] {user_id} prefers '{preferred}' → using it")
                return handler(request.override(model=MODEL_MAP[preferred]))

    # No saved preference — use the default model
    print(f"  [model] {user_id} has no preference → using default")
    return handler(request)


# Set up store with user preferences
store = InMemoryStore()
store.put(("preferences",), "u-alice", {"preferred_model": "claude-sonnet"})
store.put(("preferences",), "u-bob",   {"preferred_model": "gpt-4.1-mini"})
# u-carol has no saved preference — will use default

agent_store = create_agent(
    model="gpt-4.1",          # default for users with no preference
    tools=[],
    middleware=[store_based_model],
    context_schema=UserContext,
    store=store,
)

print("\n" + "=" * 60)
print("EXAMPLE 2: Model selection from STORE")
print("=" * 60)

print("\n─── Alice (prefers claude-sonnet) ───")
r = agent_store.invoke(
    {"messages": [HumanMessage("Explain transformers in ML")]},
    context=UserContext(user_id="u-alice"),
)
print(f"Response: {r['messages'][-1].content[:200]}")

print("\n─── Bob (prefers gpt-4.1-mini) ───")
r = agent_store.invoke(
    {"messages": [HumanMessage("Explain transformers in ML")]},
    context=UserContext(user_id="u-bob"),
)
print(f"Response: {r['messages'][-1].content[:200]}")

print("\n─── Carol (no preference → default gpt-4.1) ───")
r = agent_store.invoke(
    {"messages": [HumanMessage("Explain transformers in ML")]},
    context=UserContext(user_id="u-carol"),
)
print(f"Response: {r['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 3: Select model from RUNTIME CONTEXT — cost tier + environment
#
# WHY RUNTIME CONTEXT:
#   Cost tier and environment are per-request configuration — fixed for
#   the duration of one run but different between tenants or deployments.
#   They don't change mid-conversation, so runtime context is the right fit.
#
# LOGIC:
#   production + premium → best model   (paying customers deserve quality)
#   budget tier          → cheapest     (cost control)
#   everything else      → standard     (sensible default)
#
# USE WHEN:
#   SaaS tiers (free / pro / enterprise maps to different models)
#   Dev vs staging vs production environments
#   Per-tenant cost allocation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TenantContext:
    cost_tier:   str   # "premium" | "standard" | "budget"
    environment: str   # "production" | "staging" | "development"


@wrap_model_call
def context_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Select model based on cost tier and environment from runtime context."""
    ctx         = request.runtime.context
    cost_tier   = ctx.cost_tier
    environment = ctx.environment

    if environment == "production" and cost_tier == "premium":
        model = large_model
        label = "claude-sonnet-4-6 (production premium)"
    elif cost_tier == "budget":
        model = efficient_model
        label = "gpt-4.1-mini (budget)"
    else:
        model = standard_model
        label = "gpt-4.1 (standard)"

    print(f"  [model] env={environment}, tier={cost_tier} → {label}")
    return handler(request.override(model=model))


agent_context = create_agent(
    model="gpt-4.1",    # default — overridden by middleware
    tools=[],
    middleware=[context_based_model],
    context_schema=TenantContext,
)

print("\n" + "=" * 60)
print("EXAMPLE 3: Model selection from RUNTIME CONTEXT")
print("=" * 60)

print("\n─── Production + premium → claude-sonnet ───")
r = agent_context.invoke(
    {"messages": [HumanMessage("Analyze our Q3 performance")]},
    context=TenantContext(cost_tier="premium", environment="production"),
)
print(f"Response: {r['messages'][-1].content[:200]}")

print("\n─── Production + budget → gpt-4.1-mini ───")
r = agent_context.invoke(
    {"messages": [HumanMessage("Analyze our Q3 performance")]},
    context=TenantContext(cost_tier="budget", environment="production"),
)
print(f"Response: {r['messages'][-1].content[:200]}")

print("\n─── Staging + standard → gpt-4.1 ───")
r = agent_context.invoke(
    {"messages": [HumanMessage("Analyze our Q3 performance")]},
    context=TenantContext(cost_tier="standard", environment="staging"),
)
print(f"Response: {r['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Quick Reference — dynamic model selection")
print("=" * 60)
print("""
  Source           │ Read pattern                        │ Use for
  ─────────────────┼─────────────────────────────────────┼──────────────────────────
  STATE            │ len(request.messages)               │ Conversation length
                   │ request.state.get("key")            │ Session-specific flags
  ─────────────────┼─────────────────────────────────────┼──────────────────────────
  STORE            │ request.runtime.store               │ User's saved preference
                   │   .get(("namespace",), user_id)     │ (persists across sessions)
  ─────────────────┼─────────────────────────────────────┼──────────────────────────
  RUNTIME CONTEXT  │ request.runtime.context.field       │ Cost tier, environment,
                   │                                     │ plan, tenant config

  Override pattern:
    request = request.override(model=chosen_model)
    return handler(request)

  Important:
    Initialize models ONCE at module level — not inside the middleware
    The default model in create_agent() is the fallback
    override(model=...) affects this call only — transient
""")