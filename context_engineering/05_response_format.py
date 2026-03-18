"""
TOPIC: Dynamic Response Format (Structured Output)

WHAT IT IS:
    Instead of getting free text back from the model, you get a validated
    Python object that conforms exactly to a schema you define.
    Field names, types, and descriptions are all guaranteed.

HOW IT WORKS:
    The agent runs the normal model/tool loop until the model is done
    calling tools. Then the final response is coerced into your schema.
    The structured object lives in result["structured_response"].

WHY MAKE IT DYNAMIC:
    Not every situation needs the same output shape.
    Early conversation → simple answer only
    Established conversation → answer + reasoning + confidence
    Admin user → full debug info
    Regular user → clean simple answer

HOW IT WORKS TECHNICALLY:
    Use wrap_model_call + request.override(response_format=MySchema)
    The schema must be a Pydantic BaseModel.
    Field descriptions guide the model — write them carefully.

SCHEMA DESIGN TIPS:
    Use Field(description=...) on every field — these descriptions are
    the instructions the model follows to fill each field.
    Use Literal["a","b","c"] for categorical fields.
    Use Optional[X] for fields that may not always apply.

THREE SELECTION PATTERNS:
    1. From STATE          → conversation length drives schema complexity
    2. From STORE          → user's saved response style preference
    3. From RUNTIME CONTEXT → user role determines schema
"""

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import HumanMessage
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field
from typing import Callable, Literal, Optional


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA DEFINITIONS
# Field descriptions are the instructions the model follows to fill each field.
# Write them precisely — vague descriptions produce vague outputs.
# ─────────────────────────────────────────────────────────────────────────────

# Example of a well-defined static schema
class CustomerSupportTicket(BaseModel):
    """Structured ticket information extracted from a customer message."""
    category: Literal["billing", "technical", "account", "product"] = Field(
        description="Issue category that best describes the customer's problem"
    )
    priority: Literal["low", "medium", "high", "critical"] = Field(
        description="Urgency level based on business impact and customer frustration"
    )
    summary: str = Field(
        description="One-sentence summary of the core issue"
    )
    customer_sentiment: Literal["frustrated", "neutral", "satisfied"] = Field(
        description="The customer's emotional tone in their message"
    )


# Schemas for dynamic selection
class SimpleResponse(BaseModel):
    """Simple single-sentence answer for early conversation."""
    answer: str = Field(description="A brief, direct answer")


class DetailedResponse(BaseModel):
    """Detailed response with reasoning for established conversations."""
    answer:     str   = Field(description="A thorough answer to the question")
    reasoning:  str   = Field(description="Step-by-step explanation of how you arrived at the answer")
    confidence: float = Field(description="Confidence score from 0.0 (unsure) to 1.0 (certain)")
    caveats:    Optional[str] = Field(default=None, description="Important limitations or caveats, if any")


class ConciseResponse(BaseModel):
    """Short answer for users who prefer brevity."""
    answer: str = Field(description="Brief answer in one or two sentences maximum")


class VerboseResponse(BaseModel):
    """Detailed answer with sources for users who prefer depth."""
    answer:  str       = Field(description="Comprehensive, detailed answer")
    sources: list[str] = Field(description="List of sources or reasoning steps used")


class UserResponse(BaseModel):
    """Clean simple answer for regular users."""
    answer: str = Field(description="Plain language answer with no technical jargon")


class AdminResponse(BaseModel):
    """Full technical response for admin users."""
    answer:        str  = Field(description="Complete answer including technical details")
    debug_info:    dict = Field(description="Relevant debug information and internal state")
    system_status: str  = Field(description="Current system status: 'healthy', 'degraded', or 'down'")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 0: Static schema — always the same format
#
# WHY:
#   When the output shape never changes (e.g. a ticket parser, data extractor)
#   just pass the schema directly to create_agent(). No middleware needed.
#
# USE WHEN:
#   The agent always extracts the same fields regardless of context
#   Pipelines that feed structured data into downstream systems
#   Classification tasks with a fixed set of categories
# ─────────────────────────────────────────────────────────────────────────────

ticket_agent = create_agent(
    model="gpt-4.1",
    tools=[],
    response_format=CustomerSupportTicket,   # static — always this schema
)

print("=" * 60)
print("EXAMPLE 0: Static schema — CustomerSupportTicket")
print("=" * 60)

r = ticket_agent.invoke({
    "messages": [{
        "role":    "user",
        "content": "My payment failed 3 times and I still got charged! This is outrageous!",
    }]
})
ticket: CustomerSupportTicket = r["structured_response"]
print(f"Category  : {ticket.category}")
print(f"Priority  : {ticket.priority}")
print(f"Sentiment : {ticket.customer_sentiment}")
print(f"Summary   : {ticket.summary}")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1: Select format from STATE — conversation length
#
# WHY STATE:
#   Message count changes every turn — it lives in state.
#   A new user asking their first question needs a quick, simple answer.
#   An established conversation warrants more depth.
#
# LOGIC:
#   < 3 messages → SimpleResponse  (just an answer, no overhead)
#   3+ messages  → DetailedResponse (answer + reasoning + confidence)
#
# USE WHEN:
#   You want to ease new users in with simple answers
#   Depth and reasoning become more valuable as context grows
#   You want to match output complexity to conversation complexity
# ─────────────────────────────────────────────────────────────────────────────

@wrap_model_call
def state_based_output(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Select output schema based on conversation length from state."""
    message_count = len(request.messages)

    if message_count < 3:
        schema = SimpleResponse
        label  = "SimpleResponse (new conversation)"
    else:
        schema = DetailedResponse
        label  = "DetailedResponse (established conversation)"

    print(f"  [format] {message_count} messages → {label}")
    return handler(request.override(response_format=schema))


agent_state = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[state_based_output],
)

print("\n" + "=" * 60)
print("EXAMPLE 1: Format from STATE — conversation length")
print("=" * 60)

print("\n─── New conversation (1 message) → SimpleResponse ───")
r1 = agent_state.invoke({
    "messages": [HumanMessage("What is photosynthesis?")]
})
simple: SimpleResponse = r1["structured_response"]
print(f"Answer: {simple.answer[:200]}")

print("\n─── Established conversation (5 messages) → DetailedResponse ───")
r2 = agent_state.invoke({
    "messages": [
        {"role": "user",      "content": "Tell me about plants"},
        {"role": "assistant", "content": "Plants are organisms..."},
        {"role": "user",      "content": "How do they get energy?"},
        {"role": "assistant", "content": "Through photosynthesis..."},
        {"role": "user",      "content": "Explain the light reactions in detail"},
    ]
})
detailed: DetailedResponse = r2["structured_response"]
print(f"Answer     : {detailed.answer[:150]}")
print(f"Reasoning  : {detailed.reasoning[:100]}")
print(f"Confidence : {detailed.confidence}")
print(f"Caveats    : {detailed.caveats}")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 2: Select format from STORE — user's style preference
#
# WHY STORE:
#   A user's preference for verbose vs concise responses is a long-term
#   setting — they chose it once and expect it to apply in every future
#   conversation. Store persists across sessions; state does not.
#
# LOGIC:
#   Saved style = "verbose"  → VerboseResponse (answer + sources)
#   Saved style = "concise"  → ConciseResponse (brief answer only)
#   No saved preference      → default ConciseResponse
#
# USE WHEN:
#   Users can configure their preferred output style in account settings
#   Different teams want different output formats (sales vs engineering)
#   Any long-term output preference that should persist across sessions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserContext:
    user_id: str


@wrap_model_call
def store_based_output(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Select output schema based on user's saved preference from store."""
    user_id = request.runtime.context.user_id
    store   = request.runtime.store

    schema = ConciseResponse   # default
    label  = "ConciseResponse (default)"

    if store:
        item = store.get(("preferences",), user_id)
        if item:
            style = item.value.get("response_style", "concise")
            if style == "verbose":
                schema = VerboseResponse
                label  = "VerboseResponse (saved preference)"
            else:
                schema = ConciseResponse
                label  = "ConciseResponse (saved preference)"

    print(f"  [format] {user_id} → {label}")
    return handler(request.override(response_format=schema))


store = InMemoryStore()
store.put(("preferences",), "u-alice", {"response_style": "verbose"})
store.put(("preferences",), "u-bob",   {"response_style": "concise"})

agent_store = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[store_based_output],
    context_schema=UserContext,
    store=store,
)

print("\n" + "=" * 60)
print("EXAMPLE 2: Format from STORE — saved style preference")
print("=" * 60)

print("\n─── Alice (verbose) → VerboseResponse ───")
r_alice = agent_store.invoke(
    {"messages": [HumanMessage("How does HTTPS work?")]},
    context=UserContext(user_id="u-alice"),
)
verbose: VerboseResponse = r_alice["structured_response"]
print(f"Answer  : {verbose.answer[:200]}")
print(f"Sources : {verbose.sources}")

print("\n─── Bob (concise) → ConciseResponse ───")
r_bob = agent_store.invoke(
    {"messages": [HumanMessage("How does HTTPS work?")]},
    context=UserContext(user_id="u-bob"),
)
concise: ConciseResponse = r_bob["structured_response"]
print(f"Answer: {concise.answer[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 3: Select format from RUNTIME CONTEXT — user role
#
# WHY RUNTIME CONTEXT:
#   User role is set at login and fixed for the duration of the session.
#   It's per-request configuration — the right fit for runtime context.
#   Admins need technical debug info; regular users need clean answers.
#
# LOGIC:
#   admin + production → AdminResponse  (answer + debug_info + system_status)
#   everyone else      → UserResponse   (clean plain-language answer)
#
# USE WHEN:
#   Admins need raw technical data; users need simplified output
#   Internal vs external facing versions of the same agent
#   Different output shapes for different API consumers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TenantContext:
    user_role:   str   # "admin" | "user"
    environment: str   # "production" | "staging" | "development"


@wrap_model_call
def context_based_output(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Select output schema based on user role and environment from runtime context."""
    ctx         = request.runtime.context
    user_role   = ctx.user_role
    environment = ctx.environment

    if user_role == "admin" and environment == "production":
        schema = AdminResponse
        label  = "AdminResponse (admin in production)"
    else:
        schema = UserResponse
        label  = "UserResponse (standard)"

    print(f"  [format] role={user_role}, env={environment} → {label}")
    return handler(request.override(response_format=schema))


agent_context = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[context_based_output],
    context_schema=TenantContext,
)

print("\n" + "=" * 60)
print("EXAMPLE 3: Format from RUNTIME CONTEXT — user role")
print("=" * 60)

print("\n─── Admin in production → AdminResponse ───")
r_admin = agent_context.invoke(
    {"messages": [HumanMessage("What is the current system status?")]},
    context=TenantContext(user_role="admin", environment="production"),
)
admin_out: AdminResponse = r_admin["structured_response"]
print(f"Answer        : {admin_out.answer[:150]}")
print(f"System status : {admin_out.system_status}")
print(f"Debug info    : {admin_out.debug_info}")

print("\n─── Regular user → UserResponse ───")
r_user = agent_context.invoke(
    {"messages": [HumanMessage("What is the current system status?")]},
    context=TenantContext(user_role="user", environment="production"),
)
user_out: UserResponse = r_user["structured_response"]
print(f"Answer: {user_out.answer[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Quick Reference — response format")
print("=" * 60)
print("""
  Where result lives:
    result["structured_response"]   ← always here, typed as your schema

  Static (same schema every call):
    create_agent(..., response_format=MySchema)

  Dynamic (schema changes per call):
    @wrap_model_call
    def select_format(request, handler):
        schema = pick_schema(request)
        return handler(request.override(response_format=schema))

  Schema design tips:
    Field(description=...) on every field — model reads these as instructions
    Literal["a","b","c"] for categorical fields — constrains to valid values
    Optional[X] for fields that may not always apply
    Class docstring describes when to use this schema

  Source           │ Read pattern                        │ Use for
  ─────────────────┼─────────────────────────────────────┼──────────────────
  STATE            │ len(request.messages)               │ Conversation stage
  STORE            │ request.runtime.store               │ User style preference
                   │   .get(("namespace",), user_id)     │ (persists forever)
  RUNTIME CONTEXT  │ request.runtime.context.field       │ Role, tier, environment
""")