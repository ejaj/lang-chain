"""
TOPIC: @dynamic_prompt — All Three Data Sources

WHAT IS @dynamic_prompt:
    A middleware decorator that builds the system prompt dynamically
    before EVERY model call. Instead of a static string, you write a
    function that reads live data and returns the right prompt for
    the current moment.

    @dynamic_prompt replaces (or extends) the system_prompt= you pass
    to create_agent(). It runs before every model call in the loop.

THREE DATA SOURCES:
    1. STATE          → request.messages / request.state["key"]
                        Conversation history, auth status, uploaded files
                        Scope: this conversation only

    2. STORE          → request.runtime.store.get(("ns",), key)
                        Saved user preferences, learned facts, summaries
                        Scope: cross-conversation (persists forever)

    3. RUNTIME CONTEXT → request.runtime.context.field
                         User identity, role, environment, API config
                         Scope: this invocation (passed at invoke time)

FUNCTION SIGNATURE:
    @dynamic_prompt
    def my_prompt(request: ModelRequest) -> str:
        ...
        return "Your system prompt string"

    Takes:  ModelRequest
    Returns: str   ← the full system prompt to use for this call

HOW TO ACCESS EACH SOURCE:
    request.messages                  → list of messages (shortcut for request.state["messages"])
    request.state.get("key", default) → any other state field
    request.runtime.store             → BaseStore instance (or None)
    request.runtime.context           → your context dataclass instance
"""

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.messages import HumanMessage
from langgraph.store.memory import InMemoryStore


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1: Dynamic prompt from STATE
#   Reads the current conversation history length and auth status.
#   Adapts instructions based on where we are in the conversation.
# ─────────────────────────────────────────────────────────────────────────────

@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    """
    Adapt the system prompt based on STATE:
      - message_count: how long the conversation has been going
      - authenticated: whether the user has logged in (another state field)

    request.messages is a shortcut for request.state["messages"]
    """
    # Read from State
    message_count = len(request.messages)                             # shortcut
    authenticated = request.state.get("authenticated", False)         # other state field
    uploaded_files = request.state.get("uploaded_files", [])          # yet another field

    base = "You are a helpful assistant."

    # Adapt based on conversation length
    if message_count > 10:
        base += "\nThis is a long conversation — be extra concise. No repetition."
    elif message_count > 5:
        base += "\nConversation is in progress — stay focused on the current topic."
    else:
        base += "\nNew conversation — be welcoming and thorough."

    # Adapt based on auth status
    if authenticated:
        base += "\nThe user is authenticated. You may discuss their account details."
    else:
        base += "\nThe user has not authenticated. Do not reveal any private information."

    # Adapt based on uploaded files
    if uploaded_files:
        names = ", ".join(f['name'] for f in uploaded_files)
        base += f"\nThe user has uploaded: {names}. Reference these files when relevant."

    return base


# Wire into agent
agent_state = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[state_aware_prompt],
    # No context_schema needed — state is always available
)

print("=" * 60)
print("EXAMPLE 1: State-aware prompt")
print("=" * 60)

# New, unauthenticated user
r1 = agent_state.invoke({
    "messages":      [HumanMessage("What can you help me with?")],
    "authenticated": False,
    "uploaded_files": [],
})
print(f"New user:  {r1['messages'][-1].content[:200]}\n")

# Authenticated user with a file
r2 = agent_state.invoke({
    "messages":      [HumanMessage("Summarize my uploaded document")],
    "authenticated": True,
    "uploaded_files": [{"name": "Q3_report.pdf", "type": "PDF"}],
})
print(f"Auth+file: {r2['messages'][-1].content[:200]}\n")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 2: Dynamic prompt from STORE
#   Reads long-term user preferences saved from previous sessions.
#   The user set their communication style once — it's remembered forever.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserContext:
    user_id: str


@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    """
    Personalize the system prompt from STORE (long-term memory):
      - communication_style: verbose | balanced | concise
      - expertise_level:     beginner | intermediate | expert
      - known_facts:         list of things we know about this user
    """
    user_id   = request.runtime.context.user_id   # get user_id from context
    store     = request.runtime.store             # access the store

    base = "You are a helpful assistant."

    if store:
        item = store.get(("preferences",), user_id)
        if item:
            prefs = item.value

            # Communication style
            style = prefs.get("communication_style", "balanced")
            style_instructions = {
                "verbose":   "Give comprehensive, detailed explanations with examples.",
                "balanced":  "Balance detail and brevity based on the question.",
                "concise":   "Keep all answers as brief as possible — bullet points preferred.",
            }
            base += f"\n{style_instructions.get(style, '')}"

            # Expertise level
            level = prefs.get("expertise_level", "intermediate")
            level_instructions = {
                "beginner":     "Use simple language. Avoid jargon. Explain everything.",
                "intermediate": "Assume basic familiarity. Explain advanced concepts.",
                "expert":       "Use technical terminology freely. Skip basic explanations.",
            }
            base += f"\n{level_instructions.get(level, '')}"

            # Known facts about the user
            facts = prefs.get("known_facts", [])
            if facts:
                base += "\n\nWhat you know about this user:\n"
                base += "\n".join(f"- {f}" for f in facts)

    return base


# Set up store and pre-seed user preferences
store = InMemoryStore()
store.put(("preferences",), "u-alice", {
    "communication_style": "concise",
    "expertise_level":     "expert",
    "known_facts": [
        "Works as a senior data engineer",
        "Prefers Python over SQL",
        "Allergic to marketing buzzwords",
    ],
})
store.put(("preferences",), "u-bob", {
    "communication_style": "verbose",
    "expertise_level":     "beginner",
    "known_facts": [
        "Just started learning programming",
        "Background in accounting",
    ],
})

agent_store = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[store_aware_prompt],
    context_schema=UserContext,
    store=store,                    # wire in the store
)

print("=" * 60)
print("EXAMPLE 2: Store-aware prompt (long-term preferences)")
print("=" * 60)

# Alice: expert, concise — should get short technical answer
r_alice = agent_store.invoke(
    {"messages": [HumanMessage("What is a database index?")]},
    context=UserContext(user_id="u-alice"),
)
print(f"Alice (expert/concise): {r_alice['messages'][-1].content[:250]}\n")

# Bob: beginner, verbose — should get a thorough explanation
r_bob = agent_store.invoke(
    {"messages": [HumanMessage("What is a database index?")]},
    context=UserContext(user_id="u-bob"),
)
print(f"Bob (beginner/verbose): {r_bob['messages'][-1].content[:250]}\n")

# Unknown user: no saved prefs — should use defaults
r_unknown = agent_store.invoke(
    {"messages": [HumanMessage("What is a database index?")]},
    context=UserContext(user_id="u-new-user"),
)
print(f"Unknown (defaults):     {r_unknown['messages'][-1].content[:200]}\n")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 3: Dynamic prompt from RUNTIME CONTEXT
#   Reads role, environment, and other per-request config.
#   These are set at invoke time and never change during the run.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeploymentContext:
    user_role:      str   # "admin" | "analyst" | "viewer"
    deployment_env: str   # "development" | "staging" | "production"
    tenant_name:    str   # company name for branding


@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    """
    Build instructions from RUNTIME CONTEXT:
      - user_role:      controls what the user is allowed to do
      - deployment_env: controls caution level
      - tenant_name:    controls branding
    """
    ctx = request.runtime.context

    # ── Role-based instructions ─────────────────────────────────────────
    role_instructions = {
        "admin":   "You have ADMIN access. You can perform all read and write operations.",
        "analyst": "You have ANALYST access. You can read and analyze data, but cannot modify it.",
        "viewer":  "You have VIEWER access. You can only view public summaries. Never reveal raw data.",
    }
    role_note = role_instructions.get(ctx.user_role, "You have standard access.")

    # ── Environment-based caution ───────────────────────────────────────
    env_notes = {
        "development": "This is a DEVELOPMENT environment. Mistakes are safe here.",
        "staging":     "This is a STAGING environment. Changes may affect tests.",
        "production":  "This is PRODUCTION. Be extremely careful with any modifications. Always confirm before destructive actions.",
    }
    env_note = env_notes.get(ctx.deployment_env, "")

    # ── Tenant branding ────────────────────────────────────────────────
    brand_note = f"You are an assistant for {ctx.tenant_name}."

    return (
        f"{brand_note}\n"
        f"Access level: {role_note}\n"
        f"Environment: {env_note}"
    )


agent_context = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[context_aware_prompt],
    context_schema=DeploymentContext,
)

print("=" * 60)
print("EXAMPLE 3: Runtime context-aware prompt")
print("=" * 60)

# Admin in production — should be capable but cautious
r_admin_prod = agent_context.invoke(
    {"messages": [HumanMessage("Can you delete the old test records?")]},
    context=DeploymentContext(
        user_role="admin",
        deployment_env="production",
        tenant_name="Acme Corp",
    ),
)
print(f"Admin/Production: {r_admin_prod['messages'][-1].content[:250]}\n")

# Viewer in dev — should be restricted and relaxed
r_viewer_dev = agent_context.invoke(
    {"messages": [HumanMessage("Show me the raw user data")]},
    context=DeploymentContext(
        user_role="viewer",
        deployment_env="development",
        tenant_name="Globex Inc",
    ),
)
print(f"Viewer/Development: {r_viewer_dev['messages'][-1].content[:250]}\n")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 4: COMBINED — all three sources in one prompt
#   Production pattern: state (session) + store (history) + context (identity)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FullContext:
    user_id:   str
    user_name: str
    role:      str
    locale:    str   # "en" | "fr" | "es"


@dynamic_prompt
def combined_prompt(request: ModelRequest) -> str:
    """
    Production-grade dynamic prompt drawing from all three data sources.

    From STATE:   conversation length → conciseness instruction
    From STORE:   saved preferences   → tone and detail level
    From CONTEXT: user identity       → name, role, locale
    """
    ctx   = request.runtime.context
    store = request.runtime.store

    # ── From STATE ─────────────────────────────────────────────────────
    msg_count    = len(request.messages)
    concise_note = " Be concise." if msg_count > 8 else ""

    # ── From STORE ─────────────────────────────────────────────────────
    tone_note   = ""
    detail_note = ""
    facts_note  = ""
    if store:
        item = store.get(("preferences",), ctx.user_id)
        if item:
            prefs = item.value
            tone  = prefs.get("tone", "professional")
            tone_note = f" Use a {tone} tone."
            if prefs.get("detail") == "high":
                detail_note = " Provide thorough explanations."
            facts = prefs.get("known_facts", [])
            if facts:
                facts_note = "\n\nKnown about user:\n" + "\n".join(f"- {f}" for f in facts)

    # ── From RUNTIME CONTEXT ───────────────────────────────────────────
    lang_map = {"en": "English", "fr": "French", "es": "Spanish"}
    lang     = lang_map.get(ctx.locale, "English")

    role_map = {
        "admin":   "full access to all operations",
        "analyst": "read-only data access",
        "viewer":  "public summaries only",
    }
    role_cap = role_map.get(ctx.role, "standard access")

    return (
        f"You are a helpful assistant for {ctx.user_name} ({ctx.role}, {role_cap}).\n"
        f"Always respond in {lang}.{tone_note}{detail_note}{concise_note}"
        f"{facts_note}"
    )


store_full = InMemoryStore()
store_full.put(("preferences",), "u-diana", {
    "tone":        "friendly",
    "detail":      "high",
    "known_facts": ["Senior engineer at a startup", "Loves concise bullet-point answers"],
})

agent_combined = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[combined_prompt],
    context_schema=FullContext,
    store=store_full,
)

print("=" * 60)
print("EXAMPLE 4: Combined — all three sources")
print("=" * 60)

r_combined = agent_combined.invoke(
    {"messages": [HumanMessage("Explain REST vs GraphQL")]},
    context=FullContext(
        user_id="u-diana", user_name="Diana",
        role="analyst", locale="en",
    ),
)
print(f"Response: {r_combined['messages'][-1].content[:400]}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("@dynamic_prompt — Quick Reference")
print("=" * 60)
print("""
  Data source       │ Access pattern
  ──────────────────┼────────────────────────────────────────
  State (messages)  │ request.messages
  State (other)     │ request.state.get("key", default)
  Store             │ request.runtime.store.get(("ns",), key)
  Runtime Context   │ request.runtime.context.field_name

  Signature:
    @dynamic_prompt
    def my_prompt(request: ModelRequest) -> str:
        return "system prompt string"

  Registered as:
    create_agent(..., middleware=[my_prompt])
""")