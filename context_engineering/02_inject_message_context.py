"""
inject_message_context_examples.py
=====================================
TOPIC: Injecting Messages from All Three Data Sources

PATTERN:
    @wrap_model_call
    def inject_something(request, handler):
        1. Read from a data source  (state / store / runtime context)
        2. Build a message from that data
        3. request.override(messages=[...new list...])   ← transient, not saved
        4. return handler(request)

WHY APPEND AT THE END:
    All three examples add the injected message AFTER existing messages.
    Models weight recent content more heavily — appending context just before
    the model responds makes it more likely to actually be used.

    [user, assistant, user, assistant, ..., current user, >>> INJECTED <<<]

TRANSIENT vs PERSISTENT:
    request.override(messages=...) → TRANSIENT — model sees it, state does NOT change
    return {"messages": [...]}     → PERSISTENT — saved to state for all future turns

THREE EXAMPLES:
    1. From STATE          → uploaded files this session
    2. From STORE          → user's saved writing style (long-term memory)
    3. From RUNTIME CONTEXT → compliance rules for this tenant/jurisdiction
"""

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import HumanMessage
from langgraph.store.memory import InMemoryStore
from typing import Callable


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1: Inject from STATE — uploaded file context
#
# WHY STATE:
#   Uploaded files belong to THIS session only.
#   State is short-term memory — available throughout the conversation,
#   gone when the session ends.
#
# USE WHEN:
#   File Q&A agents ("summarize my PDF")
#   Multi-file workflows where model needs to know what's available
#   Any per-session data the user provides at conversation start
# ─────────────────────────────────────────────────────────────────────────────

@wrap_model_call
def inject_file_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """
    Read uploaded_files from state and inject a file listing message.

    State field:
        uploaded_files: list[{name, type, summary}]
    """
    # ── Read from STATE ────────────────────────────────────────────────────
    uploaded_files: list[dict] = request.state.get("uploaded_files", [])

    if not uploaded_files:
        return handler(request)   # nothing to inject — pass through

    # ── Build context message ──────────────────────────────────────────────
    file_lines = [
        f"- {f['name']} ({f['type']}): {f['summary']}"
        for f in uploaded_files
    ]
    file_context = (
        "Files you have access to in this conversation:\n"
        + "\n".join(file_lines)
        + "\n\nReference these files when answering questions."
    )

    # ── Append after existing messages (transient — not saved to state) ────
    messages = [
        *request.messages,
        {"role": "user", "content": file_context},
    ]
    request = request.override(messages=messages)

    print(f"  [inject_files] Injected {len(uploaded_files)} file(s)")
    return handler(request)


agent_files = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[inject_file_context],
)

print("=" * 60)
print("EXAMPLE 1: Inject from STATE — uploaded files")
print("=" * 60)

result = agent_files.invoke({
    "messages": [HumanMessage("What documents do I have and what are they about?")],
    "uploaded_files": [
        {"name": "Q3_report.pdf",  "type": "PDF",  "summary": "Q3 financial results showing 12% revenue growth"},
        {"name": "contracts.docx", "type": "DOCX", "summary": "Three client contracts expiring in March"},
        {"name": "users.csv",      "type": "CSV",  "summary": "Export of 4,200 active user accounts"},
    ],
})
print(f"Response: {result['messages'][-1].content[:300]}\n")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 2: Inject from STORE — writing style
#
# WHY STORE:
#   Writing style is long-term memory — the user set it once and it
#   should be remembered across ALL future conversations forever.
#   Store persists across sessions; state does not.
#
# USE WHEN:
#   Email / writing assistants that mimic the user's voice
#   Any user preference that should persist beyond one conversation
#   Learned facts about the user you want to inject into every call
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WritingContext:
    user_id: str


@wrap_model_call
def inject_writing_style(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """
    Read the user's writing style from the store and inject it as context.

    Store namespace: ("writing_style",)
    Store key:       user_id
    Store value:     {tone, greeting, sign_off, example_email}
    """
    user_id = request.runtime.context.user_id
    store   = request.runtime.store

    style_item = store.get(("writing_style",), user_id) if store else None

    if not style_item:
        return handler(request)   # no saved style — pass through

    # ── Build style guide from stored data ────────────────────────────────
    style = style_item.value
    style_context = (
        "Your writing style:\n"
        f"- Tone: {style.get('tone', 'professional')}\n"
        f"- Typical greeting: \"{style.get('greeting', 'Hi')}\"\n"
        f"- Typical sign-off: \"{style.get('sign_off', 'Best')}\"\n"
        f"- Example email you've written:\n{style.get('example_email', '')}"
    )

    # ── Append at end — models pay more attention to final messages ────────
    messages = [
        *request.messages,
        {"role": "user", "content": style_context},
    ]
    request = request.override(messages=messages)

    print(f"  [inject_style] Injected writing style for {user_id}")
    return handler(request)


# Set up store and pre-seed two users' writing styles
store = InMemoryStore()

store.put(("writing_style",), "u-alice", {
    "tone":          "warm and direct",
    "greeting":      "Hey team",
    "sign_off":      "Cheers",
    "example_email": "Hey team,\nJust a quick update — we're shipping Friday.\nLet me know if you have blockers.\nCheers, Alice",
})
store.put(("writing_style",), "u-bob", {
    "tone":          "formal and structured",
    "greeting":      "Dear colleagues",
    "sign_off":      "Kind regards",
    "example_email": "Dear colleagues,\nI am writing to inform you of the upcoming deployment scheduled for Friday, 14:00 UTC.\nPlease ensure all outstanding tasks are completed beforehand.\nKind regards, Bob",
})

agent_style = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[inject_writing_style],
    context_schema=WritingContext,
    store=store,
)

print("=" * 60)
print("EXAMPLE 2: Inject from STORE — writing style")
print("=" * 60)

print("─── Alice (warm, direct) ───")
r_alice = agent_style.invoke(
    {"messages": [HumanMessage("Write an email to the team about the project delay")]},
    context=WritingContext(user_id="u-alice"),
)
print(f"{r_alice['messages'][-1].content[:300]}\n")

print("─── Bob (formal, structured) ───")
r_bob = agent_style.invoke(
    {"messages": [HumanMessage("Write an email to the team about the project delay")]},
    context=WritingContext(user_id="u-bob"),
)
print(f"{r_bob['messages'][-1].content[:300]}\n")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 3: Inject from RUNTIME CONTEXT — compliance rules
#
# WHY RUNTIME CONTEXT:
#   Compliance requirements are static config for this request — they depend
#   on the tenant's jurisdiction, industry, and active frameworks.
#   They don't change during the run but differ between tenants.
#   Runtime context is the right place for per-request, per-tenant config.
#
# USE WHEN:
#   Multi-tenant SaaS where each org has different legal requirements
#   Region-specific behavior (EU vs US vs APAC)
#   Any config that's static for one request but varies between requests
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComplianceContext:
    user_jurisdiction:     str          # e.g. "EU", "US", "APAC"
    industry:              str          # e.g. "healthcare", "finance", "retail"
    compliance_frameworks: list[str]    # e.g. ["GDPR", "HIPAA"]


@wrap_model_call
def inject_compliance_rules(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """
    Build compliance rules from runtime context and inject as a message.

    Context fields: user_jurisdiction, industry, compliance_frameworks
    """
    ctx          = request.runtime.context
    jurisdiction = ctx.user_jurisdiction
    industry     = ctx.industry
    frameworks   = ctx.compliance_frameworks

    # ── Build rules based on active frameworks + industry ─────────────────
    rules = []

    if "GDPR" in frameworks:
        rules.append("Must obtain explicit consent before processing personal data")
        rules.append("Users have the right to data deletion upon request")
        rules.append("Data breaches must be reported within 72 hours")

    if "HIPAA" in frameworks:
        rules.append("Cannot share patient health information without authorization")
        rules.append("Must use secure, encrypted communication channels")
        rules.append("Audit logs required for all PHI access")

    if "SOC2" in frameworks:
        rules.append("All data access must be logged and auditable")
        rules.append("Security incidents must be reported to the compliance team")

    if industry == "finance":
        rules.append("Cannot provide specific financial advice without proper disclaimers")
        rules.append("All transactions above $10,000 require enhanced due diligence")

    if industry == "healthcare":
        rules.append("Always recommend consulting a qualified medical professional")
        rules.append("Do not diagnose conditions — provide general health information only")

    if not rules:
        return handler(request)   # no rules apply — pass through

    # ── Build and inject compliance context message ────────────────────────
    compliance_context = (
        f"Compliance requirements for {jurisdiction} ({industry}):\n"
        + "\n".join(f"- {r}" for r in rules)
        + "\n\nYou must adhere to these requirements in every response."
    )

    # ── Append at end — models pay more attention to final messages ────────
    messages = [
        *request.messages,
        {"role": "user", "content": compliance_context},
    ]
    request = request.override(messages=messages)

    print(f"  [inject_compliance] Injected {len(rules)} rule(s) for {jurisdiction}/{industry}")
    return handler(request)


agent_compliance = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[inject_compliance_rules],
    context_schema=ComplianceContext,
)

print("=" * 60)
print("EXAMPLE 3: Inject from RUNTIME CONTEXT — compliance rules")
print("=" * 60)

print("─── EU healthcare (GDPR + HIPAA) ───")
r_eu = agent_compliance.invoke(
    {"messages": [HumanMessage("How should we store patient email addresses?")]},
    context=ComplianceContext(
        user_jurisdiction="EU",
        industry="healthcare",
        compliance_frameworks=["GDPR", "HIPAA"],
    ),
)
print(f"{r_eu['messages'][-1].content[:300]}\n")

print("─── US finance (SOC2 only) ───")
r_us = agent_compliance.invoke(
    {"messages": [HumanMessage("Can we recommend specific stocks to our users?")]},
    context=ComplianceContext(
        user_jurisdiction="US",
        industry="finance",
        compliance_frameworks=["SOC2"],
    ),
)
print(f"{r_us['messages'][-1].content[:300]}\n")


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED — all three in one agent
# Each middleware runs in sequence, each injecting its context message.
# Final message list seen by model:
#   [conversation history..., file context, style context, compliance context]
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FullContext:
    user_id:               str
    user_jurisdiction:     str
    industry:              str
    compliance_frameworks: list[str]


# Reuse the same store from above
agent_full = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        inject_file_context,       # from state
        inject_writing_style,      # from store
        inject_compliance_rules,   # from runtime context
    ],
    context_schema=FullContext,
    store=store,
)

print("=" * 60)
print("COMBINED — all three middleware layers")
print("=" * 60)

result_full = agent_full.invoke(
    {
        "messages": [HumanMessage(
            "Draft an email to our legal team summarizing the uploaded compliance doc"
        )],
        "uploaded_files": [
            {"name": "compliance_policy.pdf", "type": "PDF",
             "summary": "Internal data handling policy updated Q1 2025"},
        ],
    },
    context=FullContext(
        user_id="u-alice",
        user_jurisdiction="EU",
        industry="healthcare",
        compliance_frameworks=["GDPR", "HIPAA"],
    ),
)
print(f"Response: {result_full['messages'][-1].content[:400]}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Quick Reference — inject message context")
print("=" * 60)
print("""
  Source           │ How to read                              │ When to use
  ─────────────────┼──────────────────────────────────────────┼────────────────────────
  STATE            │ request.state.get("key", default)        │ This-session data
                   │ request.messages                         │ (uploads, auth status)
  ─────────────────┼──────────────────────────────────────────┼────────────────────────
  STORE            │ request.runtime.store                    │ Cross-session data
                   │   .get(("namespace",), user_id)          │ (preferences, history)
  ─────────────────┼──────────────────────────────────────────┼────────────────────────
  RUNTIME CONTEXT  │ request.runtime.context.field            │ Per-request config
                   │                                          │ (tenant, role, locale)

  Override pattern:
    messages = [*request.messages, {"role": "user", "content": "...injected..."}]
    request  = request.override(messages=messages)
    return handler(request)    # transient — model sees it, state does NOT change
""")