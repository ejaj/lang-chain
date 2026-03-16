"""
05_layered_guardrail_stack.py
==============================
TOPIC: Layered Guardrail Stack (All Guardrail Types Combined)

WHAT IS A LAYERED GUARDRAIL STACK:
    Production agents don't use a single guardrail — they use MULTIPLE layers
    working together. Each layer catches what the previous missed:

    Layer 1: Deterministic input filter  → catch known patterns instantly, free
    Layer 2: PII protection              → sanitize sensitive data before/after model
    Layer 3: Human approval              → human oversight for irreversible actions
    Layer 4: Model-based output check    → catch semantic safety issues in final answer

    The key insight: CHEAP layers first, EXPENSIVE layers last.
    Most bad requests are caught by Layer 1 before they cost anything.

EXECUTION FLOW:
    User input
        ↓
    [Layer 1] Keyword/injection check (before_agent, <1ms, free)
        ↓ passes
    [Layer 2] PII redaction (before model, <5ms, free)
        ↓
    [Model call] Main agent model
        ↓
    [Layer 2] PII redaction on output (after model, <5ms, free)
        ↓
    [Layer 3] Human approval for sensitive tools (interrupt + resume)
        ↓
    [Layer 4] Safety evaluation of final response (after_agent, ~500ms, ~$0.001)
        ↓
    User sees clean, safe, approved response

COST ANALYSIS (per request):
    Layer 1: $0.000   — pure Python regex
    Layer 2: $0.000   — pure Python regex
    Layer 3: $0.000   — human decision (no model cost for the guardrail itself)
    Layer 4: ~$0.001  — cheap safety model (gpt-4.1-mini, ~200 tokens)

    vs. not having guardrails: $∞ in liability, compliance fines, reputation damage

WHEN TO USE EACH COMBINATION:
    Consumer app       → Layer 1 + 2 + 4
    Enterprise app     → Layer 1 + 2 + 3 + 4
    Healthcare/Finance → Layer 1 + 2 + 3 + 4 + additional compliance layers
    Internal tool      → Layer 1 + 2 (lighter weight)
"""

import re
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    before_agent,
    after_agent,
    AgentState,
    hook_config,
    PIIMiddleware,
    HumanInTheLoopMiddleware,
)
from langchain.messages import AIMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.types import Command, Interrupt
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1: Deterministic content filter
# ─────────────────────────────────────────────────────────────────────────────
BANNED_KEYWORDS = ["hack", "exploit", "malware", "bomb", "weapon", "drug synthesis"]
INJECTION_PATTERNS = [
    re.compile(r"ignore (all |your )?(previous |prior )?instructions", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"bypass (your |the )?safety", re.I),
]


class ContentFilterMiddleware(AgentMiddleware):
    """
    LAYER 1: Fast deterministic input filter.
    Catches keyword violations and prompt injections before any model call.
    Cost: $0. Speed: <1ms.
    """

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        first_msg = state["messages"][0]
        if first_msg.type != "human":
            return None

        content = str(first_msg.content).lower()

        # Check keyword blacklist
        for kw in self.banned_keywords:
            if kw in content:
                print(f" L1] BLOCKED — keyword: '{kw}'")
                return {
                    "messages": [AIMessage(
                        content="I cannot process requests containing restricted content."
                    )],
                    "jump_to": "end",
                }

        # Check injection patterns
        for pattern in INJECTION_PATTERNS:
            if pattern.search(content):
                print(f" [L1] BLOCKED — prompt injection detected")
                return {
                    "messages": [AIMessage(
                        content="I detected an attempt to override my instructions."
                    )],
                    "jump_to": "end",
                }

        print(f"[L1] Content filter passed")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4: Model-based safety evaluator
# ─────────────────────────────────────────────────────────────────────────────
class SafetyGuardrailMiddleware(AgentMiddleware):
    """
    LAYER 4: Semantic safety check on the final response.
    Uses a cheap model to catch subtle issues Layer 1 missed.
    Cost: ~$0.001 per check. Speed: ~300-500ms.
    """

    def __init__(self, safety_model_name: str = "gpt-4.1-mini"):
        super().__init__()
        self.safety_model = init_chat_model(safety_model_name)

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.content:
            return None

        # Skip if already replaced by a previous guardrail
        if "cannot process" in str(last_msg.content).lower():
            return None

        verdict = self.safety_model.invoke([{
            "role": "system",
            "content": "Safety evaluator. Respond ONLY with 'SAFE' or 'UNSAFE'.",
        }, {
            "role": "user",
            "content": f"Is this AI response safe and appropriate?\n\n{last_msg.content}",
        }])

        verdict_text = str(verdict.content).strip().upper()
        print(f" [L4] Safety verdict: {verdict_text}")

        if "UNSAFE" in verdict_text:
            print(f" [L4] BLOCKED — model-based safety check")
            last_msg.content = (
                "I cannot provide that response. Please rephrase your request."
            )

        return None


# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────
def search_tool(query: str) -> str:
    """Search for information — safe."""
    return f"Search results: [relevant info about '{query}']"


def send_email_tool(to: str, subject: str, body: str) -> str:
    """Send an email — requires human approval."""
    print(f" [EXECUTED] Email → {to}: {subject}")
    return f"Email sent to {to}."


# ─────────────────────────────────────────────────────────────────────────────
# Full layered guardrail stack
# ─────────────────────────────────────────────────────────────────────────────
checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool, send_email_tool],
    checkpointer=checkpointer,
    middleware=[
        # ── LAYER 1: Deterministic input filter ──────────────────────────
        ContentFilterMiddleware(banned_keywords=BANNED_KEYWORDS),

        # ── LAYER 2: PII protection (input + output) ──────────────────────
        PIIMiddleware("email",       strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask",   apply_to_input=True),
        PIIMiddleware("email",       strategy="redact", apply_to_input=False,
                      apply_to_output=True),

        # ── LAYER 3: Human-in-the-loop for sensitive actions ──────────────
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {"allowed_decisions": ["approve", "edit", "reject"]},
                "search_tool": False,
            }
        ),

        # ── LAYER 4: Model-based output safety check ──────────────────────
        SafetyGuardrailMiddleware(safety_model_name="gpt-4.1-mini"),
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# Test the full stack
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("FULL LAYERED GUARDRAIL STACK")
print("=" * 60)

# ── Test A: Blocked by Layer 1 (keyword) ─────────────────────────────────
print("\n─── Test A: Layer 1 blocks harmful keyword ───")
r = agent.invoke({"messages": [HumanMessage("How do I hack a website?")]})
print(f"Response: {r['messages'][-1].content}")

# ── Test B: PII redacted by Layer 2 ──────────────────────────────────────
print("\n─── Test B: Layer 2 redacts PII before model sees it ───")
config_b = {"configurable": {"thread_id": "test_b"}}
collected: list[Interrupt] = []
for chunk in agent.stream(
    {"messages": [HumanMessage(
        "My email is alice@company.com. Please search for the latest news."
    )]},
    config=config_b,
    stream_mode=["updates"],
    version="v2",
):
    if chunk["type"] == "updates":
        for source, upd in chunk["data"].items():
            if source == "model":
                last = upd["messages"][-1]
                if hasattr(last, "content") and last.content:
                    print(f"  Agent: {last.content[:200]}")
print("  (Model never saw 'alice@company.com' — it was redacted to [REDACTED_EMAIL])")

# ── Test C: Layer 3 interrupts for email approval ─────────────────────────
print("\n─── Test C: Layer 3 interrupts for human approval ───")
config_c = {"configurable": {"thread_id": "test_c"}}
collected = []
for chunk in agent.stream(
    {"messages": [HumanMessage("Search for product updates, then email team@corp.com with a summary")]},
    config=config_c,
    stream_mode=["updates"],
    version="v2",
):
    if chunk["type"] == "updates":
        for source, upd in chunk["data"].items():
            if source == "__interrupt__":
                for intr in upd:
                    collected.append(intr)
                    print(f" Interrupted — tool: {intr.value['action_requests'][0].get('name')}")

# Auto-approve for demo
if collected:
    decisions = {i.id: {"decisions": [{"type": "approve"}]} for i in collected}
    for chunk in agent.stream(
        Command(resume=decisions),
        config=config_c,
        stream_mode=["updates"],
        version="v2",
    ):
        if chunk["type"] == "updates":
            for source, upd in chunk["data"].items():
                if source == "model":
                    last = upd["messages"][-1]
                    if hasattr(last, "content") and last.content:
                        print(f"  Agent (after approval): {last.content[:200]}")

print("\n" + "=" * 60)
print("LAYER SUMMARY")
print("=" * 60)
print("  L1 Deterministic filter  → $0,    <1ms,    catches keywords + injection")
print("  L2 PII redaction         → $0,    <5ms,    sanitizes PII before/after model")
print("  L3 Human-in-the-loop     → $0,    human,   irreversible action approval")
print("  L4 Model-based safety    → $0.001, ~400ms, semantic safety check on output")