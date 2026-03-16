"""
TOPIC: Full Runtime Example — All Three Capabilities Together

This file combines all three runtime capabilities in one realistic agent:
    1. runtime.context      → per-user identity + injected dependencies
    2. runtime.store        → long-term memory (preferences + facts)
    3. runtime.stream_writer → live tool progress events

REALISTIC SCENARIO:
    A multi-tenant research assistant where:
    - Each user has a user_id, name, and plan (context)
    - The agent remembers things users tell it (store)
    - Long-running research tasks stream their progress (stream_writer)
    - The system prompt is personalized per user (dynamic_prompt + context)
    - Model calls are rate-limited per user (before_model + context + store)
"""

from dataclasses import dataclass, field
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    dynamic_prompt,
    before_agent,
    before_model,
    after_agent,
    ModelRequest,
    hook_config,
)
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import tool, ToolRuntime
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Context schema
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ResearchContext:
    user_id:      str
    user_name:    str
    plan:         str          # "free" | "pro"
    locale:       str = "en"
    max_searches: int = 3      # free users get 3, pro users get 20


# ─────────────────────────────────────────────────────────────────────────────
# Middleware using context
# ─────────────────────────────────────────────────────────────────────────────

@dynamic_prompt
def build_system_prompt(request: ModelRequest) -> str:
    """Personalize the system prompt from context + store."""
    ctx: ResearchContext = request.runtime.context
    store = request.runtime.store

    # Load user facts from store to personalize further
    facts_text = ""
    if store:
        item = store.get(("users", "facts"), ctx.user_id)
        if item and item.value.get("facts"):
            facts_text = (
                "\n\nThings I remember about this user:\n"
                + "\n".join(f"- {f}" for f in item.value["facts"])
            )

    lang_instruction = {
        "en": "Respond in English.",
        "fr": "Réponds en français.",
        "es": "Responde en español.",
    }.get(ctx.locale, "Respond in English.")

    plan_note = {
        "free": f"User is on the free plan (max {ctx.max_searches} searches/session).",
        "pro":  "User is on the Pro plan (unlimited searches).",
    }.get(ctx.plan, "")

    return (
        f"You are a helpful research assistant.\n"
        f"User: {ctx.user_name} (ID: {ctx.user_id})\n"
        f"{lang_instruction}\n"
        f"{plan_note}"
        f"{facts_text}"
    )


class SearchLimitMiddleware(AgentMiddleware):
    """Enforce per-user, per-session search limits via context."""

    def __init__(self):
        super().__init__()
        self._session_search_counts: dict[str, int] = {}

    def get_count(self, user_id: str) -> int:
        return self._session_search_counts.get(user_id, 0)

    def increment(self, user_id: str) -> int:
        c = self._session_search_counts.get(user_id, 0) + 1
        self._session_search_counts[user_id] = c
        return c


search_limit_mw = SearchLimitMiddleware()


@before_agent
def log_request(state: AgentState, runtime: Runtime[ResearchContext]) -> dict | None:
    ctx = runtime.context
    print(f"\n▶  [{ctx.user_name}] New request — plan={ctx.plan}, locale={ctx.locale}")
    return None


@after_agent
def log_complete(state: AgentState, runtime: Runtime[ResearchContext]) -> dict | None:
    ctx = runtime.context
    searches = search_limit_mw.get_count(ctx.user_id)
    print(f"⏹  [{ctx.user_name}] Completed — {searches} search(es) this session")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Tools using all three runtime capabilities
# ─────────────────────────────────────────────────────────────────────────────

@tool
def web_search(
    query: str,
    runtime: ToolRuntime[ResearchContext],
) -> str:
    """Search the web for research. Streams search progress."""
    ctx   = runtime.context
    writer = runtime.stream_writer

    # ── Enforce per-user limit (context) ──────────────────────────────────
    count = search_limit_mw.increment(ctx.user_id)
    if count > ctx.max_searches:
        return (
            f"Search limit reached ({ctx.max_searches} searches for {ctx.plan} plan). "
            "Please upgrade to Pro for unlimited searches."
        )

    # ── Stream progress (stream_writer) ────────────────────────────────────
    writer({"event": "search_start", "query": query, "user": ctx.user_name})

    # Simulate fetching from 3 sources
    sources = [f"source_{i+1}.com" for i in range(3)]
    results = []
    for i, src in enumerate(sources, 1):
        result = f"Result from {src}: relevant info about '{query}'"
        results.append(result)
        writer({"event": "source_fetched", "source": src, "step": i, "total": len(sources)})

    writer({"event": "search_done", "query": query, "results_count": len(results)})
    return f"Search results for '{query}':\n" + "\n".join(f"  • {r}" for r in results)


@tool
def remember_fact(
    fact: str,
    runtime: ToolRuntime[ResearchContext],
) -> str:
    """Remember an important fact about the user for future sessions."""
    user_id = runtime.context.user_id
    user_name = runtime.context.user_name

    if not runtime.store:
        return "Memory store not available."

    # ── Read current facts (store) ─────────────────────────────────────────
    item = runtime.store.get(("users", "facts"), user_id)
    existing = item.value.get("facts", []) if item else []

    # ── Write updated facts (store) ────────────────────────────────────────
    runtime.store.put(("users", "facts"), user_id, {"facts": existing + [fact]})
    print(f"  [store] Saved fact for {user_name}: '{fact}'")
    return f"Remembered: '{fact}'"


@tool
def recall_memory(runtime: ToolRuntime[ResearchContext]) -> str:
    """Recall everything remembered about the current user."""
    user_id = runtime.context.user_id

    if not runtime.store:
        return "Memory store not available."

    item = runtime.store.get(("users", "facts"), user_id)
    if not item or not item.value.get("facts"):
        return "Nothing remembered yet."

    facts = item.value["facts"]
    return "Things I remember about you:\n" + "\n".join(f"  • {f}" for f in facts)


# ─────────────────────────────────────────────────────────────────────────────
# Build agent
# ─────────────────────────────────────────────────────────────────────────────
store = InMemoryStore()

agent = create_agent(
    model="gpt-5-nano",
    tools=[web_search, remember_fact, recall_memory],
    context_schema=ResearchContext,
    store=store,
    middleware=[
        build_system_prompt,   # dynamic_prompt with context + store
        log_request,           # before_agent logging
        log_complete,          # after_agent logging
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# Run with custom stream — show all three systems working
# ─────────────────────────────────────────────────────────────────────────────
def run_session(user_message: str, context: ResearchContext) -> None:
    print(f"\n{'─'*60}")
    print(f"User: {context.user_name} | Message: {user_message[:80]}")
    print(f"{'─'*60}")

    for chunk in agent.stream(
        {"messages": [HumanMessage(user_message)]},
        context=context,
        stream_mode=["custom", "updates"],
        version="v2",
    ):
        if chunk["type"] == "custom":
            data = chunk["data"]
            evt  = data.get("event", "")
            if evt == "search_start":
                print(f" Searching: '{data['query']}' for {data['user']}")
            elif evt == "source_fetched":
                print(f"  [{data['step']}/{data['total']}] Fetched {data['source']}")
            elif evt == "search_done":
                print(f"Got {data['results_count']} results")

        elif chunk["type"] == "updates":
            for src, upd in chunk["data"].items():
                if src == "model":
                    last = upd["messages"][-1]
                    if hasattr(last, "content") and last.content:
                        print(f"\n  {last.content[:300]}")


# ─────────────────────────────────────────────────────────────────────────────
# Free user — Alice
# ─────────────────────────────────────────────────────────────────────────────
alice = ResearchContext(
    user_id="u-alice", user_name="Alice",
    plan="free", locale="en", max_searches=2,   # only 2 searches
)

run_session("Search for Python async patterns", alice)
run_session("Remember that I work on ML infrastructure at a fintech company", alice)
run_session("Search for LangGraph best practices", alice)   # 2nd search
run_session("What have you remembered about me?", alice)

# ─────────────────────────────────────────────────────────────────────────────
# Pro user — Carlos (Spanish)
# ─────────────────────────────────────────────────────────────────────────────
carlos = ResearchContext(
    user_id="u-carlos", user_name="Carlos",
    plan="pro", locale="es", max_searches=20,
)
run_session("Busca información sobre bases de datos vectoriales", carlos)