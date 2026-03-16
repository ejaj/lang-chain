"""
02_runtime_in_tools.py
========================
TOPIC: Accessing Runtime Inside Tools

WHAT YOU CAN DO IN TOOLS:
    Tools receive a `ToolRuntime[Context]` parameter that gives access to:

        runtime.context       → your custom context data (user_id, db, etc.)
        runtime.store         → BaseStore for long-term memory (read/write)
        runtime.stream_writer → push live updates to the "custom" stream

TOOL SIGNATURE WITH RUNTIME:
    The runtime parameter must be typed as ToolRuntime[YourContextClass].
    LangChain injects it automatically — you never pass it yourself.

    @tool
    def my_tool(arg1: str, runtime: ToolRuntime[MyContext]) -> str:
        ...

    The `runtime` parameter is HIDDEN from the model — the model only sees
    `arg1`. LangChain strips ToolRuntime from the tool's schema so the model
    doesn't try to fill it.

THREE RUNTIME CAPABILITIES:
    1. runtime.context       → dependency injection (user data, API clients)
    2. runtime.store         → persistent memory (read/write across threads)
    3. runtime.stream_writer → real-time progress events (custom stream mode)

WHEN TO USE:
    context       → user identity, per-request configuration, API keys
    store         → user preferences, long-term facts, cross-session memory
    stream_writer → progress bars, intermediate results, tool status updates
"""

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage
from langgraph.store.memory import InMemoryStore


# ---------------------------------------------------------------------------
# Context schema
# ---------------------------------------------------------------------------
@dataclass
class AppContext:
    user_id:  str
    org_id:   str
    api_base: str   # base URL for a hypothetical external API


# ---------------------------------------------------------------------------
# CAPABILITY 1: runtime.context — dependency injection
#    Access user identity and injected dependencies without globals
# ---------------------------------------------------------------------------
@tool
def fetch_user_profile(runtime: ToolRuntime[AppContext]) -> str:
    """Fetch the current user's profile using injected API base URL."""
    ctx = runtime.context
    # In a real app: requests.get(f"{ctx.api_base}/users/{ctx.user_id}")
    print(f"  [tool] Fetching profile from {ctx.api_base}/users/{ctx.user_id}")
    return (
        f"Profile for user {ctx.user_id} (org: {ctx.org_id}):\n"
        f"  Name: John Smith\n"
        f"  Role: Admin\n"
        f"  API: {ctx.api_base}"
    )


# ---------------------------------------------------------------------------
# CAPABILITY 2: runtime.store — long-term memory (read + write)
#    Preferences / facts that persist across multiple agent sessions
# ---------------------------------------------------------------------------
@tool
def get_email_preferences(runtime: ToolRuntime[AppContext]) -> str:
    """Get the user's email writing preferences from long-term memory."""
    user_id = runtime.context.user_id

    if runtime.store:
        # Try to load saved preferences
        memory = runtime.store.get(("users", "email_prefs"), user_id)
        if memory:
            print(f"  [tool] Loaded saved preferences for {user_id}")
            return f"User email preferences: {memory.value['preferences']}"

    # Default if nothing stored yet
    print(f"  [tool] No saved preferences for {user_id}, using default")
    return "User email preferences: Write brief, professional emails."


@tool
def save_email_preferences(
    preferences: str,
    runtime: ToolRuntime[AppContext],
) -> str:
    """Save the user's email writing preferences to long-term memory."""
    user_id = runtime.context.user_id

    if runtime.store:
        runtime.store.put(
            ("users", "email_prefs"),   # namespace tuple
            user_id,                    # key within namespace
            {"preferences": preferences},  # value to store
        )
        print(f"  [tool] Saved preferences for {user_id}")
        return f"Saved your email preferences: '{preferences}'"

    return "Store not available — preferences not saved."


# ---------------------------------------------------------------------------
# CAPABILITY 3: runtime.stream_writer — push live progress events
#    Data pushed here appears in stream_mode="custom" chunks
# ---------------------------------------------------------------------------
@tool
def process_documents(
    count: int,
    runtime: ToolRuntime[AppContext],
) -> str:
    """Process N documents, streaming progress updates."""
    writer = runtime.stream_writer

    results = []
    for i in range(1, count + 1):
        # Simulate document processing
        doc_name = f"document_{i:03d}.pdf"
        results.append(doc_name)

        # Push live progress to the stream
        writer({
            "event":   "progress",
            "current": i,
            "total":   count,
            "file":    doc_name,
            "pct":     int(i / count * 100),
        })

    writer({"event": "complete", "processed": count})
    return f"Processed {count} documents: {results}"


# ---------------------------------------------------------------------------
# Create agent with a store for long-term memory
# ---------------------------------------------------------------------------
store = InMemoryStore()   # use Redis/DynamoDB in production

agent = create_agent(
    model="gpt-5-nano",
    tools=[fetch_user_profile, get_email_preferences, save_email_preferences, process_documents],
    context_schema=AppContext,
    store=store,            
)

context = AppContext(
    user_id="u-alice-42",
    org_id="org-acme",
    api_base="https://api.example.com",
)


# ---------------------------------------------------------------------------
# TEST 1: runtime.context — profile fetch with injected API base
# ---------------------------------------------------------------------------
print("=" * 60)
print("TEST 1: runtime.context — injected API base URL")
print("=" * 60)

r = agent.invoke(
    {"messages": [HumanMessage("Fetch my user profile")]},
    context=context,
)
print(r["messages"][-1].content[:300])


# ---------------------------------------------------------------------------
# TEST 2: runtime.store — save then load preferences across invocations
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 2: runtime.store — preferences persist across invocations")
print("=" * 60)

# First invocation: no preferences saved yet
r1 = agent.invoke(
    {"messages": [HumanMessage("What are my email preferences?")]},
    context=context,
)
print(f"Before save: {r1['messages'][-1].content[:200]}")

# Second invocation: save a preference
r2 = agent.invoke(
    {"messages": [HumanMessage("Save this preference: Always start emails with 'Dear Team,'")]},
    context=context,
)
print(f"After save: {r2['messages'][-1].content[:200]}")

# Third invocation: preference should now be loaded from store
r3 = agent.invoke(
    {"messages": [HumanMessage("What are my email preferences now?")]},
    context=context,
)
print(f"Loaded from store: {r3['messages'][-1].content[:200]}")


# ---------------------------------------------------------------------------
# TEST 3: runtime.stream_writer — live progress events
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 3: runtime.stream_writer — streaming progress events")
print("=" * 60)

for chunk in agent.stream(
    {"messages": [HumanMessage("Process 4 documents")]},
    context=context,
    stream_mode=["updates", "custom"],
    version="v2",
):
    if chunk["type"] == "custom":
        data = chunk["data"]
        if isinstance(data, dict):
            if data.get("event") == "progress":
                bar = "█" * data["current"] + "░" * (data["total"] - data["current"])
                print(f"  [{bar}] {data['pct']}% — {data['file']}")
            elif data.get("event") == "complete":
                print(f"  Complete! {data['processed']} documents processed")
    elif chunk["type"] == "updates":
        for source, upd in chunk["data"].items():
            if source == "model" and upd["messages"][-1].content:
                print(f"  Agent: {upd['messages'][-1].content[:200]}")