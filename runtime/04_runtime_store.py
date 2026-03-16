"""
TOPIC: runtime.store — Long-Term Memory for Agents

WHAT IS THE STORE:
    A BaseStore instance that persists data ACROSS multiple agent invocations.
    Unlike agent state (which lives only for one .invoke() call), the store
    survives across separate calls — even from different users or threads.

    Think of it as a key-value database attached to your agent.

HOW IT DIFFERS FROM STATE:
    Agent state  → lives for ONE .invoke() call, then gone
    Store        → persists indefinitely across all .invoke() calls

STORE API:
    store.put(namespace, key, value_dict)   → write
    store.get(namespace, key)               → read (returns Item | None)
    store.search(namespace, ...)            → search/list
    store.delete(namespace, key)            → delete

    namespace is a TUPLE of strings:  ("users", "prefs")
    key is a STRING:                  "user-42"
    value is a DICT:                  {"name": "Alice", "lang": "en"}

ACCESS PATTERNS:
    In tools:      runtime.store.put(...) / runtime.store.get(...)
    In middleware: runtime.store.put(...) / runtime.store.get(...)
                   (same API, accessed via the runtime parameter)

PRODUCTION STORES:
    InMemoryStore      → dev/testing (data lost on restart)
    RedisStore         → production (persistent, scalable)
    DynamoDB / Postgres → production (persistent, queryable)

WHEN TO USE:
    User preferences (communication style, language, timezone)
    Cross-session facts ("the user mentioned they're allergic to X")
    Usage counters, rate limiting across sessions
    Conversation summaries for returning users
    Agent "memory" — things the agent learned about a user over time
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
class UserContext:
    user_id: str
    user_name: str


# ---------------------------------------------------------------------------
# TOOLS THAT USE THE STORE
# ---------------------------------------------------------------------------

@tool
def remember_preference(
    preference_key: str,
    preference_value: str,
    runtime: ToolRuntime[UserContext],
) -> str:
    """
    Save a user preference to long-term memory.
    Example: remember_preference("tone", "casual and friendly")
    """
    user_id = runtime.context.user_id

    if not runtime.store:
        return "Store not available."

    # namespace=(category, subcategory), key=user_id
    runtime.store.put(
        ("users", "preferences"),   # namespace tuple
        user_id,                    # key
        {preference_key: preference_value},  # value dict
    )
    print(f"  [store] Saved: users/preferences/{user_id} → {preference_key}={preference_value}")
    return f"Remembered: your {preference_key} is '{preference_value}'"


@tool
def recall_preferences(runtime: ToolRuntime[UserContext]) -> str:
    """
    Retrieve all saved preferences for the current user.
    """
    user_id = runtime.context.user_id

    if not runtime.store:
        return "Store not available."

    item = runtime.store.get(("users", "preferences"), user_id)
    if not item:
        print(f"  [store] No preferences found for {user_id}")
        return "No preferences saved yet."

    print(f"  [store] Loaded: users/preferences/{user_id} → {item.value}")
    prefs = "\n".join(f"  {k}: {v}" for k, v in item.value.items())
    return f"Your saved preferences:\n{prefs}"


@tool
def remember_fact(
    fact: str,
    runtime: ToolRuntime[UserContext],
) -> str:
    """
    Remember an important fact about the user for future sessions.
    Example: remember_fact("Alice prefers dark mode and Python over JavaScript")
    """
    user_id = runtime.context.user_id

    if not runtime.store:
        return "Store not available."

    # Load existing facts, append new one
    item = runtime.store.get(("users", "facts"), user_id)
    existing_facts: list[str] = item.value.get("facts", []) if item else []
    updated_facts = existing_facts + [fact]

    runtime.store.put(
        ("users", "facts"),
        user_id,
        {"facts": updated_facts},
    )
    print(f"  [store] Saved fact for {user_id}: '{fact}'")
    return f"Remembered: '{fact}'"


@tool
def recall_facts(runtime: ToolRuntime[UserContext]) -> str:
    """
    Recall all facts remembered about the current user.
    """
    user_id = runtime.context.user_id

    if not runtime.store:
        return "Store not available."

    item = runtime.store.get(("users", "facts"), user_id)
    if not item or not item.value.get("facts"):
        return "No facts remembered yet."

    facts = item.value["facts"]
    formatted = "\n".join(f"  • {f}" for f in facts)
    print(f"  [store] Recalled {len(facts)} facts for {user_id}")
    return f"Facts I remember about you:\n{formatted}"


# ---------------------------------------------------------------------------
# Create agent + store
# ---------------------------------------------------------------------------
store = InMemoryStore()   # replace with RedisStore in production

agent = create_agent(
    model="gpt-5-nano",
    tools=[remember_preference, recall_preferences, remember_fact, recall_facts],
    context_schema=UserContext,
    store=store,   # wire the store in here
)

alice_context = UserContext(user_id="u-alice", user_name="Alice")
bob_context   = UserContext(user_id="u-bob",   user_name="Bob")


# ---------------------------------------------------------------------------
# SESSION 1 (Alice): Save preferences and facts
# ---------------------------------------------------------------------------
print("=" * 60)
print("SESSION 1 — Alice saves her preferences and a fact")
print("=" * 60)

r = agent.invoke(
    {"messages": [HumanMessage(
        "Remember my preference: my tone is 'casual and friendly'. "
        "Also remember the fact that I'm a Python developer who loves async code."
    )]},
    context=alice_context,
)
print(f"Agent: {r['messages'][-1].content[:300]}")


# ---------------------------------------------------------------------------
# SESSION 2 (Alice, DIFFERENT invocation): Recall saved data
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SESSION 2 — Alice returns (new invocation, store persists)")
print("=" * 60)

r2 = agent.invoke(
    {"messages": [HumanMessage(
        "What preferences do you have saved for me? "
        "And what facts do you remember about me?"
    )]},
    context=alice_context,   # same user_id → loads same store data
)
print(f"Agent: {r2['messages'][-1].content[:400]}")


# ---------------------------------------------------------------------------
# SESSION 3 (Bob): Bob has NO saved data — store is user-scoped
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SESSION 3 — Bob (different user, empty store)")
print("=" * 60)

r3 = agent.invoke(
    {"messages": [HumanMessage("What preferences and facts do you have about me?")]},
    context=bob_context,   # different user_id → no data in store
)
print(f"Agent: {r3['messages'][-1].content[:300]}")


# ---------------------------------------------------------------------------
# DIRECT STORE ACCESS (inspection / admin use case)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Direct store inspection (outside agent)")
print("=" * 60)

alice_prefs = store.get(("users", "preferences"), "u-alice")
alice_facts = store.get(("users", "facts"), "u-alice")

print(f"Alice's preferences: {alice_prefs.value if alice_prefs else 'none'}")
print(f"Alice's facts:       {alice_facts.value if alice_facts else 'none'}")

bob_prefs = store.get(("users", "preferences"), "u-bob")
print(f"Bob's preferences:   {bob_prefs.value if bob_prefs else 'none'}")