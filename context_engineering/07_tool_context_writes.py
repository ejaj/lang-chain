"""
tool_context_writes.py
=======================
TOPIC: Tool Context — Writes

WHAT IT MEANS:
    Tools don't just return a result to the model — they can also
    SAVE data so future tool calls and future conversations can use it.

TWO WRITE TARGETS:
    STATE  → save something for the rest of THIS conversation
             e.g. mark the user as authenticated after login
    STORE  → save something FOREVER across all future conversations
             e.g. save the user's preferred language

HOW TO WRITE:

    TO STATE — return a Command instead of a string:
        return Command(update={"authenticated": True})
        The agent state is updated. A ToolMessage still appears in the conversation.

    TO STORE — call runtime.store.put(...):
        runtime.store.put(("namespace",), key, {"field": value})
        Data is saved to the store immediately.
        Return a normal string to tell the model what happened.
"""

from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command


# ─────────────────────────────────────────────────────────────────────────────
# WRITE 1: To STATE using Command
#
# WHY COMMAND:
#   Some tool results need to change the conversation state, not just
#   return text. Returning a Command tells the agent to update state
#   while still producing a ToolMessage in the conversation history.
#
# HOW:
#   return Command(update={"key": value})
#   Instead of return "some string"
#
# USE WHEN:
#   Logging a user in — set authenticated=True in state
#   Tracking that a step was completed — set step_done=True
#   Recording a decision — set selected_plan="pro"
#   Any state change that should affect future tool calls this session
# ─────────────────────────────────────────────────────────────────────────────

@tool
def authenticate_user(
    password: str,
    runtime: ToolRuntime,
) -> Command:
    """Authenticate the user and update session state."""
    success = (password == "correct123")

    print(f"  [tool] authenticate_user — success={success}")

    # Write to STATE via Command
    # The agent state gets updated AND a ToolMessage appears in the conversation
    return Command(
        update={
            "authenticated": success,
            "auth_user_id":  "u-alice" if success else None,
        }
    )


@tool
def complete_onboarding_step(
    step_name: str,
    runtime: ToolRuntime,
) -> Command:
    """Mark an onboarding step as complete in session state."""
    # Read current completed steps from state
    completed = runtime.state.get("completed_steps", [])
    updated   = completed + [step_name]

    print(f"  [tool] Completed steps: {updated}")

    # Write updated list back to state
    return Command(
        update={"completed_steps": updated}
    )


# ─────────────────────────────────────────────────────────────────────────────
# WRITE 2: To STORE
#
# WHY STORE:
#   Some data should outlive the current conversation.
#   User preferences, learned facts, and history should be there
#   next week when the user comes back. State would forget them.
#   Store keeps them forever.
#
# HOW:
#   runtime.store.put(("namespace",), key, value_dict)
#   Then return a normal string to confirm to the model.
#
# USE WHEN:
#   Saving user preferences (tone, language, format)
#   Remembering facts about the user for future conversations
#   Any data that should survive beyond this session
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserContext:
    user_id: str


@tool
def save_preference(
    preference_key: str,
    preference_value: str,
    runtime: ToolRuntime[UserContext],
) -> str:
    """Save a user preference to long-term memory."""
    user_id = runtime.context.user_id
    store   = runtime.store

    # Read existing preferences first so we don't overwrite others
    item   = store.get(("preferences",), user_id)
    prefs  = item.value if item else {}

    # Merge new preference in
    prefs[preference_key] = preference_value

    # Write to STORE — persists forever
    store.put(("preferences",), user_id, prefs)

    print(f"  [tool] Saved preference for {user_id}: {preference_key}={preference_value}")
    return f"Saved: {preference_key} = {preference_value}"


@tool
def remember_fact(
    fact: str,
    runtime: ToolRuntime[UserContext],
) -> str:
    """Save an important fact about the user to long-term memory."""
    user_id = runtime.context.user_id
    store   = runtime.store

    # Load existing facts
    item   = store.get(("facts",), user_id)
    facts  = item.value.get("facts", []) if item else []

    # Append new fact
    facts.append(fact)

    # Write back to store
    store.put(("facts",), user_id, {"facts": facts})

    print(f"  [tool] Remembered fact for {user_id}: '{fact}'")
    return f"Remembered: '{fact}'"


# ─────────────────────────────────────────────────────────────────────────────
# READ BACK tools — to verify the writes worked
# ─────────────────────────────────────────────────────────────────────────────

@tool
def check_auth(runtime: ToolRuntime) -> str:
    """Check current authentication status from state."""
    is_auth = runtime.state.get("authenticated", False)
    user_id = runtime.state.get("auth_user_id", "unknown")
    return f"Authenticated: {is_auth} (user: {user_id})"


@tool
def recall_preferences(runtime: ToolRuntime[UserContext]) -> str:
    """Recall all saved preferences from the store."""
    user_id = runtime.context.user_id
    item    = runtime.store.get(("preferences",), user_id)
    if not item or not item.value:
        return "No preferences saved yet."
    lines = [f"  {k}: {v}" for k, v in item.value.items()]
    return "Saved preferences:\n" + "\n".join(lines)


@tool
def recall_facts(runtime: ToolRuntime[UserContext]) -> str:
    """Recall all saved facts about the user from the store."""
    user_id = runtime.context.user_id
    item    = runtime.store.get(("facts",), user_id)
    if not item or not item.value.get("facts"):
        return "No facts remembered yet."
    facts = item.value["facts"]
    return "Remembered facts:\n" + "\n".join(f"  - {f}" for f in facts)


# ─────────────────────────────────────────────────────────────────────────────
# Agents and tests
# ─────────────────────────────────────────────────────────────────────────────

# Agent 1: writes to state
agent_state = create_agent(
    model="gpt-4.1",
    tools=[authenticate_user, complete_onboarding_step, check_auth],
)

print("=" * 60)
print("WRITE 1: To STATE via Command")
print("=" * 60)

print("\n─── Login with correct password ───")
r = agent_state.invoke({
    "messages":      [HumanMessage("Login with password 'correct123', then check if I'm authenticated")],
    "authenticated": False,
})
# The Command updated these fields in state
print(f"authenticated in state : {r.get('authenticated')}")
print(f"auth_user_id in state  : {r.get('auth_user_id')}")
print(f"Response: {r['messages'][-1].content[:200]}")

print("\n─── Login with wrong password ───")
r = agent_state.invoke({
    "messages":      [HumanMessage("Login with password 'wrongpass'")],
    "authenticated": False,
})
print(f"authenticated in state : {r.get('authenticated')}")
print(f"Response: {r['messages'][-1].content[:200]}")

print("\n─── Track onboarding steps ───")
r = agent_state.invoke({
    "messages":         [HumanMessage("Complete the 'profile_setup' and 'email_verified' onboarding steps")],
    "completed_steps":  [],
})
print(f"completed_steps in state : {r.get('completed_steps')}")
print(f"Response: {r['messages'][-1].content[:200]}")


# Agent 2: writes to store
store = InMemoryStore()

agent_store = create_agent(
    model="gpt-4.1",
    tools=[save_preference, remember_fact, recall_preferences, recall_facts],
    context_schema=UserContext,
    store=store,
)

alice = UserContext(user_id="u-alice")

print("\n" + "=" * 60)
print("WRITE 2: To STORE")
print("=" * 60)

print("\n─── Session 1: Save preferences and a fact ───")
r1 = agent_store.invoke(
    {"messages": [HumanMessage(
        "Save these preferences: tone=casual, language=English. "
        "Also remember that I am a Python developer at a fintech company."
    )]},
    context=alice,
)
print(f"Response: {r1['messages'][-1].content[:200]}")

print("\n─── Session 2: Recall what was saved (new invocation) ───")
r2 = agent_store.invoke(
    {"messages": [HumanMessage("What preferences and facts do you have saved for me?")]},
    context=alice,   # same user_id → loads same store data
)
print(f"Response: {r2['messages'][-1].content[:300]}")

# Direct store inspection — confirm data was actually saved
print("\n─── Store inspection ───")
prefs = store.get(("preferences",), "u-alice")
facts = store.get(("facts",),       "u-alice")
print(f"Preferences: {prefs.value if prefs else 'none'}")
print(f"Facts:       {facts.value if facts else 'none'}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Quick Reference — tool writes")
print("=" * 60)
print("""
  Target  │ How to write                                  │ Scope
  ────────┼───────────────────────────────────────────────┼────────────────────
  STATE   │ return Command(update={"key": value})         │ This conversation only
          │ (instead of return "string")                  │
  ────────┼───────────────────────────────────────────────┼────────────────────
  STORE   │ runtime.store.put(("namespace",), key, dict)  │ Forever, all sessions
          │ return "confirmation string"                  │

  Command also adds a ToolMessage automatically — the model still
  sees a result, AND state gets updated.

  Store write pattern:
    item  = store.get(("ns",), user_id)          # read first
    data  = item.value if item else {}            # or empty dict
    data["new_key"] = new_value                   # merge
    store.put(("ns",), user_id, data)             # write back
""")