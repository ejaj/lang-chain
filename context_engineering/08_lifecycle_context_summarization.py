"""
TOPIC: Life-cycle Context — Summarization

WHAT IS LIFE-CYCLE CONTEXT:
    Everything that happens BETWEEN the model call and tool call steps.
    Middleware hooks let you intercept data at those junction points to
    implement cross-cutting concerns without changing your agent logic.

TWO THINGS MIDDLEWARE CAN DO:
    1. Update context  → modify state/store to persist changes permanently
    2. Jump in lifecycle → skip steps, repeat calls, exit early

WHAT IS SUMMARIZATION:
    When a conversation gets too long, old messages are compressed into
    a single summary message that replaces them in state — permanently.
    Future turns see the summary, not the original messages.

WHY SUMMARIZATION IS A LIFE-CYCLE CONCERN:
    It is a PERSISTENT state update — it changes saved state forever.
    This is different from transient message trimming (which only affects
    what one model call sees without saving anything).

    Transient trimming   → model sees fewer messages for ONE call, state unchanged
    Summarization        → old messages REPLACED in state, all future turns see summary

BUILT-IN vs CUSTOM:
    SummarizationMiddleware → drop-in, production-ready, just configure it
    Custom middleware       → full control when you need custom logic
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    before_model,
    AgentState,
)
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1: Built-in SummarizationMiddleware
#
# The simplest approach — configure and add to middleware list.
# No custom code needed.
#
# PARAMETERS:
#   model   → which model writes the summary (use a cheap/fast one)
#   trigger → when to fire:
#               ("tokens",   N) → when context exceeds N tokens
#               ("messages", N) → when more than N messages exist
#   keep    → what to preserve verbatim:
#               ("messages", N) → keep last N messages as-is
#               ("tokens",   N) → keep messages fitting within N tokens
#
# WHAT HAPPENS WHEN IT FIRES:
#   1. Takes all messages EXCEPT the last N (keep)
#   2. Sends them to the summary model
#   3. Replaces them with one summary SystemMessage in state (permanent)
#   4. Agent continues — future turns see the summary, not the originals
# ─────────────────────────────────────────────────────────────────────────────

agent = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4.1-mini",        # cheap model writes the summary
            trigger=("tokens", 4000),    # fire when context exceeds 4000 tokens
            keep=("messages", 10),       # always keep last 10 messages verbatim
        ),
    ],
)

print("=" * 60)
print("EXAMPLE 1: Built-in SummarizationMiddleware")
print("=" * 60)
print("""
Configuration:
  model   = gpt-4.1-mini    (cheap — summaries don't need the best model)
  trigger = 4000 tokens     (fire when conversation exceeds this size)
  keep    = 10 messages     (preserve last 10 messages verbatim)

What happens when trigger fires:
  BEFORE: [msg1, msg2, msg3, ..., msg25, msg26, msg27, msg28, msg29, msg30]
  AFTER:  [SUMMARY of msg1-msg20, msg21, msg22, msg23, msg24, msg25, ..., msg30]

  The summary is saved to state permanently.
  Future turns see: [summary, recent messages]
  The original old messages are gone.
""")

# Simulate a long conversation to trigger summarization
messages = []
topics = [
    "Python list comprehensions",
    "dictionary methods",
    "decorators",
    "context managers",
    "async/await",
    "dataclasses",
    "type hints",
    "generators",
]

for topic in topics:
    messages.append({"role": "user",      "content": f"Explain {topic} in Python"})
    messages.append({"role": "assistant", "content": f"{topic} in Python are... [detailed explanation]"})

# Add current question
messages.append(HumanMessage("Summarize everything we've covered so far"))

result = agent.invoke({"messages": messages})

# Check if summarization happened — look for a SystemMessage summary
summary_msgs = [
    m for m in result["messages"]
    if type(m).__name__ == "SystemMessage"
    and "summary" in str(getattr(m, "content", "")).lower()
]
print(f"Messages before : {len(messages)}")
print(f"Messages after  : {len(result['messages'])}")
print(f"Summary found   : {len(summary_msgs) > 0}")
print(f"Response: {result['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 2: Custom summarization middleware
#
# Use this when you need control over:
#   - The trigger condition (e.g. based on message count, not tokens)
#   - The summary format (e.g. bullet points, structured data)
#   - What gets preserved (e.g. always keep tool results)
# ─────────────────────────────────────────────────────────────────────────────

summarize_model = init_chat_model("gpt-4.1-mini")
SUMMARIZE_AFTER = 8    # summarize when more than 8 messages
KEEP_LAST       = 4    # always keep last 4 messages verbatim


@before_model
def custom_summarize(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    """
    PERSISTENT state update:
    When message count exceeds threshold, compress old messages into a summary.
    The summary replaces them in state permanently — all future turns see it.
    """
    messages = state["messages"]

    if len(messages) <= SUMMARIZE_AFTER:
        return None   # not enough messages yet — do nothing

    # Split: recent messages to keep, old messages to summarize
    recent       = messages[-KEEP_LAST:]
    to_summarize = messages[:-KEEP_LAST]

    if not to_summarize:
        return None

    # Build the conversation text to summarize
    history_text = "\n".join(
        f"{m.type.upper()}: {str(getattr(m, 'content', ''))[:300]}"
        for m in to_summarize
    )

    # Call the summary model
    summary_response = summarize_model.invoke([
        {
            "role":    "system",
            "content": (
                "Summarize this conversation history in 3-5 bullet points. "
                "Preserve all key facts, decisions, and topics covered."
            ),
        },
        {
            "role":    "user",
            "content": f"Summarize:\n\n{history_text}",
        },
    ])

    summary_content = str(summary_response.content)
    summary_msg     = SystemMessage(
        content=f"[Conversation Summary]\n{summary_content}"
    )

    # Replace old messages with summary + keep recent messages
    new_messages = [summary_msg] + list(recent)

    print(f"  [summarize] {len(to_summarize)} messages → 1 summary + {len(recent)} recent")
    print(f"  [summarize] {summary_content[:150]}...")

    # PERSISTENT update — saves to state, visible to all future turns
    return {"messages": new_messages}


agent_custom = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[custom_summarize],
)

print("\n" + "=" * 60)
print("EXAMPLE 2: Custom summarization middleware")
print("=" * 60)

long_history = []
for i in range(1, 7):
    long_history.append({"role": "user",      "content": f"Question {i}: Tell me about topic {i}"})
    long_history.append({"role": "assistant", "content": f"Answer {i}: Topic {i} is about... [explanation]"})

long_history.append(HumanMessage("What have we covered so far?"))

r = agent_custom.invoke({"messages": long_history})

print(f"Messages before : {len(long_history)}")
print(f"Messages after  : {len(r['messages'])}")
print(f"Response: {r['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# KEY DISTINCTION — transient trimming vs persistent summarization
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Transient trimming vs Persistent summarization")
print("=" * 60)
print("""
  TRANSIENT TRIMMING (model context):
    Where:  wrap_model_call + request.override(messages=trimmed)
    Effect: model sees fewer messages for ONE call
    State:  UNCHANGED — original messages still in state
    Future: next turn sees full original history again

  PERSISTENT SUMMARIZATION (life-cycle context):
    Where:  before_model returning {"messages": [summary + recent]}
    Effect: old messages REPLACED in state with summary
    State:  CHANGED — summary is now the permanent record
    Future: all future turns see summary, originals are gone

  Use transient trimming when:
    You want to reduce tokens for one call without losing history
    The full history might be needed again later

  Use persistent summarization when:
    The conversation is genuinely too long to keep in full
    Old messages are no longer needed verbatim
    You want to permanently reduce context window usage
""")