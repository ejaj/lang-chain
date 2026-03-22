"""
TOPIC: Human-in-the-Loop — Streaming

WHAT IT DOES:
    Same HITL flow as invoke(), but with real-time token streaming.
    You see the model's tokens AS it generates them, then the interrupt fires,
    then after the human decision you see the continuation tokens live.

WHY STREAM:
    For long agent runs, streaming shows progress instead of a blank wait.
    Users see the model "thinking" and planning before the interrupt fires.

STREAM MODES:
    "messages" → individual LLM tokens as they arrive
    "updates"  → agent step completions (model done, tool done, interrupt)

    Use both together: stream_mode=["updates", "messages"]

DETECTING THE INTERRUPT IN STREAM:
    When chunk["type"] == "updates" and "__interrupt__" in chunk["data"]
    → the interrupt has fired, streaming will stop here

RESUMING WITH STREAMING:
    Pass Command(resume=...) to agent.stream() with the same config.
    Streaming continues from where it left off.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt


# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────

def execute_sql(query: str) -> str:
    """Execute a SQL query."""
    print(f"  [TOOL EXECUTED] execute_sql → {query}")
    return f"Query executed: {query}"


def read_table(table: str) -> str:
    """Read a table — safe, no interrupt."""
    return f"Data from {table}: [row1, row2, row3]"


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

agent = create_agent(
    model="gpt-4.1",
    tools=[execute_sql, read_table],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "execute_sql": True,
                "read_table":  False,
            }
        )
    ],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "hitl_stream_01"}}


# ─────────────────────────────────────────────────────────────────────────────
# RUN 1: Stream until interrupt fires
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("RUN 1 — Streaming until interrupt")
print("=" * 60)

collected_interrupts: list[Interrupt] = []

for chunk in agent.stream(
    {"messages": [HumanMessage("Delete old records from the database")]},
    config=config,
    stream_mode=["updates", "messages"],
    version="v2",
):
    if chunk["type"] == "messages":
        # LLM token arriving in real time
        token, metadata = chunk["data"]
        if hasattr(token, "content") and token.content:
            print(token.content, end="", flush=True)

    elif chunk["type"] == "updates":
        data = chunk["data"]

        if "__interrupt__" in data:
            # Interrupt fired — execution is now paused
            interrupts = data["__interrupt__"]
            collected_interrupts.extend(interrupts)
            print(f"\n\n⏸️  INTERRUPTED")
            for interrupt in interrupts:
                for req in interrupt.value["action_requests"]:
                    print(f"  Tool : {req['name']}")
                    print(f"  Args : {req['arguments']}")

print()


# ─────────────────────────────────────────────────────────────────────────────
# Human reviews and decides
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Human reviewing the planned action...")
print("=" * 60)

# In a real app: show the interrupt data to a human in a UI
# Here: auto-approve for demo
print("Decision: APPROVE")


# ─────────────────────────────────────────────────────────────────────────────
# RUN 2: Resume with streaming after the human decision
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("RUN 2 — Resuming with streaming")
print("=" * 60)

for chunk in agent.stream(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,          # same thread_id — resumes from checkpoint
    stream_mode=["updates", "messages"],
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if hasattr(token, "content") and token.content:
            print(token.content, end="", flush=True)

    elif chunk["type"] == "updates":
        data = chunk["data"]
        # Check for tool results
        for source, update in data.items():
            if source == "tools" and "messages" in update:
                last = update["messages"][-1]
                print(f"\n  📤 Tool result: {last.content[:100]}")

print()


# ─────────────────────────────────────────────────────────────────────────────
# FULL STREAMING HELPER — reusable pattern
# ─────────────────────────────────────────────────────────────────────────────

async def stream_with_hitl(agent, messages: list, config: dict) -> list:
    """
    Stream an agent run with HITL support.
    Returns collected interrupts for human review.
    Caller is responsible for resuming after review.
    """
    interrupts = []

    print("Agent streaming...")
    for chunk in agent.stream(
        {"messages": messages},
        config=config,
        stream_mode=["updates", "messages"],
        version="v2",
    ):
        if chunk["type"] == "messages":
            token, _ = chunk["data"]
            if hasattr(token, "content") and token.content:
                print(token.content, end="", flush=True)

        elif chunk["type"] == "updates":
            if "__interrupt__" in chunk["data"]:
                interrupts.extend(chunk["data"]["__interrupt__"])
                print(f"\n⏸️  Paused — {len(interrupts)} action(s) need review")

    return interrupts


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("""
Streaming HITL Quick Reference:

  Run until interrupt:
    for chunk in agent.stream({"messages": [...]}, config=config,
                               stream_mode=["updates", "messages"], version="v2"):
        if chunk["type"] == "messages":
            token, _ = chunk["data"]
            print(token.content, end="")          # live tokens

        elif chunk["type"] == "updates":
            if "__interrupt__" in chunk["data"]:  # interrupt fired
                interrupts = chunk["data"]["__interrupt__"]

  Resume with streaming:
    for chunk in agent.stream(
        Command(resume={"decisions": [...]}),
        config=config,                            # same thread_id
        stream_mode=["updates", "messages"],
        version="v2",
    ):
        ...                                       # same loop structure

  Key point: same config (thread_id) in both stream() calls.
""")