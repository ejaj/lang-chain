"""
TOPIC: Human-in-the-Loop — Execution Lifecycle

WHAT THIS COVERS:
    The exact sequence of events that happen inside HITL middleware —
    what fires when, in what order, and why.

EXECUTION LIFECYCLE (step by step):
    1. Agent calls the model → model generates a response with tool calls
    2. after_model hook fires → HumanInTheLoopMiddleware inspects tool calls
    3. For each tool call: check against interrupt_on policy
       - Matches policy → add to action_requests list
       - No match / False → let it run immediately
    4. If any action_requests exist → build HITLRequest and call interrupt()
    5. Execution pauses → graph state saved to checkpointer
    6. Your code receives the interrupt in result.interrupts
    7. Human reviews action_requests and provides decisions
    8. You call agent.invoke(Command(resume=decisions), ...)
    9. Middleware processes decisions:
       - "approve" → executes the tool as-is
       - "edit"    → executes the tool with modified args
       - "reject"  → synthesizes a ToolMessage with rejection reason
    10. Execution continues from where it left off

WHY after_model:
    The interrupt happens AFTER the model responds but BEFORE any tools run.
    This is the right moment — model has committed to a plan, but nothing
    irreversible has happened yet.

WHAT IS SAVED IN THE CHECKPOINT:
    The entire graph state including:
    - All messages so far
    - The model's proposed tool calls (waiting for human decision)
    - Any custom state fields
    This lets the run pause for hours/days and resume correctly.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented tools — print when they actually execute
# ─────────────────────────────────────────────────────────────────────────────

def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    print(f" [TOOL RUNS] write_file(path={path!r})")
    return f"Written to {path}."


def execute_sql(query: str) -> str:
    """Execute a SQL query."""
    print(f" [TOOL RUNS] execute_sql(query={query[:60]!r})")
    return f"Executed: {query}"


def read_data(table: str) -> str:
    """Read data — safe, skips the interrupt."""
    print(f" [TOOL RUNS] read_data(table={table!r})  ← runs immediately, no interrupt")
    return f"Data from {table}: [row1, row2, row3]"


# ─────────────────────────────────────────────────────────────────────────────
# Agent — showing the full lifecycle
# ─────────────────────────────────────────────────────────────────────────────

checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-4.1",
    tools=[write_file, execute_sql, read_data],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file":  True,
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},
                "read_data":   False,   # safe — runs without interrupt
            },
            description_prefix="Action requires human approval",
        )
    ],
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "lifecycle_demo"}}


# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE STEP 1-5: First invoke — runs until interrupt
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("LIFECYCLE — Steps 1–5: Run until interrupt")
print("=" * 60)
print("""
Step 1: Agent calls model
Step 2: Model plans tool calls
Step 3: after_model hook inspects planned tools
Step 4: Matching tools → HITLRequest → interrupt()
Step 5: State saved to checkpoint, execution paused
""")

result = agent.invoke(
    {"messages": [HumanMessage(
        "Read the users table, then write a summary to report.txt, "
        "then run a cleanup SQL query"
    )]},
    config=config,
    version="v2",
)

print(f"\nInterrupts raised: {len(result.interrupts)}")
for i, interrupt in enumerate(result.interrupts):
    reqs = interrupt.value["action_requests"]
    cfgs = interrupt.value["review_configs"]
    print(f"\nInterrupt {i+1}:")
    for req, cfg in zip(reqs, cfgs):
        print(f"  Tool     : {req['name']}")
        print(f"  Args     : {req['arguments']}")
        print(f"  Allowed  : {cfg['allowed_decisions']}")


# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE STEP 6-7: Human reviews and decides
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("LIFECYCLE — Steps 6–7: Human reviews")
print("=" * 60)
print("""
Step 6: Your code receives result.interrupts
Step 7: Human reviews action_requests and decides
""")

# Build decisions from the interrupt
decisions = []
for req in result.interrupts[0].value["action_requests"]:
    tool = req["name"]
    if tool == "write_file":
        # Edit: change the filename
        decisions.append({
            "type": "edit",
            "edited_action": {
                "name": "write_file",
                "args": {
                    "path":    "summary_v2.txt",    # human changed the filename
                    "content": req["arguments"].get("content", ""),
                },
            },
        })
        print(f"  ✏️  EDIT   write_file → changed path to 'summary_v2.txt'")

    elif tool == "execute_sql":
        # Reject: dangerous query
        decisions.append({
            "type":    "reject",
            "message": "Do not delete data. Use soft deletes (set deleted_at) instead.",
        })
        print(f" REJECT execute_sql → asked to use soft deletes instead")


# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE STEP 8-10: Resume — middleware applies decisions
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("LIFECYCLE — Steps 8–10: Resume, apply decisions, continue")
print("=" * 60)
print("""
Step  8: agent.invoke(Command(resume=decisions), config=config)
Step  9: Middleware processes each decision:
           approve → run tool as-is
           edit    → run tool with modified args
           reject  → add rejection ToolMessage to conversation
Step 10: Execution continues from checkpoint
""")

result2 = agent.invoke(
    Command(resume={"decisions": decisions}),
    config=config,
    version="v2",
)

print(f"\nFinal response:\n{result2.value['messages'][-1].content[:400]}")


# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("""
Lifecycle Summary:

  invoke(messages) ──► model call ──► after_model hook
                                          │
                                    inspect tool calls
                                          │
                              ┌───────────┴───────────┐
                        matches policy          no match / False
                              │                        │
                         interrupt()            run immediately
                              │
                         state saved
                              │
                    result.interrupts ◄── your code reads this
                              │
                       human decides
                              │
                    invoke(Command(resume=decisions))
                              │
                    middleware applies decisions:
                       approve → tool runs as-is
                       edit    → tool runs with new args
                       reject  → ToolMessage with reason
                              │
                    execution continues ──► final response
""")