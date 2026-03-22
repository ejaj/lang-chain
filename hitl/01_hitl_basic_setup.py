"""
TOPIC: Human-in-the-Loop — Basic Setup and Configuration

WHAT IT IS:
    Middleware that pauses agent execution before running sensitive tool calls
    and waits for a human to approve, edit, or reject them.

WHY IT EXISTS:
    Some tool calls are irreversible — deleting database records, sending emails,
    writing files. An agent might plan the right action but get the details wrong.
    HITL puts a human checkpoint before execution so mistakes can be caught.

HOW IT WORKS:
    1. Agent calls the model → model proposes tool calls
    2. HumanInTheLoopMiddleware inspects each tool call against interrupt_on policy
    3. If a tool matches → interrupt raised, execution paused, state saved
    4. Human reviews the action_requests in the interrupt
    5. Human sends decisions back via Command(resume=...)
    6. Execution continues: approved tools run, rejected ones get feedback messages

THREE DECISION TYPES:
    approve → run the tool exactly as planned
    edit    → run the tool but with modified arguments
    reject  → skip the tool, add explanation to conversation

REQUIRES:
    A checkpointer — saves graph state so execution can pause and resume.
    Use InMemorySaver() for dev/testing.
    Use AsyncPostgresSaver() or similar for production.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────

def write_file(path: str, content: str) -> str:
    """Write content to a file on disk."""
    print(f"  [TOOL EXECUTED] write_file: {path}")
    return f"Written to {path} successfully."


def execute_sql(query: str) -> str:
    """Execute a SQL query against the database."""
    print(f"  [TOOL EXECUTED] execute_sql: {query}")
    return f"Query executed: {query}"


def read_data(table: str) -> str:
    """Read data from a table. Safe — no approval needed."""
    print(f"  [TOOL EXECUTED] read_data: {table}")
    return f"Data from {table}: [row1, row2, row3]"


# ─────────────────────────────────────────────────────────────────────────────
# Agent setup
# ─────────────────────────────────────────────────────────────────────────────

agent = create_agent(
    model="gpt-4.1",
    tools=[write_file, execute_sql, read_data],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # All three decisions allowed (approve, edit, reject)
                "write_file": True,

                # Only approve or reject — no editing the SQL
                "execute_sql": {
                    "allowed_decisions": ["approve", "reject"],
                },

                # Never interrupt — safe read operation
                "read_data": False,
            },

            # Prefix for the interrupt description shown to the reviewer
            # Full message: "Tool execution pending approval\n\nTool: execute_sql\nArgs: {...}"
            description_prefix="Tool execution pending approval",
        ),
    ],

    # Required — saves state between the interrupt and resume
    checkpointer=InMemorySaver(),
)


# ─────────────────────────────────────────────────────────────────────────────
# Run — agent pauses at the interrupt
# ─────────────────────────────────────────────────────────────────────────────

# Thread ID ties the paused run to its resumed run
config = {"configurable": {"thread_id": "hitl_demo_01"}}

print("=" * 60)
print("HITL Basic Setup — initial run")
print("=" * 60)

result = agent.invoke(
    {"messages": [HumanMessage("Delete old records from the database")]},
    config=config,
    version="v2",       # returns GraphOutput with .value and .interrupts
)

print(f"Interrupts: {len(result.interrupts)}")

for interrupt in result.interrupts:
    print(f"\nInterrupt ID: {interrupt.id}")
    for req in interrupt.value["action_requests"]:
        print(f"  Tool        : {req['name']}")
        print(f"  Arguments   : {req['arguments']}")
        print(f"  Description : {req['description'][:80]}...")
    for cfg in interrupt.value["review_configs"]:
        print(f"  Allowed decisions: {cfg['allowed_decisions']}")


# ─────────────────────────────────────────────────────────────────────────────
# Resume with approve
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Resuming with APPROVE decision")
print("=" * 60)

result2 = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,   # same thread_id — resumes from saved checkpoint
    version="v2",
)

print(f"Final response: {result2.value['messages'][-1].content[:300]}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("""
HITL Configuration Quick Reference:

  interrupt_on options:
    "tool_name": True                              → all decisions allowed
    "tool_name": {"allowed_decisions": ["approve", "reject"]} → specific decisions only
    "tool_name": False                             → never interrupt

  Always required:
    checkpointer=InMemorySaver()    (dev)
    checkpointer=AsyncPostgresSaver() (production)

  Always required when invoking:
    config = {"configurable": {"thread_id": "unique_id"}}

  result.interrupts → tuple of Interrupt objects
  result.value      → final agent state dict
""")