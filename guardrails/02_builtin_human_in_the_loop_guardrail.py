"""
TOPIC: Built-in Human-in-the-Loop Guardrail

WHAT IT DOES:
    Pauses the agent BEFORE executing sensitive or irreversible tool calls
    and waits for a human to approve, edit, or reject. This is one of the
    most effective guardrails for high-stakes decisions because a human
    makes the final call — no AI can bypass it.

WHY IT'S A GUARDRAIL:
    Unlike content filters (which catch known bad patterns), HITL catches
    anything the human recognizes as wrong — edge cases, context-specific
    risks, and novel attack vectors that automated rules miss.

BEST FOR:
    Financial transactions (wire transfers, charges)
    Sending communications (email, SMS, Slack)
    Deleting or modifying production data
    Deploying software
    Any action with real-world irreversible consequences

HOW IT WORKS:
    1. Agent plans a tool call
    2. HumanInTheLoopMiddleware interrupts BEFORE execution
    3. Your code shows the planned action to a human (UI, CLI, webhook, etc.)
    4. Human makes a decision: approve / edit / reject
    5. You call agent.stream(Command(resume=decisions), ...) to continue

DECISIONS:
    {"type": "approve"}                              → run as-is
    {"type": "edit", "edited_action": {...}}         → run with changed args
    {"type": "reject"}                               → skip tool, inform agent

REQUIRES:
    A checkpointer to save state between the pause and resume.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt


# ---------------------------------------------------------------------------
# Tools — safe (search) and sensitive (send_email, delete_database)
# ---------------------------------------------------------------------------
def search_tool(query: str) -> str:
    """Search for information — safe, no approval needed."""
    return f"Search results for '{query}': [result 1, result 2, result 3]"


def send_email_tool(to: str, subject: str, body: str) -> str:
    """Send an email — irreversible, requires approval."""
    print(f" [EXECUTED] Sending email to {to}: '{subject}'")
    return f"Email sent to {to}."


def delete_database_tool(table: str, condition: str) -> str:
    """Delete database records — highly dangerous, requires approval."""
    print(f" [EXECUTED] DELETE FROM {table} WHERE {condition}")
    return f"Deleted records from {table}."


# ---------------------------------------------------------------------------
# Agent — approve send_email and delete_database, auto-allow search
# ---------------------------------------------------------------------------
checkpointer = InMemorySaver()   # use Redis/Postgres in production

agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool, send_email_tool, delete_database_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # Sensitive: require human decision with all options
                "send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                # Dangerous: require human decision for delete
                "delete_database_tool": {
                    "allowed_decisions": ["approve", "reject"],
                    # no "edit" here — we don't allow editing delete conditions
                },
                # Safe: never interrupt
                "search_tool": False,
            }
        ),
    ],
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "guardrail_demo_01"}}


# ---------------------------------------------------------------------------
# RUN 1 — Agent plans tool calls, pauses at sensitive ones
# ---------------------------------------------------------------------------
print("=" * 60)
print("RUN 1 — Agent streams until interrupt")
print("=" * 60)

collected_interrupts: list[Interrupt] = []

for chunk in agent.stream(
    {"messages": [{
        "role": "user",
        "content": (
            "Search for our latest product info, then email the team at "
            "team@company.com with a summary, subject 'Product Update'."
        ),
    }]},
    config=config,
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source == "__interrupt__":
                for interrupt in update:
                    collected_interrupts.append(interrupt)
                    print(f"\n INTERRUPTED — approval required:")
                    for req in interrupt.value["action_requests"]:
                        print(f"   Tool : {req.get('name', '?')}")
                        print(f"   Args : {req.get('args', {})}")
    elif chunk["type"] == "messages":
        token, _ = chunk["data"]
        if hasattr(token, "tool_call_chunks") and token.tool_call_chunks:
            pass   # don't clutter output with raw chunks


# ---------------------------------------------------------------------------
# HUMAN REVIEW — simulate a human making decisions
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Human reviewing planned tool calls...")
print("=" * 60)

decisions = {}
for interrupt in collected_interrupts:
    action_decisions = []
    for req in interrupt.value["action_requests"]:
        tool_name = req.get("name", "")
        args = req.get("args", {})

        if tool_name == "send_email_tool":
            # Edit: correct the email body before approving
            action_decisions.append({
                "type": "edit",
                "edited_action": {
                    "name": "send_email_tool",
                    "args": {
                        "to": args.get("to", ""),
                        "subject": args.get("subject", ""),
                        "body": "Hi team, here's the product update summary. [REVIEWED BY HUMAN]",
                    },
                },
            })
            print(f"  ✏️  EDITED email body before sending to {args.get('to')}")

        elif tool_name == "delete_database_tool":
            # Reject: this is too dangerous
            action_decisions.append({"type": "reject"})
            print(f" REJECTED delete operation on table: {args.get('table')}")

        else:
            action_decisions.append({"type": "approve"})
            print(f" APPROVED: {tool_name}")

    decisions[interrupt.id] = {"decisions": action_decisions}


# ---------------------------------------------------------------------------
# RUN 2 — Resume with human decisions
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RUN 2 — Resuming with human decisions")
print("=" * 60)

for chunk in agent.stream(
    Command(resume=decisions),
    config=config,
    stream_mode=["updates"],
    version="v2",
):
    if chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source == "tools":
                last = update["messages"][-1]
                print(f" Tool result: {last.content[:100]}")
            elif source == "model":
                last = update["messages"][-1]
                if hasattr(last, "content") and last.content:
                    print(f"\n Agent: {last.content[:400]}")

# ---------------------------------------------------------------------------
# EXPECTED OUTPUT:
#
# RUN 1:
#   INTERRUPTED — approval required:
#      Tool : send_email_tool
#      Args : {to: 'team@company.com', subject: 'Product Update', body: '...'}
#
# Human reviewing:
#   ✏️  EDITED email body
#
# RUN 2:
#   Tool result: Email sent to team@company.com.
#   Agent: I found the product info and sent the team an edited email summary.
# ---------------------------------------------------------------------------