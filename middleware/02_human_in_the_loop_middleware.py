"""
TOPIC: HumanInTheLoopMiddleware

WHAT IT DOES:
    Pauses the agent BEFORE executing a tool call and waits for a human to
    approve, edit, or reject it. Execution only continues once the human
    sends a decision back via Command(resume=...).

WHY THIS MATTERS:
    Some tool calls are irreversible (sending emails, writing to databases,
    making payments). You want a human checkpoint before they fire.

HOW IT WORKS:
    1. Agent plans a tool call
    2. Middleware raises an Interrupt before the tool executes
    3. Your code collects the interrupt and shows it to the human
    4. Human provides a decision: "approve", "edit", or "reject"
    5. You call agent.stream(Command(resume=decisions), ...) to continue

DECISIONS:
    {"type": "approve"}                  → run the tool as-is
    {"type": "edit", "edited_action": …} → run with modified args
    {"type": "reject"}                   → skip the tool, tell the agent

REQUIRES:
    A checkpointer — saves state between the pause and resume.
    Use InMemorySaver() for dev, Redis/Postgres for production.

CONFIGURATION:
    interrupt_on = {
        "tool_name": True,                         # pause on every call
        "tool_name": {"allowed_decisions": [...]}, # pause with specific decisions
        "tool_name": False,                        # never pause (explicit allow)
    }
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt


# ---------------------------------------------------------------------------
# 1. Tools — one safe (read), one dangerous (send)
# ---------------------------------------------------------------------------
def read_email(email_id: str) -> str:
    """Read an email by its ID."""
    return f"Email #{email_id}: 'Hello, please confirm your order.'"


def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email. IRREVERSIBLE — requires human approval."""
    print(f"  [TOOL EXECUTED] Sending to {recipient}: {subject}")
    return f"Email sent to {recipient}."


# ---------------------------------------------------------------------------
# 2. Agent with HumanInTheLoopMiddleware
# ---------------------------------------------------------------------------
checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-4.1",
    tools=[read_email, send_email],
    checkpointer=checkpointer,          # required for state persistence
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "read_email": False,    # never pause for read_email
            }
        ),
    ],
)

config = {"configurable": {"thread_id": "email_thread_01"}}


# ---------------------------------------------------------------------------
# 3. First run — agent plans to send an email, gets interrupted
# ---------------------------------------------------------------------------
print("=" * 60)
print("RUN 1 — Agent plans tool calls, hits interrupt on send_email")
print("=" * 60)

collected_interrupts: list[Interrupt] = []

for chunk in agent.stream(
    {"messages": [{"role": "user", "content":
        "Read email #42, then reply to alice@example.com saying 'Order confirmed!'"}]},
    config=config,
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source == "__interrupt__":
                for interrupt in update:
                    collected_interrupts.append(interrupt)
                    print("\n INTERRUPT — Approval needed:")
                    for req in interrupt.value["action_requests"]:
                        print(f"   Tool : {req.get('name', 'unknown')}")
                        print(f"   Args : {req.get('args', {})}")


# ---------------------------------------------------------------------------
# 4. Human reviews and makes decisions
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Human reviewing the planned send_email call...")
print("=" * 60)

decisions = {}
for interrupt in collected_interrupts:
    action_decisions = []
    for req in interrupt.value["action_requests"]:
        tool_name = req.get("name", "")
        if tool_name == "send_email":
            # Edit: change body for clarity
            decision = {
                "type": "edit",
                "edited_action": {
                    "name": "send_email",
                    "args": {
                        "recipient": req["args"]["recipient"],
                        "subject": "Order Confirmation",
                        "body": "Hello! Your order #42 has been confirmed. Thank you!",
                    },
                },
            }
            print(f"  ✏️  EDITED body for {req['args']['recipient']}")
        else:
            decision = {"type": "approve"}
            print(f"  APPROVED: {tool_name}")
        action_decisions.append(decision)

    decisions[interrupt.id] = {"decisions": action_decisions}


# ---------------------------------------------------------------------------
# 5. Resume with human decisions
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RUN 2 — Resuming with human decisions")
print("=" * 60)

for chunk in agent.stream(
    Command(resume=decisions),
    config=config,
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source == "tools":
                last = update["messages"][-1]
                print(f"  Tool result: {last.content}")
            elif source == "model":
                last = update["messages"][-1]
                if hasattr(last, "content") and last.content:
                    print(f"  Agent: {last.content[:200]}")