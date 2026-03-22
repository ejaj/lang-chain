"""
TOPIC: Human-in-the-Loop — All Three Decision Types

THREE DECISIONS:
    approve → run the tool exactly as the model planned, no changes
    edit   → run the tool but with human-modified arguments
    reject  → don't run the tool, add explanation to conversation

WHEN TO USE EACH:
    approve → the model's plan is correct
    edit    → the model picked the right tool but got some args wrong
              e.g. model plans to email "team@company.com", human changes to "alice@company.com"
    reject  → the tool call is wrong entirely, model needs to try differently
              e.g. model tries to delete production data, human rejects with "use staging only"

IMPORTANT — EDIT CONSERVATIVELY:
    Make small, targeted changes to arguments.
    Large edits can confuse the model and cause it to re-evaluate or repeat actions.

ORDER MATTERS:
    When multiple tool calls are interrupted at once, decisions must be
    provided in THE SAME ORDER as the action_requests in the interrupt.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────

def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    print(f"  [TOOL EXECUTED] send_email → to={to}, subject={subject}")
    return f"Email sent to {to}."


def execute_sql(query: str) -> str:
    """Execute a SQL query."""
    print(f"  [TOOL EXECUTED] execute_sql → {query}")
    return f"Executed: {query}"


def delete_records(table: str, condition: str) -> str:
    """Delete records from a table."""
    print(f"  [TOOL EXECUTED] delete_records → table={table}, condition={condition}")
    return f"Deleted from {table} where {condition}."


# ─────────────────────────────────────────────────────────────────────────────
# Shared agent
# ─────────────────────────────────────────────────────────────────────────────

def make_agent(thread_id: str):
    """Create a fresh agent with a unique thread ID."""
    agent = create_agent(
        model="gpt-4.1",
        tools=[send_email, execute_sql, delete_records],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email":     True,   # approve, edit, reject all allowed
                    "execute_sql":    True,
                    "delete_records": {"allowed_decisions": ["approve", "reject"]},
                }
            )
        ],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": thread_id}}
    return agent, config


# ─────────────────────────────────────────────────────────────────────────────
# DECISION 1: approve
# Run the tool exactly as the model planned.
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("DECISION 1: approve")
print("=" * 60)

agent, config = make_agent("thread_approve")

result = agent.invoke(
    {"messages": [HumanMessage("Send a welcome email to alice@company.com")]},
    config=config,
    version="v2",
)
print(f"Interrupted on: {result.interrupts[0].value['action_requests'][0]['name']}")

# Approve — run exactly as planned
result2 = agent.invoke(
    Command(resume={
        "decisions": [
            {"type": "approve"}   # no changes, run as-is
        ]
    }),
    config=config,
    version="v2",
)
print(f"Response: {result2.value['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# DECISION 2: edit
# Run the tool but with human-modified arguments.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DECISION 2: edit")
print("=" * 60)

agent, config = make_agent("thread_edit")

result = agent.invoke(
    {"messages": [HumanMessage("Send a welcome email to the team")]},
    config=config,
    version="v2",
)

req = result.interrupts[0].value["action_requests"][0]
print(f"Model planned: {req['name']}({req['arguments']})")

# Edit — change the recipient before sending
result2 = agent.invoke(
    Command(resume={
        "decisions": [
            {
                "type": "edit",
                "edited_action": {
                    "name": "send_email",        # tool name (usually same as original)
                    "args": {
                        "to":      "specific-team@company.com",  # changed
                        "subject": req["arguments"].get("subject", "Welcome"),  # kept
                        "body":    req["arguments"].get("body", ""),             # kept
                    },
                },
            }
        ]
    }),
    config=config,
    version="v2",
)
print(f"Response: {result2.value['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# DECISION 3: reject
# Skip the tool entirely, add explanation to conversation.
# The model will see the rejection message and can try a different approach.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DECISION 3: reject")
print("=" * 60)

agent, config = make_agent("thread_reject")

result = agent.invoke(
    {"messages": [HumanMessage("Delete all old user records from the database")]},
    config=config,
    version="v2",
)

req = result.interrupts[0].value["action_requests"][0]
print(f"Model planned: {req['name']}({req['arguments']})")

# Reject — explain why and what to do instead
result2 = agent.invoke(
    Command(resume={
        "decisions": [
            {
                "type":    "reject",
                "message": (
                    "Do not delete production data directly. "
                    "Instead, mark records as archived with status='archived' "
                    "and set archived_at to the current timestamp."
                ),
            }
        ]
    }),
    config=config,
    version="v2",
)
print(f"Response: {result2.value['messages'][-1].content[:300]}")


# ─────────────────────────────────────────────────────────────────────────────
# MULTIPLE TOOL CALLS — one decision per action, in order
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("MULTIPLE TOOL CALLS — mixed decisions")
print("=" * 60)

agent, config = make_agent("thread_multiple")

result = agent.invoke(
    {"messages": [HumanMessage(
        "Send an update email to the team and execute a cleanup query on the users table"
    )]},
    config=config,
    version="v2",
)

if result.interrupts:
    reqs = result.interrupts[0].value["action_requests"]
    print(f"Actions interrupted: {[r['name'] for r in reqs]}")

    # Decisions MUST be in the same order as action_requests
    result2 = agent.invoke(
        Command(resume={
            "decisions": [
                {"type": "approve"},   # first action: approve email
                {                      # second action: edit the SQL
                    "type": "edit",
                    "edited_action": {
                        "name": "execute_sql",
                        "args": {"query": "UPDATE users SET status='inactive' WHERE last_login < NOW() - INTERVAL '90 days'"},
                    },
                },
            ]
        }),
        config=config,
        version="v2",
    )
    print(f"Response: {result2.value['messages'][-1].content[:300]}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("""
Decision Types Quick Reference:

  Approve:
    {"type": "approve"}

  Edit:
    {
        "type": "edit",
        "edited_action": {
            "name": "tool_name",          # usually same as original
            "args": {"key": "new_value"}  # full args dict with changes
        }
    }

  Reject:
    {
        "type": "reject",
        "message": "Reason for rejection + what to do instead"
    }

  Multiple decisions (must match order of action_requests):
    Command(resume={"decisions": [decision_1, decision_2, ...]})
""")