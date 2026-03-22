"""
Context engineering for handoffs
=============================================================

WHAT IS IT?
-----------
When handing off between agents, conversation history can break.
LLMs require every tool call to have a matching ToolMessage response.
If you pass incomplete history to the next agent, it errors or
produces garbage output.

This file shows:
  - WHY history breaks during handoffs
  - The rule for what to always pass
  - A helper to safely extract handoff messages

THE RULE:
When using Command(goto=..., graph=Command.PARENT), always pass:
  1. The last AIMessage from the current agent
     (so the next agent sees what was said)
  2. A ToolMessage with the matching tool_call_id
     (completes the tool call/response cycle)

Never pass:
  - The full message history (causes context bloat)
  - Only the ToolMessage without the AIMessage (malformed history)
  - Only the AIMessage without the ToolMessage (unpaired tool call)

INSTALL:
    pip install langchain-anthropic langgraph

RUN:
    export ANTHROPIC_API_KEY=your_key
    python 03_context_engineering.py
"""

from typing import TypedDict, Annotated
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

# ------------------------------------------------------------------
# Shared state
# ------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    active_agent: str

model = ChatAnthropic(model="claude-sonnet-4-20250514")

# ------------------------------------------------------------------
# Helper — builds the correct pair of messages for a handoff
# ------------------------------------------------------------------

def build_handoff_messages(
    state: dict,
    tool_call_id: str,
    transfer_note: str,
) -> list[BaseMessage]:
    """
    Returns the two messages required for a valid handoff:
      1. Last AIMessage from the current agent
      2. ToolMessage completing the transfer tool call

    WHY:
    The LLM made a tool call (the transfer tool).
    The next agent's history must show:
      - AIMessage: "I'll transfer you now" (with tool_calls=[...])
      - ToolMessage: "Transfer complete" (matching tool_call_id)
    Without both, the conversation history is malformed.
    """
    # Find the last AIMessage (the one that called the transfer tool)
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None
    )

    tool_msg = ToolMessage(
        content=transfer_note,
        tool_call_id=tool_call_id,
    )

    # Always include both — never just one
    if last_ai:
        return [last_ai, tool_msg]
    return [tool_msg]

# ------------------------------------------------------------------
# WRONG WAY — common mistakes shown for comparison
# ------------------------------------------------------------------

@tool
def transfer_wrong_no_ai_message(
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """WRONG: missing AIMessage — the tool call has no source."""
    return Command(
        goto="other_agent",
        update={
            "messages": [
                # Only ToolMessage — but where's the AIMessage that made this call?
                # Next agent sees: ToolMessage with no paired AIMessage → malformed
                ToolMessage(content="Transferred", tool_call_id=tool_call_id)
            ]
        },
        graph=Command.PARENT
    )


@tool
def transfer_wrong_full_history(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, "state"],
) -> Command:
    """WRONG: passing full history — bloats context, causes confusion."""
    return Command(
        goto="other_agent",
        update={
            # Don't pass ALL messages — the next agent doesn't need the
            # entire conversation, just the handoff pair
            "messages": state["messages"],  # BAD: context bloat
        },
        graph=Command.PARENT
    )

# ------------------------------------------------------------------
# RIGHT WAY — using the helper
# ------------------------------------------------------------------

@tool
def transfer_to_billing(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, "state"],
) -> Command:
    """
    CORRECT: pass exactly the last AIMessage + matching ToolMessage.
    The next agent receives a clean, valid 2-message handoff context.
    """
    messages = build_handoff_messages(
        state=state,
        tool_call_id=tool_call_id,
        transfer_note="Customer transferred to billing agent.",
    )
    return Command(
        goto="billing_agent",
        update={
            "active_agent": "billing_agent",
            "messages": messages,
        },
        graph=Command.PARENT
    )


@tool
def transfer_to_support(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, "state"],
) -> Command:
    """Transfer back to support agent."""
    messages = build_handoff_messages(
        state=state,
        tool_call_id=tool_call_id,
        transfer_note="Customer transferred back to support agent.",
    )
    return Command(
        goto="support_agent",
        update={
            "active_agent": "support_agent",
            "messages": messages,
        },
        graph=Command.PARENT
    )

# ------------------------------------------------------------------
# Agents
# ------------------------------------------------------------------

support_agent = create_react_agent(
    model=model,
    tools=[transfer_to_billing],
    prompt=(
        "You are a support agent. Handle general issues. "
        "If the customer has a billing question, use transfer_to_billing."
    )
)

billing_agent = create_react_agent(
    model=model,
    tools=[transfer_to_support],
    prompt=(
        "You are a billing agent. Handle invoices, charges, and refunds. "
        "If the customer has a general support issue, use transfer_to_support."
    )
)

# ------------------------------------------------------------------
# Node wrappers
# ------------------------------------------------------------------

def run_support(state: AgentState) -> Command:
    result = support_agent.invoke({"messages": state["messages"]})
    return Command(
        update={"messages": result["messages"], "active_agent": "support_agent"},
        goto=END,
    )


def run_billing(state: AgentState) -> Command:
    result = billing_agent.invoke({"messages": state["messages"]})
    return Command(
        update={"messages": result["messages"], "active_agent": "billing_agent"},
        goto=END,
    )


def router(state: AgentState) -> str:
    return state.get("active_agent", "support_agent")

# ------------------------------------------------------------------
# Graph
# ------------------------------------------------------------------

builder = StateGraph(AgentState)
builder.add_node("support_agent", run_support)
builder.add_node("billing_agent", run_billing)
builder.add_conditional_edges(START, router, {
    "support_agent": "support_agent",
    "billing_agent": "billing_agent",
})

graph = builder.compile()

# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def chat(state: AgentState, user_message: str) -> tuple[str, AgentState]:
    state["messages"].append(HumanMessage(content=user_message))
    result = graph.invoke(state)
    reply = next(
        (m.content for m in reversed(result["messages"])
         if isinstance(m, AIMessage) and m.content),
        "(no reply)"
    )
    return reply, result

# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    state: AgentState = {
        "messages": [],
        "active_agent": "support_agent",
    }

    print("=== Context Engineering Demo ===\n")

    print("USER: I have a question about my last invoice.")
    reply, state = chat(state, "I have a question about my last invoice.")
    print(f"AGENT [{state['active_agent']}]: {reply}\n")

    print("USER: I was charged twice for the same month.")
    reply, state = chat(state, "I was charged twice for the same month.")
    print(f"AGENT [{state['active_agent']}]: {reply}\n")

    print("USER: Yes please process the refund.")
    reply, state = chat(state, "Yes please process the refund.")
    print(f"AGENT [{state['active_agent']}]: {reply}\n")

    # Show the handoff messages in history
    print("\n=== Message history summary ===")
    for i, m in enumerate(state["messages"]):
        mtype = type(m).__name__
        preview = str(m.content)[:80].replace("\n", " ")
        print(f"  [{i}] {mtype}: {preview}")