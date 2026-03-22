"""
Handoffs: multiple agent subgraphs
================================================================

WHAT IS IT?
-----------
Distinct agents live as separate NODES in a graph.
A handoff tool uses Command(goto="other_agent", graph=Command.PARENT)
to navigate from one agent node to another.

Each agent is a fully independent create_react_agent — it could have
its own complex internal logic, reflection steps, retrieval, etc.

WHEN TO USE:
- Each agent needs bespoke, complex implementation
  (not just a different prompt — a different graph structure)
- You need agents developed and deployed by separate teams
- Each agent has genuinely different internal workflows

USE SINGLE AGENT MIDDLEWARE (01) INSTEAD WHEN:
- The difference between states is just prompt + tools
- You want simpler code
- You don't need independent agent graph structures

CRITICAL — CONTEXT ENGINEERING:
Unlike single-agent middleware where history flows naturally,
here you must EXPLICITLY decide what messages to pass to the
next agent. LLMs expect every tool call to have a matching
ToolMessage response — if you break this, the history is malformed.

Rule: always pass the last AIMessage + a ToolMessage together.

SCENARIO: Sales + Support
  - Support agent handles general questions
  - Sales agent handles purchase intent
  - Either can transfer to the other

INSTALL:
    pip install langchain-anthropic langgraph

RUN:
    export ANTHROPIC_API_KEY=your_key
    python 02_multi_agent_subgraph.py
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
    active_agent: str   # tracks which agent is currently active

# ------------------------------------------------------------------
# Shared model
# ------------------------------------------------------------------

model = ChatAnthropic(model="claude-sonnet-4-20250514")

# ------------------------------------------------------------------
# Handoff tools
# ------------------------------------------------------------------
# IMPORTANT: when using Command.PARENT to hand off to another agent,
# you MUST include:
#   1. The last AIMessage (so the history shows what the agent said)
#   2. A ToolMessage matching the tool_call_id (completes the tool cycle)
#
# Without both, the conversation history is malformed and the
# next agent will error or behave unpredictably.

@tool
def transfer_to_sales(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, "state"],
) -> Command:
    """Transfer the conversation to the sales agent."""
    # Find the last AI message to carry forward
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None
    )
    transfer_msg = ToolMessage(
        content="Transferred to sales agent.",
        tool_call_id=tool_call_id,
    )

    messages_to_pass = []
    if last_ai:
        messages_to_pass.append(last_ai)
    messages_to_pass.append(transfer_msg)

    return Command(
        goto="sales_agent",             # which node to go to next
        update={
            "active_agent": "sales_agent",
            "messages": messages_to_pass,
        },
        graph=Command.PARENT            # navigate in the parent graph
    )


@tool
def transfer_to_support(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, "state"],
) -> Command:
    """Transfer the conversation back to the support agent."""
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None
    )
    transfer_msg = ToolMessage(
        content="Transferred to support agent.",
        tool_call_id=tool_call_id,
    )

    messages_to_pass = []
    if last_ai:
        messages_to_pass.append(last_ai)
    messages_to_pass.append(transfer_msg)

    return Command(
        goto="support_agent",
        update={
            "active_agent": "support_agent",
            "messages": messages_to_pass,
        },
        graph=Command.PARENT
    )

# ------------------------------------------------------------------
# Agent nodes — each is a full create_react_agent
# Each could have its own internal graph, retrieval, reflection, etc.
# ------------------------------------------------------------------

support_agent = create_react_agent(
    model=model,
    tools=[transfer_to_sales],  # can hand off to sales
    prompt=(
        "You are a customer support agent. "
        "You help with product issues, refunds, and general questions. "
        "If the customer wants to buy something or upgrade, "
        "transfer them to the sales agent using transfer_to_sales."
    )
)

sales_agent = create_react_agent(
    model=model,
    tools=[transfer_to_support],  # can hand off back to support
    prompt=(
        "You are a sales agent. "
        "You help customers choose products, explain pricing, and close deals. "
        "If the customer has a support issue or complaint, "
        "transfer them to the support agent using transfer_to_support."
    )
)

# ------------------------------------------------------------------
# Node wrappers — invoke each agent and return a Command
# ------------------------------------------------------------------

def run_support_agent(state: AgentState) -> Command:
    result = support_agent.invoke({"messages": state["messages"]})
    return Command(
        update={"messages": result["messages"], "active_agent": "support_agent"},
        goto=END,  # stop after each turn so user can reply
    )


def run_sales_agent(state: AgentState) -> Command:
    result = sales_agent.invoke({"messages": state["messages"]})
    return Command(
        update={"messages": result["messages"], "active_agent": "sales_agent"},
        goto=END,
    )

# ------------------------------------------------------------------
# Router — decides which agent to start with based on active_agent
# ------------------------------------------------------------------

def router(state: AgentState) -> str:
    return state.get("active_agent", "support_agent")

# ------------------------------------------------------------------
# Graph
# ------------------------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("support_agent", run_support_agent)
builder.add_node("sales_agent", run_sales_agent)

# Router decides first destination
builder.add_conditional_edges(START, router, {
    "support_agent": "support_agent",
    "sales_agent": "sales_agent",
})

graph = builder.compile()

# ------------------------------------------------------------------
# Helper: multi-turn chat
# ------------------------------------------------------------------

def chat(state: AgentState, user_message: str) -> tuple[str, AgentState]:
    """Send a message, get a reply, return updated state."""
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
        "active_agent": "support_agent",  # start with support
    }

    print("=== Sales + Support Handoff Demo ===\n")

    # Turn 1: support question
    print("USER: My order hasn't arrived yet.")
    reply, state = chat(state, "My order hasn't arrived yet.")
    print(f"AGENT [{state['active_agent']}]: {reply}\n")

    # Turn 2: customer switches to purchase intent → triggers handoff
    print("USER: Actually, I also want to buy the premium plan. Can you help?")
    reply, state = chat(state, "Actually, I also want to buy the premium plan. Can you help?")
    print(f"AGENT [{state['active_agent']}]: {reply}\n")

    # Turn 3: continue with sales agent
    print("USER: What's included in the premium plan?")
    reply, state = chat(state, "What's included in the premium plan?")
    print(f"AGENT [{state['active_agent']}]: {reply}\n")

    # Turn 4: back to support question → triggers handoff back
    print("USER: Wait, I still have a problem with my original order.")
    reply, state = chat(state, "Wait, I still have a problem with my original order.")
    print(f"AGENT [{state['active_agent']}]: {reply}\n")

    print(f"\n=== Final active agent: {state['active_agent']} ===")