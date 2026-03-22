"""
Handoffs: single agent with middleware
=======================================================================

WHAT IS IT?
-----------
ONE agent that changes its behavior based on state.
Middleware intercepts every model call and swaps in the right
system prompt and tools for the current step.

When a tool call updates current_step, the middleware automatically
uses a different configuration on the next turn.

WHEN TO USE:
- Most handoff scenarios — this is the simpler approach
- When each "state" is just a different configuration, not a
  completely different piece of code
- When conversation history should flow naturally between states

WHY SIMPLER THAN MULTI-AGENT:
- One graph, one agent, one message history
- No need to worry about which messages to pass between agents
- State change = prompt+tools swap, not a full agent swap

SCENARIO: Customer support
  Step 1 — warranty_check : ask about warranty, record answer
  Step 2 — diagnose       : ask about the problem, record issue type
  Step 3 — resolve        : provide solution based on what was collected

INSTALL:
    pip install langchain-anthropic langgraph

RUN:
    export ANTHROPIC_API_KEY=your_key
    python 01_single_agent_middleware.py
"""

from typing import TypedDict, Literal, Annotated, Optional
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command

# ------------------------------------------------------------------
# State schema — persists across all conversation turns
# ------------------------------------------------------------------

class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_step: str           # controls which prompt+tools are active
    warranty_status: str        # collected in step 1
    issue_type: str             # collected in step 2

# ------------------------------------------------------------------
# Step configurations — prompt + tools per step
# ------------------------------------------------------------------

STEP_CONFIGS = {
    "warranty_check": {
        "prompt": (
            "You are a customer support agent. "
            "Your ONLY job right now is to find out if the customer's device "
            "is under warranty. Ask them clearly, then call record_warranty_status "
            "with either 'in_warranty' or 'out_of_warranty'."
        ),
        "tools": ["record_warranty_status"],
    },
    "diagnose": {
        "prompt": (
            "You are a customer support agent. "
            "The warranty status has been recorded. "
            "Now ask the customer to describe their issue. "
            "Once they describe it, classify it and call record_issue_type "
            "with 'hardware', 'software', or 'user_error'."
        ),
        "tools": ["record_issue_type"],
    },
    "resolve": {
        "prompt": (
            "You are a customer support agent providing a resolution. "
            "Based on the collected information, give the customer "
            "a clear, helpful resolution. You can escalate to a human if needed."
        ),
        "tools": ["provide_solution", "escalate_to_human"],
    },
}

# ------------------------------------------------------------------
# Tools — each one updates state to move to the next step
# ------------------------------------------------------------------

@tool
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Record the warranty status and move to the diagnose step."""
    return Command(
        update={
            "warranty_status": status,
            "current_step": "diagnose",       # triggers step change
            "messages": [
                ToolMessage(
                    content=f"Warranty status recorded: {status}. Moving to diagnose step.",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def record_issue_type(
    issue_type: Literal["hardware", "software", "user_error"],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Record the issue type and move to the resolve step."""
    return Command(
        update={
            "issue_type": issue_type,
            "current_step": "resolve",        # triggers step change
            "messages": [
                ToolMessage(
                    content=f"Issue type recorded: {issue_type}. Moving to resolve step.",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def provide_solution(
    solution: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Provide the final resolution to the customer."""
    return Command(
        update={
            "current_step": "done",
            "messages": [
                ToolMessage(
                    content=f"Solution provided: {solution}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def escalate_to_human(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Escalate to a human agent when the issue is too complex."""
    return Command(
        update={
            "current_step": "escalated",
            "messages": [
                ToolMessage(
                    content=f"Escalated to human: {reason}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


ALL_TOOLS = {
    "record_warranty_status": record_warranty_status,
    "record_issue_type": record_issue_type,
    "provide_solution": provide_solution,
    "escalate_to_human": escalate_to_human,
}

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

model = ChatAnthropic(model="claude-sonnet-4-20250514")

# ------------------------------------------------------------------
# Middleware node — reads state, picks prompt + tools, calls model
# ------------------------------------------------------------------

def support_agent(state: SupportState) -> Command:
    """
    The middleware. Runs on every turn.
    Reads current_step → picks the right prompt + tools → calls model.
    """
    step = state.get("current_step", "warranty_check")

    # If we're done, stop
    if step in ("done", "escalated"):
        return Command(goto=END)

    config = STEP_CONFIGS.get(step, STEP_CONFIGS["warranty_check"])

    # Build system message for this step
    system = SystemMessage(content=config["prompt"])

    # Bind only the tools relevant to this step
    active_tools = [ALL_TOOLS[name] for name in config["tools"]]
    bound_model = model.bind_tools(active_tools)

    # Call the model
    response = bound_model.invoke([system] + state["messages"])

    return Command(
        update={"messages": [response]},
        goto="support_agent",   # loop back — tool results will update state
    )

# ------------------------------------------------------------------
# Graph — single node that loops until done
# ------------------------------------------------------------------

builder = StateGraph(SupportState)
builder.add_node("support_agent", support_agent)
builder.add_edge(START, "support_agent")

graph = builder.compile()

# ------------------------------------------------------------------
# Helper: multi-turn chat
# ------------------------------------------------------------------

def chat(state: SupportState, user_message: str) -> tuple[str, SupportState]:
    """Send a message, get a reply, return updated state."""
    state["messages"].append(HumanMessage(content=user_message))
    result = graph.invoke(state)
    reply = next(
        (m.content for m in reversed(result["messages"])
         if hasattr(m, "type") and m.type == "ai" and m.content),
        "(no reply)"
    )
    return reply, result

# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    state: SupportState = {
        "messages": [],
        "current_step": "warranty_check",
        "warranty_status": "",
        "issue_type": "",
    }

    print("=== Customer Support Demo ===\n")

    # Turn 1: user reports a problem
    print("USER: My phone screen is broken.")
    reply, state = chat(state, "My phone screen is broken.")
    print(f"AGENT [{state['current_step']}]: {reply}\n")

    # Turn 2: user answers warranty question
    print("USER: Yes, I bought it 6 months ago so it should be under warranty.")
    reply, state = chat(state, "Yes, I bought it 6 months ago so it should be under warranty.")
    print(f"AGENT [{state['current_step']}]: {reply}\n")

    # Turn 3: user describes the issue
    print("USER: The screen cracked after I dropped it.")
    reply, state = chat(state, "The screen cracked after I dropped it.")
    print(f"AGENT [{state['current_step']}]: {reply}\n")

    # Turn 4: agent resolves
    print("USER: What do I need to do next?")
    reply, state = chat(state, "What do I need to do next?")
    print(f"AGENT [{state['current_step']}]: {reply}\n")

    print(f"\n=== Final state ===")
    print(f"Warranty:   {state['warranty_status']}")
    print(f"Issue type: {state['issue_type']}")
    print(f"Step:       {state['current_step']}")