"""
Simplest stateful approach: router as a tool
===========================================================================

WHAT IS IT?
-----------
The simplest way to make a router work in multi-turn conversations:
wrap the entire stateless router as a TOOL that a conversational
agent calls when it needs to search.

The conversational agent handles memory and context naturally.
The router stays completely stateless and simple.
The user always talks to the same conversational agent — no
inconsistency from switching between agents mid-conversation.

HOW IT WORKS:
  Conversational agent (has memory, handles all turns)
      ↓
  Calls search_docs("user question") as a tool
      ↓
  Stateless router fan-out → multiple source agents in parallel
      ↓
  Synthesizer → combined answer string
      ↓
  Returns answer string back to conversational agent
      ↓
  Conversational agent replies to user (has full history)

WHY THIS IS SIMPLER THAN A STATEFUL ROUTER:
- No custom history management
- No inconsistent agent tones (one conversational agent throughout)
- No complex state passing between parallel agents across turns
- The router is just a tool — black box to the conversational agent

WHEN TO USE THIS OVER 03_stateless_vs_stateful.py:
- You need multi-turn conversation
- You want the simplest possible implementation
- You're OK with one extra LLM call per search (conversational agent + router)

SCENARIO: A support assistant that can search GitHub, Notion, and Slack.
The user has a multi-turn conversation and the assistant remembers context.

INSTALL:
    pip install langchain-anthropic langgraph

RUN:
    export ANTHROPIC_API_KEY=your_key
    python 04_tool_wrapper_stateful.py
"""

from typing import TypedDict, Annotated
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

model = ChatAnthropic(model="claude-sonnet-4-20250514")

# ------------------------------------------------------------------
# Source agents (stateless — called fresh each time)
# ------------------------------------------------------------------

from langchain_core.tools import tool as lc_tool

@lc_tool
def search_github(query: str) -> str:
    """Search GitHub issues and pull requests."""
    return (
        f"GitHub: Issue #482 'Auth 503 errors' opened 2h ago. "
        f"PR #491 'Fix token refresh' merged 45min ago."
    )

@lc_tool
def search_notion(query: str) -> str:
    """Search Notion docs and runbooks."""
    return (
        f"Notion: Runbook 'Auth Incident Response' updated today. "
        f"Step 3: check Redis pool. Step 4: restart pods."
    )

@lc_tool
def search_slack(query: str) -> str:
    """Search Slack channels."""
    return (
        f"Slack #incidents: @sre-alice 'Redis at 94%, scaling.' "
        f"@sre-bob 'Deployed fix, monitoring.' (22min ago)"
    )

github_agent = create_react_agent(model=model, tools=[search_github],
    prompt="Search GitHub and summarize relevant findings concisely.")

notion_agent = create_react_agent(model=model, tools=[search_notion],
    prompt="Search Notion and summarize relevant findings concisely.")

slack_agent  = create_react_agent(model=model, tools=[search_slack],
    prompt="Search Slack and summarize relevant findings concisely.")

SOURCE_AGENTS = {"github": github_agent, "notion": notion_agent, "slack": slack_agent}

# ------------------------------------------------------------------
# Inner stateless router + synthesizer (same as 02_parallel_router.py)
# ------------------------------------------------------------------

class InnerState(TypedDict):
    query: str
    agent_results: Annotated[list[str], lambda a, b: a + b]
    final_answer: str

class AgentState(TypedDict):
    query: str
    source: str

def inner_router(state: InnerState) -> list[Send]:
    return [
        Send("search_agent", {"query": state["query"], "source": "github"}),
        Send("search_agent", {"query": state["query"], "source": "notion"}),
        Send("search_agent", {"query": state["query"], "source": "slack"}),
    ]

def search_agent_node(state: AgentState) -> dict:
    agent = SOURCE_AGENTS[state["source"]]
    result = agent.invoke({"messages": [HumanMessage(content=state["query"])]})
    final = next(
        (m.content for m in reversed(result["messages"])
         if isinstance(m, AIMessage) and m.content), "(no result)"
    )
    return {"agent_results": [f"[{state['source'].upper()}] {final}"]}

SYNTHESIZE_PROMPT = """Synthesize these search results into a clear, concise answer.
Remove duplication. Highlight the most important findings.

Results:
{results}

Question: {query}
"""

def synthesizer_node(state: InnerState) -> dict:
    combined = "\n\n".join(state["agent_results"])
    response = model.invoke([
        SystemMessage(content=SYNTHESIZE_PROMPT.format(
            results=combined, query=state["query"]
        ))
    ])
    return {"final_answer": response.content}

# Build inner graph (stateless)
inner_builder = StateGraph(InnerState)
inner_builder.add_node("search_agent", search_agent_node)
inner_builder.add_node("synthesizer",  synthesizer_node)
inner_builder.add_conditional_edges(START, inner_router, ["search_agent"])
inner_builder.add_edge("search_agent", "synthesizer")
inner_builder.add_edge("synthesizer",  END)
inner_workflow = inner_builder.compile()

# ------------------------------------------------------------------
# The tool wrapper — exposes the stateless router as a single tool
# ------------------------------------------------------------------

@tool
def search_knowledge_base(query: str) -> str:
    """Search across GitHub, Notion, and Slack for relevant information.

    Use this tool to find information about incidents, documentation,
    runbooks, or team communications.
    """
    result = inner_workflow.invoke({
        "query": query,
        "agent_results": [],
        "final_answer": "",
    })
    return result["final_answer"]

# ------------------------------------------------------------------
# Conversational agent — the ONLY thing the user talks to
# Has memory, handles multi-turn naturally via create_react_agent
# ------------------------------------------------------------------

conversational_agent = create_react_agent(
    model=model,
    tools=[search_knowledge_base],
    prompt=(
        "You are a helpful assistant with access to the company knowledge base. "
        "Use search_knowledge_base to find information from GitHub, Notion, and Slack. "
        "You remember the conversation and can answer follow-up questions. "
        "If a follow-up question refers to something discussed earlier, "
        "use that context when forming your search query."
    )
)

# ------------------------------------------------------------------
# Helper: multi-turn conversation
# ------------------------------------------------------------------

def chat(history: list, user_message: str) -> tuple[str, list]:
    history.append(HumanMessage(content=user_message))
    result = conversational_agent.invoke({"messages": history})
    reply = result["messages"][-1].content
    history = result["messages"]
    return reply, history

# ------------------------------------------------------------------
# Demo — multi-turn conversation, router stays stateless underneath
# ------------------------------------------------------------------

if __name__ == "__main__":
    history = []

    print("=== Tool Wrapper Stateful Router Demo ===")
    print("The user talks to ONE conversational agent with memory.")
    print("The router underneath is stateless — called fresh each time.\n")

    # Turn 1: search question
    print("USER: What's the current status of the auth service?")
    reply, history = chat(history, "What's the current status of the auth service?")
    print(f"AGENT: {reply}\n")
    print("-" * 60)

    # Turn 2: follow-up — conversational agent uses history to form query
    print("USER: Who was the on-call engineer handling it?")
    reply, history = chat(history, "Who was the on-call engineer handling it?")
    print(f"AGENT: {reply}\n")
    print("-" * 60)

    # Turn 3: unrelated question — agent answers from general knowledge
    print("USER: How long do incidents like this usually take to resolve?")
    reply, history = chat(history, "How long do incidents like this usually take to resolve?")
    print(f"AGENT: {reply}\n")
    print("-" * 60)

    # Turn 4: another search
    print("USER: Is there a runbook for Redis memory issues?")
    reply, history = chat(history, "Is there a runbook for Redis memory issues?")
    print(f"AGENT: {reply}\n")

    print(f"\n(Total turns: {sum(1 for m in history if isinstance(m, HumanMessage))})")
    print("The conversational agent remembered context throughout.")
    print("The inner router ran stateless each time search_knowledge_base was called.")