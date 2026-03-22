"""
Fan out to multiple agents in parallel
==============================================================

WHAT IS IT?
-----------
The router sends the query to MULTIPLE specialized agents simultaneously
using Send. All agents run in parallel. A synthesizer node then
combines all their outputs into one final answer.

This is useful when the query spans multiple knowledge domains and
you want results from all of them, not just one.

WHEN TO USE:
- Query requires input from multiple sources simultaneously
  e.g. "What does GitHub, Notion, and Slack say about this incident?"
- You want to minimize total latency (parallel beats sequential)
- Each source has independent knowledge — they don't depend on each other

WHY Send INSTEAD OF Command:
- Command(goto="agent") → routes to ONE agent
- Send("agent", state)  → spawns a parallel task to ONE agent
- [Send(...), Send(...), Send(...)] → spawns MULTIPLE parallel tasks
  All run at the same time, results collected when all complete.

SCENARIO: A knowledge base search across GitHub, Notion, and Slack.
  User asks: "What's the status of the auth service outage?"
  → GitHub agent  : searches issues and PRs
  → Notion agent  : searches internal runbooks
  → Slack agent   : searches incident channel history
  → Synthesizer   : combines all three into one coherent answer

INSTALL:
    pip install langchain-anthropic langgraph

RUN:
    export ANTHROPIC_API_KEY=your_key
    python 02_parallel_router.py
"""

from typing import TypedDict, Annotated
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, Send

# ------------------------------------------------------------------
# State schemas
# ------------------------------------------------------------------

class RouterState(TypedDict):
    """Top-level graph state."""
    query: str
    agent_results: Annotated[list[str], lambda a, b: a + b]  # collected from all agents
    final_answer: str

class AgentState(TypedDict):
    """State passed to each parallel agent."""
    query: str
    source: str   # which source this agent searches

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

model = ChatAnthropic(model="claude-sonnet-4-20250514")

# ------------------------------------------------------------------
# Mock source search tools (replace with real API calls)
# ------------------------------------------------------------------

from langchain_core.tools import tool

@tool
def search_github(query: str) -> str:
    """Search GitHub issues, PRs, and commits."""
    # In production: call GitHub API
    return (
        f"GitHub results for '{query}': "
        "Issue #482 'Auth service returns 503' opened 2h ago by @devops-team. "
        "PR #491 'Fix token refresh race condition' merged 45 min ago. "
        "Last commit: 'Revert cache TTL change' 30 min ago."
    )

@tool
def search_notion(query: str) -> str:
    """Search Notion pages and runbooks."""
    # In production: call Notion API
    return (
        f"Notion results for '{query}': "
        "Runbook 'Auth Service Incident Response' — Last updated today. "
        "Step 3: check Redis connection pool. Step 4: restart auth pods. "
        "Known issue: token refresh fails under high load."
    )

@tool
def search_slack(query: str) -> str:
    """Search Slack channels and messages."""
    # In production: call Slack API
    return (
        f"Slack results for '{query}': "
        "#incidents channel — @sre-alice: 'Auth service degraded, investigating cache.' "
        "@sre-bob: 'Redis memory at 94%, scaling up.' "
        "@sre-alice: 'Deployed fix, monitoring.' (22 min ago)"
    )

# ------------------------------------------------------------------
# Specialized agents — one per knowledge source
# ------------------------------------------------------------------

github_agent = create_react_agent(
    model=model,
    tools=[search_github],
    prompt=(
        "You search GitHub for relevant issues, PRs, and commits. "
        "Always call search_github with the user's query. "
        "Summarize findings clearly and concisely."
    )
)

notion_agent = create_react_agent(
    model=model,
    tools=[search_notion],
    prompt=(
        "You search Notion for internal documentation and runbooks. "
        "Always call search_notion with the user's query. "
        "Summarize findings clearly and concisely."
    )
)

slack_agent = create_react_agent(
    model=model,
    tools=[search_slack],
    prompt=(
        "You search Slack for team communications and incident updates. "
        "Always call search_slack with the user's query. "
        "Summarize findings clearly and concisely."
    )
)

SOURCE_AGENTS = {
    "github": github_agent,
    "notion": notion_agent,
    "slack":  slack_agent,
}

# ------------------------------------------------------------------
# Router node — fans out to ALL agents using Send
# ------------------------------------------------------------------

def router_node(state: RouterState) -> list[Send]:
    """
    Send the query to all agents in parallel.
    Each Send creates an independent parallel task.
    All tasks run simultaneously. Results collected when all finish.
    """
    return [
        Send("search_agent", {"query": state["query"], "source": "github"}),
        Send("search_agent", {"query": state["query"], "source": "notion"}),
        Send("search_agent", {"query": state["query"], "source": "slack"}),
    ]

# ------------------------------------------------------------------
# Agent node — runs one source agent, returns result to top-level state
# ------------------------------------------------------------------

def search_agent_node(state: AgentState) -> dict:
    """
    Called once per Send. Runs the appropriate source agent.
    Returns a result string that gets appended to agent_results.
    """
    source = state["source"]
    agent = SOURCE_AGENTS[source]

    result = agent.invoke({
        "messages": [HumanMessage(content=state["query"])]
    })

    final = next(
        (m.content for m in reversed(result["messages"])
         if isinstance(m, AIMessage) and m.content),
        "(no result)"
    )

    return {"agent_results": [f"[{source.upper()}]\n{final}"]}

# ------------------------------------------------------------------
# Synthesizer node — combines all agent outputs into one answer
# ------------------------------------------------------------------

SYNTHESIZE_PROMPT = """You have search results from multiple sources.
Synthesize them into a single, coherent answer.
Remove duplicates. Highlight the most important and recent information.
Be concise and well-structured.

Results:
{results}

Original question: {query}
"""

def synthesizer_node(state: RouterState) -> dict:
    """Combines all agent_results into one final_answer."""
    combined = "\n\n".join(state["agent_results"])

    response = model.invoke([
        SystemMessage(content=SYNTHESIZE_PROMPT.format(
            results=combined,
            query=state["query"]
        ))
    ])

    return {"final_answer": response.content}

# ------------------------------------------------------------------
# Graph
# ------------------------------------------------------------------

builder = StateGraph(RouterState)

builder.add_node("search_agent", search_agent_node)
builder.add_node("synthesizer",  synthesizer_node)

# Router fans out to parallel agents using Send
builder.add_conditional_edges(START, router_node, ["search_agent"])

# All parallel agents feed into synthesizer
builder.add_edge("search_agent", "synthesizer")
builder.add_edge("synthesizer",  END)

graph = builder.compile()

# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Parallel Router Demo ===\n")

    queries = [
        "What's the current status of the auth service outage?",
        "Has anyone documented the Redis memory leak fix?",
    ]

    for query in queries:
        print(f"QUERY: {query}")
        result = graph.invoke({
            "query": query,
            "agent_results": [],
            "final_answer": "",
        })

        print("\nRAW RESULTS FROM EACH SOURCE:")
        for r in result["agent_results"]:
            print(f"  {r[:120]}...")

        print(f"\nSYNTHESIZED ANSWER:\n{result['final_answer']}")
        print("\n" + "=" * 60 + "\n")