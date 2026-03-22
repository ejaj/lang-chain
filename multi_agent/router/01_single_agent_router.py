"""
Route to a single specialized agent
================================================================

WHAT IS IT?
-----------
The router classifies the incoming query and sends it to exactly
ONE specialized agent. Uses Command(goto=agent_name) to navigate
to the correct node in the graph.

WHEN TO USE:
- Input clearly belongs to one domain at a time
- You want deterministic, lightweight routing (rule-based or LLM call)
- No need to query multiple sources simultaneously

WHY NOT JUST USE IF/ELSE:
- Rule-based routing (keywords, regex) is fast and cheap
- LLM-based routing handles ambiguous or complex queries better
- The graph structure makes it easy to add new agents later

SCENARIO: A knowledge base assistant with 3 domains.
  "How do I reset my password?" → support_agent
  "What's in the Q3 report?"   → finance_agent
  "How do I use the API?"      → docs_agent

INSTALL:
    pip install langchain-anthropic langgraph

RUN:
    export ANTHROPIC_API_KEY=your_key
    python 01_single_agent_router.py
"""

from typing import TypedDict, Annotated
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

class RouterState(TypedDict):
    query: str                                          # original user query
    messages: Annotated[list[BaseMessage], add_messages]
    route: str                                          # which agent was chosen

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

model = ChatAnthropic(model="claude-sonnet-4-20250514")

# ------------------------------------------------------------------
# Specialized agents — each owns a distinct domain
# ------------------------------------------------------------------

support_agent = create_react_agent(
    model=model,
    tools=[],   # add real support tools here
    prompt=(
        "You are a customer support specialist. "
        "Help users with account issues, billing, troubleshooting, and general help. "
        "Be empathetic, clear, and actionable."
    )
)

finance_agent = create_react_agent(
    model=model,
    tools=[],   # add real finance/reporting tools here
    prompt=(
        "You are a financial analyst assistant. "
        "Help users with financial reports, metrics, forecasts, and data. "
        "Be precise and data-driven."
    )
)

docs_agent = create_react_agent(
    model=model,
    tools=[],   # add real documentation tools here
    prompt=(
        "You are a technical documentation specialist. "
        "Help users find API references, guides, and technical how-tos. "
        "Be concise and include code examples where relevant."
    )
)

# ------------------------------------------------------------------
# Router node — classifies the query, returns Command(goto=agent)
# ------------------------------------------------------------------

ROUTE_PROMPT = """Classify this user query into exactly one category.

Categories:
- support  : account issues, billing, passwords, troubleshooting, general help
- finance  : financial reports, revenue, metrics, forecasts, data analysis
- docs     : API usage, technical guides, how-to, code examples, integration

Query: {query}

Reply with ONLY one word: support, finance, or docs.
"""

def router_node(state: RouterState) -> Command:
    """
    Single LLM call to classify the query.
    Routes to the correct agent node using Command(goto=...).
    """
    response = model.invoke([
        SystemMessage(content=ROUTE_PROMPT.format(query=state["query"]))
    ])

    raw = response.content.strip().lower()

    # Map to valid agent names (fallback to support if unexpected)
    route_map = {"support": "support_agent", "finance": "finance_agent", "docs": "docs_agent"}
    destination = route_map.get(raw, "support_agent")

    return Command(
        update={"route": raw},
        goto=destination
    )

# ------------------------------------------------------------------
# Agent wrapper nodes — invoke each specialist agent
# ------------------------------------------------------------------

def run_support(state: RouterState) -> dict:
    result = support_agent.invoke({
        "messages": [HumanMessage(content=state["query"])]
    })
    return {"messages": result["messages"]}


def run_finance(state: RouterState) -> dict:
    result = finance_agent.invoke({
        "messages": [HumanMessage(content=state["query"])]
    })
    return {"messages": result["messages"]}


def run_docs(state: RouterState) -> dict:
    result = docs_agent.invoke({
        "messages": [HumanMessage(content=state["query"])]
    })
    return {"messages": result["messages"]}

# ------------------------------------------------------------------
# Graph
# ------------------------------------------------------------------

builder = StateGraph(RouterState)

builder.add_node("router",        router_node)
builder.add_node("support_agent", run_support)
builder.add_node("finance_agent", run_finance)
builder.add_node("docs_agent",    run_docs)

builder.add_edge(START,           "router")
builder.add_edge("support_agent", END)
builder.add_edge("finance_agent", END)
builder.add_edge("docs_agent",    END)

graph = builder.compile()

# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

def ask(query: str) -> None:
    result = graph.invoke({
        "query": query,
        "messages": [],
        "route": "",
    })
    reply = next(
        (m.content for m in reversed(result["messages"])
         if isinstance(m, AIMessage) and m.content),
        "(no reply)"
    )
    print(f"QUERY:  {query}")
    print(f"ROUTED TO: {result['route']}")
    print(f"ANSWER: {reply[:300]}...\n")
    print("-" * 60)


if __name__ == "__main__":
    print("=== Single Agent Router Demo ===\n")

    ask("How do I reset my password?")
    ask("What was our Q3 revenue compared to Q2?")
    ask("How do I authenticate with the REST API using OAuth2?")
    ask("I was charged twice this month — can you help?")