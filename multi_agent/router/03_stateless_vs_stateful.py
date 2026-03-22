"""
Stateless vs. stateful routers
=============================================================

WHAT IS IT?
-----------
Routers are stateless by default — each request is independent,
no memory of previous turns. This is fine for single-shot queries.

For multi-turn conversations you need a stateful router that
remembers what was asked before so it can route follow-up questions
correctly.

TWO APPROACHES SHOWN HERE:
  Part A — Stateless  : no memory, each query stands alone
  Part B — Stateful   : router maintains conversation history
                        uses it to classify follow-up questions

STATELESS IS SIMPLER — use it unless you need follow-up question handling.

WHEN STATEFUL IS NECESSARY:
  User:  "What was our Q3 revenue?"      → routes to finance_agent ✓
  User:  "How does it compare to Q2?"    → this is a FOLLOW-UP
         Without history: router sees "How does it compare to Q2?"
         and may not classify it as finance → wrong agent
         With history: router sees full context → routes to finance_agent ✓

WARNING FROM THE DOCS:
  Stateful routers with multiple parallel agents are complex.
  If the router switches between agents across turns, conversations
  may feel inconsistent (different agent tones, different context).
  Consider handoffs or subagents instead for complex multi-turn flows.

INSTALL:
    pip install langchain-anthropic langgraph

RUN:
    export ANTHROPIC_API_KEY=your_key
    python 03_stateless_vs_stateful.py
"""

from typing import TypedDict, Annotated, Optional
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

model = ChatAnthropic(model="claude-sonnet-4-20250514")

# ------------------------------------------------------------------
# Shared: specialized agents
# ------------------------------------------------------------------

support_agent = create_react_agent(
    model=model, tools=[],
    prompt="You are a customer support specialist. Help with account and billing issues."
)

finance_agent = create_react_agent(
    model=model, tools=[],
    prompt="You are a financial analyst. Help with revenue, reports, and metrics."
)

docs_agent = create_react_agent(
    model=model, tools=[],
    prompt="You are a technical documentation specialist. Help with API and how-to questions."
)

AGENTS = {
    "support_agent": support_agent,
    "finance_agent": finance_agent,
    "docs_agent":    docs_agent,
}

# ==================================================================
# PART A — STATELESS ROUTER
# ==================================================================
# No memory. Each query classified from scratch.
# Fast, simple, cheap. Works for one-shot questions.

CLASSIFY_PROMPT_STATELESS = """Classify this query into one category.

Categories:
- support  : account issues, billing, troubleshooting
- finance  : revenue, reports, metrics, forecasts
- docs     : API, technical guides, code examples

Query: {query}

Reply with ONLY one word: support, finance, or docs.
"""

class StatelessState(TypedDict):
    query: str
    route: str
    answer: str

def stateless_router(state: StatelessState) -> Command:
    response = model.invoke([
        SystemMessage(content=CLASSIFY_PROMPT_STATELESS.format(query=state["query"]))
    ])
    raw = response.content.strip().lower()
    route_map = {"support": "support_agent", "finance": "finance_agent", "docs": "docs_agent"}
    destination = route_map.get(raw, "support_agent")
    return Command(update={"route": raw}, goto=destination)

def run_agent_stateless(agent_name: str):
    def node(state: StatelessState) -> dict:
        result = AGENTS[agent_name].invoke({
            "messages": [HumanMessage(content=state["query"])]
        })
        answer = next(
            (m.content for m in reversed(result["messages"])
             if isinstance(m, AIMessage) and m.content), ""
        )
        return {"answer": answer}
    return node

# Build stateless graph
def build_stateless_graph():
    b = StateGraph(StatelessState)
    b.add_node("router",        stateless_router)
    b.add_node("support_agent", run_agent_stateless("support_agent"))
    b.add_node("finance_agent", run_agent_stateless("finance_agent"))
    b.add_node("docs_agent",    run_agent_stateless("docs_agent"))
    b.add_edge(START, "router")
    b.add_edge("support_agent", END)
    b.add_edge("finance_agent", END)
    b.add_edge("docs_agent",    END)
    return b.compile()

stateless_graph = build_stateless_graph()

# ==================================================================
# PART B — STATEFUL ROUTER
# ==================================================================
# Maintains message history across turns.
# Router sees full conversation when classifying follow-up questions.

CLASSIFY_PROMPT_STATEFUL = """You are a query classifier. Classify the LATEST user message.

Use the conversation history to understand follow-up questions.
For example, if the conversation is about finance and the user asks
"how does it compare to last year?" — that's still a finance question.

Categories:
- support  : account issues, billing, troubleshooting
- finance  : revenue, reports, metrics, forecasts
- docs     : API, technical guides, code examples

Conversation history:
{history}

Latest query: {query}

Reply with ONLY one word: support, finance, or docs.
"""

class StatefulState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    answer: str

def stateful_router(state: StatefulState) -> Command:
    # Format conversation history for context
    history_lines = []
    for m in state["messages"][:-1]:   # all but the latest
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        history_lines.append(f"{role}: {m.content[:200]}")
    history_str = "\n".join(history_lines) if history_lines else "(no prior conversation)"

    # Latest query is the last message
    latest_query = state["messages"][-1].content

    response = model.invoke([
        SystemMessage(content=CLASSIFY_PROMPT_STATEFUL.format(
            history=history_str,
            query=latest_query,
        ))
    ])

    raw = response.content.strip().lower()
    route_map = {"support": "support_agent", "finance": "finance_agent", "docs": "docs_agent"}
    destination = route_map.get(raw, "support_agent")

    return Command(update={"route": raw}, goto=destination)

def run_agent_stateful(agent_name: str):
    def node(state: StatefulState) -> dict:
        # Pass full message history so agent has conversation context
        result = AGENTS[agent_name].invoke({"messages": state["messages"]})
        answer = next(
            (m.content for m in reversed(result["messages"])
             if isinstance(m, AIMessage) and m.content), ""
        )
        # Append agent reply to history for next turn
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
        }
    return node

# Build stateful graph
def build_stateful_graph():
    b = StateGraph(StatefulState)
    b.add_node("router",        stateful_router)
    b.add_node("support_agent", run_agent_stateful("support_agent"))
    b.add_node("finance_agent", run_agent_stateful("finance_agent"))
    b.add_node("docs_agent",    run_agent_stateful("docs_agent"))
    b.add_edge(START, "router")
    b.add_edge("support_agent", END)
    b.add_edge("finance_agent", END)
    b.add_edge("docs_agent",    END)
    return b.compile()

stateful_graph = build_stateful_graph()

# ------------------------------------------------------------------
# Demo — shows the difference on follow-up questions
# ------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("PART A — STATELESS ROUTER")
    print("Each query classified independently. No memory.\n")

    queries_a = [
        "What was our Q3 revenue?",
        "How does it compare to Q2?",     # ← follow-up: no context in stateless
    ]
    for q in queries_a:
        result = stateless_graph.invoke({"query": q, "route": "", "answer": ""})
        print(f"  QUERY:  {q}")
        print(f"  ROUTED: {result['route']}")
        print(f"  ANSWER: {result['answer'][:150]}...\n")

    print("=" * 60)
    print("PART B — STATEFUL ROUTER")
    print("Router sees full history. Follow-ups routed correctly.\n")

    state: StatefulState = {"messages": [], "route": "", "answer": ""}

    queries_b = [
        "What was our Q3 revenue?",
        "How does it compare to Q2?",     # ← follow-up: history gives context
        "Can I also get the API docs for our reporting endpoint?",  # ← domain switch
    ]

    for q in queries_b:
        state["messages"].append(HumanMessage(content=q))
        result = stateful_graph.invoke(state)
        state = result
        print(f"  QUERY:  {q}")
        print(f"  ROUTED: {result['route']}")
        print(f"  ANSWER: {result['answer'][:150]}...\n")