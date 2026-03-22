"""
Embedding other patterns as nodes
============================================================

WHAT IS IT?
-----------
A custom workflow can embed any other pattern as a node.
A subagent, a router, a skills agent — each is just a compiled
LangGraph object. You call .invoke() on it inside a node function,
the same way you call any Python function.

This lets you build complex pipelines where some steps are
deterministic (plain functions) and others are fully agentic
(with their own tools, memory, and sub-graphs).

WHEN TO USE:
- You need a pipeline that mixes deterministic steps with agent steps
- Some stages are complex enough to need their own tools/logic
- You want to reuse an existing agent in a larger workflow
- Different teams build different pipeline stages independently

CORE INSIGHT FROM THE DOCS:
  "Each node in your workflow can be a simple function, an LLM call,
   or an entire agent with tools. You can also compose other
   architectures within a custom workflow — for example, embedding
   a multi-agent system as a single node."

SCENARIO: A customer feedback processing pipeline.
  Step 1 — ingest        : (deterministic) parse and clean raw feedback
  Step 2 — sentiment     : (LLM call) classify sentiment
  Step 3 — specialist    : (embedded subagents) route to product/support/billing
                           team for a detailed response draft
  Step 4 — format        : (deterministic) format final output as JSON

The "specialist" step is a full subagents pattern embedded as one node.

INSTALL:
    pip install langchain-anthropic langgraph

RUN:
    export ANTHROPIC_API_KEY=your_key
    python 05_composed_patterns.py
"""

import json
from typing import TypedDict, Literal, Annotated
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

# ------------------------------------------------------------------
# Shared model
# ------------------------------------------------------------------

model = ChatAnthropic(model="claude-sonnet-4-20250514")

def llm(system: str, user: str) -> str:
    r = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return r.content.strip()

# ==================================================================
# INNER PATTERN: Subagents embedded inside the pipeline
# Three specialist agents + a supervisor that routes between them
# ==================================================================

@tool
def lookup_product_info(product_name: str) -> str:
    """Look up product details and known issues."""
    return f"Product '{product_name}': version 2.3, last updated March 2025, no open critical bugs."

@tool
def lookup_billing_history(customer_id: str) -> str:
    """Look up a customer's billing history."""
    return f"Customer {customer_id}: active since 2023, paid on time, no disputes."

@tool
def lookup_support_tickets(customer_id: str) -> str:
    """Look up previous support tickets for a customer."""
    return f"Customer {customer_id}: 2 previous tickets (both resolved), avg resolution 2 days."

product_agent = create_react_agent(
    model=model,
    tools=[lookup_product_info],
    prompt=(
        "You are a product specialist. Draft a helpful response to product-related feedback. "
        "Use lookup_product_info to get accurate product details. "
        "Be specific, empathetic, and actionable."
    )
)

support_agent = create_react_agent(
    model=model,
    tools=[lookup_support_tickets],
    prompt=(
        "You are a customer support specialist. Draft a helpful response to support issues. "
        "Use lookup_support_tickets to check history. "
        "Be empathetic, clear, and provide next steps."
    )
)

billing_agent = create_react_agent(
    model=model,
    tools=[lookup_billing_history],
    prompt=(
        "You are a billing specialist. Draft a helpful response to billing concerns. "
        "Use lookup_billing_history to check the customer's account. "
        "Be professional and resolution-focused."
    )
)

SPECIALIST_AGENTS = {
    "product": product_agent,
    "support": support_agent,
    "billing": billing_agent,
}

# ------------------------------------------------------------------
# Outer pipeline state
# ------------------------------------------------------------------

class FeedbackState(TypedDict):
    raw_feedback: str       # input: raw customer feedback text
    customer_id: str        # input: customer identifier
    cleaned_text: str       # step 1: cleaned/parsed feedback
    sentiment: str          # step 2: positive/negative/neutral
    category: str           # step 2: product/support/billing
    response_draft: str     # step 3: specialist response draft
    final_output: str       # step 4: formatted JSON output

# ------------------------------------------------------------------
# Node 1 — Ingest (deterministic)
# Plain Python — no LLM needed, just clean the text
# ------------------------------------------------------------------

def ingest_node(state: FeedbackState) -> dict:
    """Clean and normalize the raw feedback text."""
    text = state["raw_feedback"].strip()
    # Remove excessive whitespace
    text = " ".join(text.split())
    # Truncate if too long
    if len(text) > 1000:
        text = text[:1000] + "..."
    print(f"  [ingest] cleaned — {len(text)} chars")
    return {"cleaned_text": text}

# ------------------------------------------------------------------
# Node 2 — Classify (single LLM call)
# Sentiment + category in one call
# ------------------------------------------------------------------

def classify_node(state: FeedbackState) -> dict:
    """Classify sentiment and category of the feedback."""
    response = llm(
        system=(
            "Classify customer feedback. Reply with ONLY two words separated by a comma.\n"
            "Word 1 — sentiment: positive, negative, or neutral\n"
            "Word 2 — category: product, support, or billing\n"
            "Example: negative,billing"
        ),
        user=f"Feedback: {state['cleaned_text']}"
    )

    parts = [p.strip().lower() for p in response.split(",")]
    sentiment = parts[0] if parts[0] in ("positive", "negative", "neutral") else "neutral"
    category  = parts[1] if len(parts) > 1 and parts[1] in ("product", "support", "billing") else "support"

    print(f"  [classify] sentiment={sentiment}, category={category}")
    return {"sentiment": sentiment, "category": category}

# ------------------------------------------------------------------
# Node 3 — Specialist response (embedded subagents)
# A full create_react_agent called inside a pipeline node
# ------------------------------------------------------------------

def specialist_node(state: FeedbackState) -> dict:
    """
    Route to the appropriate specialist agent and get a response draft.
    This is a full subagent (create_react_agent) embedded as one pipeline step.
    """
    agent = SPECIALIST_AGENTS[state["category"]]

    print(f"  [specialist] invoking {state['category']} agent")

    result = agent.invoke({
        "messages": [
            HumanMessage(content=(
                f"Customer ID: {state['customer_id']}\n"
                f"Sentiment: {state['sentiment']}\n"
                f"Feedback: {state['cleaned_text']}\n\n"
                f"Draft a professional response to this feedback."
            ))
        ]
    })

    draft = next(
        (m.content for m in reversed(result["messages"])
         if isinstance(m, AIMessage) and m.content),
        "(no draft generated)"
    )

    print(f"  [specialist] draft ready — {len(draft)} chars")
    return {"response_draft": draft}

# ------------------------------------------------------------------
# Node 4 — Format output (deterministic)
# Plain Python — structure the data, no LLM needed
# ------------------------------------------------------------------

def format_node(state: FeedbackState) -> dict:
    """Format all collected data into a structured JSON output."""
    output = {
        "customer_id":     state["customer_id"],
        "sentiment":       state["sentiment"],
        "category":        state["category"],
        "original_feedback": state["raw_feedback"][:200],
        "response_draft":  state["response_draft"],
        "processing_steps": ["ingest", "classify", "specialist_response", "format"],
    }
    formatted = json.dumps(output, indent=2)
    print(f"  [format] output ready")
    return {"final_output": formatted}

# ------------------------------------------------------------------
# Graph — four nodes, straight pipeline
# (branching or parallel could easily be added at any step)
# ------------------------------------------------------------------

builder = StateGraph(FeedbackState)

builder.add_node("ingest",     ingest_node)
builder.add_node("classify",   classify_node)
builder.add_node("specialist", specialist_node)
builder.add_node("format",     format_node)

builder.add_edge(START,        "ingest")
builder.add_edge("ingest",     "classify")
builder.add_edge("classify",   "specialist")
builder.add_edge("specialist", "format")
builder.add_edge("format",     END)

graph = builder.compile()

# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

FEEDBACKS = [
    {
        "customer_id": "cust_001",
        "raw_feedback": "Your latest app update broke the export feature completely. "
                        "I've lost 3 hours of work because of this bug. "
                        "This is unacceptable for a paid product.",
    },
    {
        "customer_id": "cust_002",
        "raw_feedback": "I love the new dashboard! The analytics view is exactly "
                        "what I've been asking for. Really impressed with the improvement.",
    },
    {
        "customer_id": "cust_003",
        "raw_feedback": "I was charged twice for my subscription this month. "
                        "Please refund the duplicate charge immediately.",
    },
]

if __name__ == "__main__":
    print("=== Composed Patterns Demo ===")
    print("Pipeline: ingest → classify → specialist_agent → format\n")

    for fb in FEEDBACKS:
        print(f"--- Processing feedback from {fb['customer_id']} ---")
        print(f"    \"{fb['raw_feedback'][:80]}...\"\n")

        result = graph.invoke({
            "raw_feedback":    fb["raw_feedback"],
            "customer_id":     fb["customer_id"],
            "cleaned_text":    "",
            "sentiment":       "",
            "category":        "",
            "response_draft":  "",
            "final_output":    "",
        })

        output = json.loads(result["final_output"])
        print(f"\nSentiment: {output['sentiment']} | Category: {output['category']}")
        print(f"Response draft:\n{output['response_draft'][:300]}...")
        print("\n" + "=" * 60 + "\n")