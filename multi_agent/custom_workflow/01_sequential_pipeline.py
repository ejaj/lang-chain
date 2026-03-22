"""
Sequential pipeline
=================================================

WHAT IS IT?
-----------
Nodes run one after another in a fixed order.
The output of each node is passed to the next via shared state.
This is the simplest custom workflow — a straight line.

WHEN TO USE:
- Tasks that must happen in order: fetch → process → format → deliver
- Each step depends on the output of the previous step
- No branching or looping needed

SCENARIO: A content generation pipeline.
  Step 1 — research    : gather facts on the topic
  Step 2 — outline     : create a structured outline from the facts
  Step 3 — write       : write the article from the outline
  Step 4 — proofread   : fix grammar and style issues
  → Output: polished article

Each step is a separate LLM call with a focused prompt.
Separation of concerns: each node does one thing well.

INSTALL:
    pip install langchain-anthropic langgraph

RUN:
    export ANTHROPIC_API_KEY=your_key
    python 01_sequential_pipeline.py
"""

from typing import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# ------------------------------------------------------------------
# State — accumulates data as it flows through the pipeline
# ------------------------------------------------------------------

class ContentState(TypedDict):
    topic: str          # input: the topic to write about
    facts: str          # filled by research node
    outline: str        # filled by outline node
    draft: str          # filled by write node
    final_article: str  # filled by proofread node

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

model = ChatAnthropic(model="claude-sonnet-4-20250514")

def llm(system: str, user: str) -> str:
    """Helper: single LLM call, return text."""
    response = model.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return response.content

# ------------------------------------------------------------------
# Node 1 — Research
# ------------------------------------------------------------------

def research_node(state: ContentState) -> dict:
    """Gather key facts and background on the topic."""
    facts = llm(
        system=(
            "You are a research assistant. Given a topic, produce a concise "
            "list of 5-8 key facts, statistics, and background points. "
            "Be accurate and specific. Use bullet points."
        ),
        user=f"Research this topic: {state['topic']}"
    )
    print(f"  [research] done — {len(facts)} chars")
    return {"facts": facts}

# ------------------------------------------------------------------
# Node 2 — Outline
# ------------------------------------------------------------------

def outline_node(state: ContentState) -> dict:
    """Turn research facts into a structured article outline."""
    outline = llm(
        system=(
            "You are a content strategist. Given research facts, "
            "create a clear article outline with: "
            "Introduction, 3-4 main sections with subpoints, Conclusion. "
            "Use heading and bullet format."
        ),
        user=(
            f"Topic: {state['topic']}\n\n"
            f"Research facts:\n{state['facts']}"
        )
    )
    print(f"  [outline] done — {len(outline)} chars")
    return {"outline": outline}

# ------------------------------------------------------------------
# Node 3 — Write
# ------------------------------------------------------------------

def write_node(state: ContentState) -> dict:
    """Write a full article draft from the outline."""
    draft = llm(
        system=(
            "You are a skilled content writer. "
            "Write a complete, engaging article following the provided outline. "
            "Use clear language, concrete examples, and smooth transitions. "
            "Target length: 400-600 words."
        ),
        user=(
            f"Topic: {state['topic']}\n\n"
            f"Outline:\n{state['outline']}"
        )
    )
    print(f"  [write] done — {len(draft)} chars")
    return {"draft": draft}

# ------------------------------------------------------------------
# Node 4 — Proofread
# ------------------------------------------------------------------

def proofread_node(state: ContentState) -> dict:
    """Fix grammar, style, and clarity issues in the draft."""
    final = llm(
        system=(
            "You are a professional editor. "
            "Fix grammar, spelling, punctuation, and style issues. "
            "Improve sentence flow and word choice where needed. "
            "Do NOT change the structure or content — only polish the language. "
            "Return only the corrected article, no commentary."
        ),
        user=f"Proofread and polish this article:\n\n{state['draft']}"
    )
    print(f"  [proofread] done — {len(final)} chars")
    return {"final_article": final}

# ------------------------------------------------------------------
# Graph — straight line: research → outline → write → proofread
# ------------------------------------------------------------------

builder = StateGraph(ContentState)

builder.add_node("research",  research_node)
builder.add_node("outline",   outline_node)
builder.add_node("write",     write_node)
builder.add_node("proofread", proofread_node)

builder.add_edge(START,       "research")
builder.add_edge("research",  "outline")
builder.add_edge("outline",   "write")
builder.add_edge("write",     "proofread")
builder.add_edge("proofread", END)

graph = builder.compile()

# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Sequential Pipeline Demo ===\n")
    print("Pipeline: research → outline → write → proofread\n")

    result = graph.invoke({
        "topic": "How quantum computing will change cybersecurity",
        "facts": "",
        "outline": "",
        "draft": "",
        "final_article": "",
    })

    print("\n=== FINAL ARTICLE ===\n")
    print(result["final_article"])

    print("\n=== PIPELINE STAGES ===")
    print(f"Facts length:   {len(result['facts'])} chars")
    print(f"Outline length: {len(result['outline'])} chars")
    print(f"Draft length:   {len(result['draft'])} chars")
    print(f"Final length:   {len(result['final_article'])} chars")