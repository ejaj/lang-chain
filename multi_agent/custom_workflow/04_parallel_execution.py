"""
Parallel execution with Send
========================================================

WHAT IS IT?
-----------
Multiple nodes run at the same time using Send.
Instead of processing items one by one (sequential),
all items are dispatched simultaneously and results collected
when all workers finish.

WHEN TO USE:
- Independent tasks that don't depend on each other
- Processing a list of items (documents, queries, URLs)
- Querying multiple data sources simultaneously
- Any situation where sequential execution is unnecessarily slow

Send vs sequential loop:
  Sequential: item1 → item2 → item3 (total time = sum of all)
  Parallel:   item1 ↘
              item2 → merge (total time ≈ slowest item)
              item3 ↗

IMPORTANT:
- Parallel nodes cannot share mutable state during execution
- Results are merged via a reducer function on the state field
- Use Annotated[list[X], lambda a, b: a + b] to collect results

SCENARIO: Analyse multiple company documents in parallel.
  Input: 3 documents (annual report, risk report, ESG report)
  Each document is analysed by the same worker node simultaneously.
  Results collected and summarised into one combined report.

INSTALL:
    pip install langchain-anthropic langgraph

RUN:
    export ANTHROPIC_API_KEY=your_key
    python 04_parallel_execution.py
"""

from typing import TypedDict, Annotated
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# ------------------------------------------------------------------
# State schemas
# ------------------------------------------------------------------

class PipelineState(TypedDict):
    """Top-level graph state."""
    documents: list[dict]           # list of {name, content}
    analyses: Annotated[            # collected from parallel workers
        list[str],
        lambda a, b: a + b          # reducer: append results as they arrive
    ]
    final_summary: str              # produced by synthesizer

class WorkerState(TypedDict):
    """State passed to each parallel worker."""
    doc_name: str
    doc_content: str

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

model = ChatAnthropic(model="claude-sonnet-4-20250514")

def llm(system: str, user: str) -> str:
    response = model.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return response.content.strip()

# ------------------------------------------------------------------
# Fan-out node — spawns one worker per document
# ------------------------------------------------------------------

def dispatch_documents(state: PipelineState) -> list[Send]:
    """
    Create one Send per document.
    All workers start simultaneously — no waiting between them.
    """
    return [
        Send("analyse_document", {
            "doc_name":    doc["name"],
            "doc_content": doc["content"],
        })
        for doc in state["documents"]
    ]

# ------------------------------------------------------------------
# Worker node — runs once per document, all in parallel
# ------------------------------------------------------------------

def analyse_document_node(state: WorkerState) -> dict:
    """Analyse a single document. Called in parallel for each document."""
    print(f"  [analyse] starting: {state['doc_name']}")

    analysis = llm(
        system=(
            "You are a business analyst. Analyse this document and extract:\n"
            "1. Main purpose/type of document\n"
            "2. Key findings or metrics (top 3)\n"
            "3. Notable risks or concerns\n"
            "4. One-line overall assessment\n"
            "Be concise — max 150 words."
        ),
        user=(
            f"Document: {state['doc_name']}\n\n"
            f"{state['doc_content']}"
        )
    )

    print(f"  [analyse] done: {state['doc_name']}")
    # Return value is APPENDED to state["analyses"] via the reducer
    return {"analyses": [f"=== {state['doc_name']} ===\n{analysis}"]}

# ------------------------------------------------------------------
# Synthesizer node — runs after all workers complete
# ------------------------------------------------------------------

def synthesize_node(state: PipelineState) -> dict:
    """Combine all individual analyses into one summary report."""
    combined = "\n\n".join(state["analyses"])

    summary = llm(
        system=(
            "You are a senior analyst. You have received analyses of multiple "
            "company documents. Write a concise executive summary that:\n"
            "1. Identifies common themes across all documents\n"
            "2. Highlights the most important findings overall\n"
            "3. Notes any contradictions or concerns\n"
            "4. Gives an overall assessment in 2-3 sentences\n"
            "Max 200 words."
        ),
        user=f"Individual document analyses:\n\n{combined}"
    )

    print(f"  [synthesize] done")
    return {"final_summary": summary}

# ------------------------------------------------------------------
# Graph — dispatch → parallel workers → synthesize
# ------------------------------------------------------------------

builder = StateGraph(PipelineState)

builder.add_node("analyse_document", analyse_document_node)
builder.add_node("synthesize",       synthesize_node)

# Fan-out: START → dispatch_documents → multiple analyse_document nodes in parallel
builder.add_conditional_edges(START, dispatch_documents, ["analyse_document"])

# Fan-in: all workers → synthesize (runs when all workers are done)
builder.add_edge("analyse_document", "synthesize")
builder.add_edge("synthesize",       END)

graph = builder.compile()

# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

DOCUMENTS = [
    {
        "name": "Annual Report 2024",
        "content": (
            "Acme Corp Annual Report 2024. Revenue: $125M (+18% YoY). "
            "Net profit: $22M (+5%). Headcount grew from 450 to 580. "
            "Launched 3 new product lines. Entered APAC market. "
            "Key challenge: rising infrastructure costs (+35%). "
            "Outlook: expecting 20% revenue growth in 2025."
        )
    },
    {
        "name": "Risk Assessment Q4 2024",
        "content": (
            "Top risks identified: (1) Customer concentration — top 3 clients "
            "represent 41% of revenue; (2) Regulatory risk in EU market — new "
            "data privacy law takes effect Q2 2025; (3) Key person dependency — "
            "CTO and VP Engineering both at retention risk. Mitigation plans in "
            "progress. Overall risk level: MEDIUM-HIGH."
        )
    },
    {
        "name": "ESG Report 2024",
        "content": (
            "Environmental: reduced carbon emissions by 12% vs 2023 through "
            "data center efficiency improvements. Social: 45% female leadership "
            "(up from 38%). Launched apprenticeship program (24 participants). "
            "Governance: added 2 independent board members. Whistleblower policy "
            "updated. ESG score from third-party: B+ (up from B)."
        )
    },
]

if __name__ == "__main__":
    print("=== Parallel Execution Demo ===")
    print(f"Processing {len(DOCUMENTS)} documents simultaneously...\n")

    result = graph.invoke({
        "documents": DOCUMENTS,
        "analyses": [],
        "final_summary": "",
    })

    print("\n=== INDIVIDUAL ANALYSES ===\n")
    for analysis in result["analyses"]:
        print(analysis)
        print()

    print("=== EXECUTIVE SUMMARY ===\n")
    print(result["final_summary"])
    print(f"\n(All {len(DOCUMENTS)} documents processed in parallel)")