"""
Conditional branching
=====================================================

WHAT IS IT?
-----------
A node inspects the current state and returns a string that
determines which node runs next. This creates different execution
paths through the graph based on conditions.

LangGraph calls this a "conditional edge" — instead of a fixed edge
from A to B, a function decides which node to go to.

WHEN TO USE:
- Different input types need different processing
- A quality check might approve or reject and retry
- A classifier splits work into distinct paths
- You want to skip steps that aren't relevant

SCENARIO: A document processing pipeline.
  Step 1 — classify : what type is this document? (contract / invoice / report)
  Step 2 — branch   :
    contract → extract parties, dates, clauses
    invoice  → extract vendor, amount, line items
    report   → extract summary, key metrics
  Step 3 — format   : format the extracted data as JSON

Each document type gets a specialized extraction node.
The classifier decides which extractor to use.

INSTALL:
    pip install langchain-anthropic langgraph

RUN:
    export ANTHROPIC_API_KEY=your_key
    python 02_conditional_branching.py
"""

from typing import TypedDict, Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

class DocState(TypedDict):
    document: str               # raw document text
    doc_type: str               # filled by classifier: contract/invoice/report
    extracted_data: str         # filled by the appropriate extractor
    formatted_output: str       # filled by formatter

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
# Node 1 — Classifier
# ------------------------------------------------------------------

def classify_node(state: DocState) -> dict:
    """Determine the document type."""
    doc_type = llm(
        system=(
            "Classify this document into exactly one category.\n"
            "Reply with ONLY one word: contract, invoice, or report."
        ),
        user=f"Document:\n{state['document'][:500]}"
    )
    # Normalise to known types
    doc_type = doc_type.lower().strip()
    if doc_type not in ("contract", "invoice", "report"):
        doc_type = "report"   # safe default

    print(f"  [classify] doc_type = {doc_type}")
    return {"doc_type": doc_type}

# ------------------------------------------------------------------
# Decision function — reads doc_type, returns node name to go to
# ------------------------------------------------------------------

def route_by_type(state: DocState) -> Literal["extract_contract", "extract_invoice", "extract_report"]:
    """Return the name of the next node based on doc_type."""
    return f"extract_{state['doc_type']}"

# ------------------------------------------------------------------
# Node 2a — Contract extractor
# ------------------------------------------------------------------

def extract_contract_node(state: DocState) -> dict:
    data = llm(
        system=(
            "You extract key information from legal contracts.\n"
            "Extract and list: parties involved, effective date, "
            "term length, payment terms, termination clause, governing law.\n"
            "Use clear labels for each field."
        ),
        user=f"Contract:\n{state['document']}"
    )
    print(f"  [extract_contract] done")
    return {"extracted_data": data}

# ------------------------------------------------------------------
# Node 2b — Invoice extractor
# ------------------------------------------------------------------

def extract_invoice_node(state: DocState) -> dict:
    data = llm(
        system=(
            "You extract key information from invoices.\n"
            "Extract and list: vendor name, invoice number, invoice date, "
            "due date, line items (description + amount), subtotal, tax, total.\n"
            "Use clear labels for each field."
        ),
        user=f"Invoice:\n{state['document']}"
    )
    print(f"  [extract_invoice] done")
    return {"extracted_data": data}

# ------------------------------------------------------------------
# Node 2c — Report extractor
# ------------------------------------------------------------------

def extract_report_node(state: DocState) -> dict:
    data = llm(
        system=(
            "You extract key information from business reports.\n"
            "Extract and list: report title, date, author/team, "
            "executive summary (2-3 sentences), key metrics or findings, "
            "main recommendations.\n"
            "Use clear labels for each field."
        ),
        user=f"Report:\n{state['document']}"
    )
    print(f"  [extract_report] done")
    return {"extracted_data": data}

# ------------------------------------------------------------------
# Node 3 — Formatter (shared by all paths)
# ------------------------------------------------------------------

def format_node(state: DocState) -> dict:
    """Format the extracted data as a clean JSON-like structure."""
    formatted = llm(
        system=(
            "You format extracted document data into clean, readable output.\n"
            "Produce a well-structured summary with clear field labels.\n"
            "Add a 'document_type' field at the top.\n"
            "Keep it concise and professional."
        ),
        user=(
            f"Document type: {state['doc_type']}\n\n"
            f"Extracted data:\n{state['extracted_data']}"
        )
    )
    print(f"  [format] done")
    return {"formatted_output": formatted}

# ------------------------------------------------------------------
# Graph
# ------------------------------------------------------------------

builder = StateGraph(DocState)

builder.add_node("classify",         classify_node)
builder.add_node("extract_contract", extract_contract_node)
builder.add_node("extract_invoice",  extract_invoice_node)
builder.add_node("extract_report",   extract_report_node)
builder.add_node("format",           format_node)

builder.add_edge(START, "classify")

# Conditional edge: classify → one of three extractors
builder.add_conditional_edges(
    "classify",
    route_by_type,
    {
        "extract_contract": "extract_contract",
        "extract_invoice":  "extract_invoice",
        "extract_report":   "extract_report",
    }
)

# All three extractors converge to the same formatter
builder.add_edge("extract_contract", "format")
builder.add_edge("extract_invoice",  "format")
builder.add_edge("extract_report",   "format")
builder.add_edge("format",           END)

graph = builder.compile()

# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

SAMPLE_CONTRACT = """
SERVICE AGREEMENT between Acme Corp ("Client") and Dev Studio ("Provider").
Effective Date: January 1, 2025. Term: 12 months.
Payment: $10,000/month, due within 30 days of invoice.
Liability: capped at one month's fees.
Termination: 30 days written notice by either party.
Governing law: State of Delaware.
"""

SAMPLE_INVOICE = """
INVOICE #INV-2025-042
From: Cloud Hosting Ltd
To: Acme Corp
Invoice Date: March 1, 2025 | Due Date: March 31, 2025
Line items:
  - Server hosting (10 nodes × $200): $2,000
  - Bandwidth overage (500GB): $250
  - Support plan (Standard): $300
Subtotal: $2,550 | Tax (10%): $255 | Total Due: $2,805
"""

SAMPLE_REPORT = """
Q1 2025 PERFORMANCE REPORT — Engineering Team
Author: CTO Office | Date: April 1, 2025
Summary: Engineering delivered 94% of planned features in Q1, with system
uptime at 99.8%. The team grew from 12 to 15 engineers.
Key metrics: 47 features shipped, 12ms avg API latency, 3 critical bugs resolved.
Recommendations: invest in observability tooling, hire 2 more senior engineers.
"""

if __name__ == "__main__":
    print("=== Conditional Branching Demo ===\n")

    for label, doc in [
        ("CONTRACT", SAMPLE_CONTRACT),
        ("INVOICE",  SAMPLE_INVOICE),
        ("REPORT",   SAMPLE_REPORT),
    ]:
        print(f"--- Processing: {label} ---")
        result = graph.invoke({
            "document": doc,
            "doc_type": "",
            "extracted_data": "",
            "formatted_output": "",
        })
        print(f"\nFormatted output:\n{result['formatted_output']}\n")
        print("=" * 60 + "\n")