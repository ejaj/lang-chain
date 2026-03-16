"""
TOPIC: runtime.stream_writer — Live Updates from Inside Tools

WHAT IS STREAM WRITER:
    A callable that pushes ARBITRARY data into the agent's stream in real time,
    from inside a tool or middleware, BEFORE the tool has even finished.

    Without stream_writer: you wait for the tool to return, then see the result.
    With stream_writer:    you see progress updates as the tool runs.

HOW TO USE IN TOOLS:
    @tool
    def my_tool(runtime: ToolRuntime[Context]) -> str:
        writer = runtime.stream_writer
        writer({"event": "step", "msg": "Starting..."})   # fires immediately
        # ... do work ...
        writer({"event": "step", "msg": "Done!"})
        return "final result"

HOW TO CONSUME:
    Stream with stream_mode="custom" (or include "custom" in a list).
    Filter chunks by chunk["type"] == "custom".
    The data you passed to writer() is in chunk["data"].

HOW TO USE IN MIDDLEWARE:
    In node-style hooks: call get_stream_writer() from langgraph.config
    In wrap-style hooks: request.runtime.stream_writer

WHAT YOU CAN PUSH:
    Anything JSON-serializable:
        strings, dicts, lists, numbers, booleans
    Common patterns:
        progress dicts {"step": N, "total": N, "pct": 50}
        status strings "Connecting to database..."
        partial results {"found": 42, "records": [...]}
        structured events {"event": "warning", "msg": "..."}

CAVEAT:
    Tools that use runtime.stream_writer can only run inside a LangGraph
    execution context (i.e. via an agent). They cannot be called standalone.

WHEN TO USE:
    ✅ Long-running tools (web scraping, ETL, large file processing)
    ✅ Multi-step tools where progress is valuable to the user
    ✅ Real-time dashboards that display agent activity
    ✅ Debugging — see what tools are doing internally
"""

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage
import time


# ---------------------------------------------------------------------------
# Context schema
# ---------------------------------------------------------------------------
@dataclass
class PipelineContext:
    user_id:    str
    job_name:   str


# ---------------------------------------------------------------------------
# TOOL 1: Simple progress bar — integer progress updates
# ---------------------------------------------------------------------------
@tool
def process_records(
    source: str,
    runtime: ToolRuntime[PipelineContext],
) -> str:
    """Process records from a source, streaming progress."""
    writer = runtime.stream_writer
    total = 5   # simulate 5 batches

    writer({"event": "start", "source": source, "total_batches": total})

    results = []
    for i in range(1, total + 1):
        # Simulate work
        batch_result = f"batch_{i}_from_{source}"
        results.append(batch_result)

        # Push progress BEFORE the loop finishes
        writer({
            "event":   "batch_done",
            "batch":   i,
            "total":   total,
            "pct":     int(i / total * 100),
            "result":  batch_result,
        })

    writer({"event": "done", "processed": total})
    return f"Processed {total} batches from {source}: {results}"


# ---------------------------------------------------------------------------
# TOOL 2: Multi-phase pipeline with named phases
# ---------------------------------------------------------------------------
@tool
def run_data_pipeline(
    pipeline_name: str,
    runtime: ToolRuntime[PipelineContext],
) -> str:
    """Run a multi-phase data pipeline with per-phase status updates."""
    writer = runtime.stream_writer

    phases = [
        ("validate",   "Validating input data..."),
        ("transform",  "Transforming records..."),
        ("enrich",     "Enriching with external data..."),
        ("load",       "Loading into destination..."),
    ]

    writer({"event": "pipeline_start", "name": pipeline_name, "phases": len(phases)})

    for idx, (phase_id, phase_msg) in enumerate(phases, 1):
        writer({
            "event":       "phase_start",
            "phase":       phase_id,
            "message":     phase_msg,
            "step":        idx,
            "total_steps": len(phases),
        })

        # Simulate phase work (no real sleep in prod)

        writer({
            "event":   "phase_done",
            "phase":   phase_id,
            "step":    idx,
            "success": True,
        })

    writer({"event": "pipeline_done", "name": pipeline_name})
    return f"Pipeline '{pipeline_name}' completed all {len(phases)} phases successfully."


# ---------------------------------------------------------------------------
# TOOL 3: Partial results streaming — send found items as they arrive
# ---------------------------------------------------------------------------
@tool
def search_knowledge_base(
    query: str,
    runtime: ToolRuntime[PipelineContext],
) -> str:
    """Search a knowledge base, streaming found results in real time."""
    writer = runtime.stream_writer

    writer({"event": "search_start", "query": query})

    # Simulate finding results one by one
    fake_results = [
        {"id": "doc_001", "title": "Introduction to Python",     "score": 0.95},
        {"id": "doc_042", "title": "Async Programming Guide",    "score": 0.87},
        {"id": "doc_018", "title": "LangChain Best Practices",   "score": 0.82},
    ]

    for result in fake_results:
        # Send each result as it's "found" — user sees them arrive live
        writer({"event": "result_found", "result": result})

    writer({"event": "search_done", "total_found": len(fake_results)})
    titles = [r["title"] for r in fake_results]
    return f"Found {len(fake_results)} results for '{query}': {titles}"


# ---------------------------------------------------------------------------
# Create agent
# ---------------------------------------------------------------------------
agent = create_agent(
    model="gpt-5-nano",
    tools=[process_records, run_data_pipeline, search_knowledge_base],
    context_schema=PipelineContext,
)

context = PipelineContext(user_id="u-demo", job_name="demo_job")


# ---------------------------------------------------------------------------
# Consume the custom stream — pretty-print each event type
# ---------------------------------------------------------------------------
def render_custom_event(data: dict) -> None:
    """Render a custom stream event to the console."""
    event = data.get("event", "unknown")

    if event == "start":
        print(f"  🚀 Starting: {data.get('source')}, {data.get('total_batches')} batches")
    elif event == "batch_done":
        bar = "█" * data["batch"] + "░" * (data["total"] - data["batch"])
        print(f"  [{bar}] {data['pct']}% — {data['result']}")
    elif event == "done":
        print(f"  ✅ Done! {data.get('processed')} batches processed")

    elif event == "pipeline_start":
        print(f"  🔧 Pipeline '{data['name']}' starting ({data['phases']} phases)")
    elif event == "phase_start":
        print(f"  [{data['step']}/{data['total_steps']}] ⏳ {data['message']}")
    elif event == "phase_done":
        mark = "✅" if data["success"] else "❌"
        print(f"  [{data['step']}] {mark} Phase '{data['phase']}' complete")
    elif event == "pipeline_done":
        print(f"  🎉 Pipeline '{data['name']}' all done!")

    elif event == "search_start":
        print(f"  🔍 Searching for: '{data['query']}'")
    elif event == "result_found":
        r = data["result"]
        print(f"  📄 Found: {r['title']} (score={r['score']})")
    elif event == "search_done":
        print(f"  ✅ Search complete — {data['total_found']} results")


# ---------------------------------------------------------------------------
# TEST 1: Process records with progress bar
# ---------------------------------------------------------------------------
print("=" * 60)
print("TEST 1: stream_writer — progress bar")
print("=" * 60)

for chunk in agent.stream(
    {"messages": [HumanMessage("Process records from the sales_db source")]},
    context=context,
    stream_mode=["custom", "updates"],
    version="v2",
):
    if chunk["type"] == "custom":
        render_custom_event(chunk["data"])
    elif chunk["type"] == "updates":
        for src, upd in chunk["data"].items():
            if src == "model" and upd["messages"][-1].content:
                print(f"\n  🤖 {upd['messages'][-1].content[:200]}")


# ---------------------------------------------------------------------------
# TEST 2: Multi-phase pipeline
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 2: stream_writer — multi-phase pipeline")
print("=" * 60)

for chunk in agent.stream(
    {"messages": [HumanMessage("Run the ETL pipeline")]},
    context=context,
    stream_mode=["custom", "updates"],
    version="v2",
):
    if chunk["type"] == "custom":
        render_custom_event(chunk["data"])
    elif chunk["type"] == "updates":
        for src, upd in chunk["data"].items():
            if src == "model" and upd["messages"][-1].content:
                print(f"\n  🤖 {upd['messages'][-1].content[:200]}")


# ---------------------------------------------------------------------------
# TEST 3: Streaming search results as they arrive
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 3: stream_writer — search results arriving live")
print("=" * 60)

for chunk in agent.stream(
    {"messages": [HumanMessage("Search the knowledge base for 'Python async'")]},
    context=context,
    stream_mode=["custom", "updates"],
    version="v2",
):
    if chunk["type"] == "custom":
        render_custom_event(chunk["data"])
    elif chunk["type"] == "updates":
        for src, upd in chunk["data"].items():
            if src == "model" and upd["messages"][-1].content:
                print(f"\n  🤖 {upd['messages'][-1].content[:200]}")