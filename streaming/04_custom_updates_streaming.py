"""
==============================
TOPIC: Streaming Custom Updates (stream_mode="custom")

WHAT IT DOES:
    Lets you emit ANY arbitrary data from inside a tool or graph node.
    Think of it like a WebSocket you can write to from inside your tool code.

HOW IT WORKS:
    - Import `get_stream_writer` from `langgraph.config`
    - Call it inside your tool to get a `writer` callable
    - Call `writer(anything)` to push data into the stream
    - Outside, filter chunks for type="custom" and read chunk["data"]

WHEN TO USE:
    You want progress updates from inside a long-running tool
       e.g. "Fetched 10/100 records", "Scraped page 3/20"
    You want to surface intermediate results before the tool finishes
    You want to send structured metadata (dicts, dataclasses) mid-stream

⚠️  CAVEAT:
    Using `get_stream_writer()` inside a tool means the tool can ONLY be
    called from within a LangGraph execution context (not standalone).
"""

from langchain.agents import create_agent
from langgraph.config import get_stream_writer


# ---------------------------------------------------------------------------
# 1. Define a tool that emits custom stream events
# ---------------------------------------------------------------------------
def fetch_records(source: str) -> str:
    """Fetch records from a data source, emitting progress updates."""
    writer = get_stream_writer()   # get the stream writer for this context

    total = 5
    results = []

    for i in range(1, total + 1):
        # Simulate fetching one batch of records
        record = f"Record #{i} from {source}"
        results.append(record)

        # 👇 Push custom data into the stream RIGHT NOW (not after tool finishes)
        writer({
            "event": "progress",
            "fetched": i,
            "total": total,
            "latest": record,
        })

    writer({"event": "done", "total_fetched": total})
    return f"Fetched {total} records from {source}: {results}"


# ---------------------------------------------------------------------------
# 2. Create an agent with this tool
# ---------------------------------------------------------------------------
agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[fetch_records],
)


# ---------------------------------------------------------------------------
# 3. Stream with stream_mode="custom"
# ---------------------------------------------------------------------------
print("=" * 60)
print("Streaming CUSTOM UPDATES from inside a tool")
print("=" * 60)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Fetch records from the sales database."}]},
    stream_mode="custom",
    version="v2",
):
    if chunk["type"] == "custom":
        data = chunk["data"]   # this is whatever you passed to writer(...)

        if isinstance(data, dict) and data.get("event") == "progress":
            pct = int(data["fetched"] / data["total"] * 100)
            bar = "" * data["fetched"] + "░" * (data["total"] - data["fetched"])
            print(f"  [{bar}] {pct}% — {data['latest']}")

        elif isinstance(data, dict) and data.get("event") == "done":
            print(f"\n Done! Total fetched: {data['total_fetched']}")

# ---------------------------------------------------------------------------
# EXPECTED OUTPUT:
#
#   [█░░░░] 20% — Record #1 from sales database
#   [██░░░] 40% — Record #2 from sales database
#   [███░░] 60% — Record #3 from sales database
#   [████░] 80% — Record #4 from sales database
#   [█████] 100% — Record #5 from sales database
#
#   Done! Total fetched: 5
# ---------------------------------------------------------------------------