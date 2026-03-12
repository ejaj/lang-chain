"""
===========================
TOPIC: Streaming Multiple Modes Simultaneously

WHAT IT DOES:
    You can combine any stream modes into a list.
    Each chunk tells you its own type so you can route accordingly.

HOW IT WORKS:
    - Pass stream_mode=["updates", "messages", "custom"]  (any subset)
    - Every chunk is a StreamPart: {"type": ..., "ns": ..., "data": ...}
    - Switch on chunk["type"] to handle each mode separately

WHEN TO USE:
    You want BOTH step-level progress AND token-level output
    You want to show a progress log AND live typing at the same time
    You want custom tool events AND final LLM text simultaneously

DESIGN PATTERN:
    Think of it as a single multiplexed stream — all modes flow through
    one for-loop and you demux with if/elif on chunk["type"].
"""

from langchain.agents import create_agent
from langchain.messages import AIMessageChunk
from langgraph.config import get_stream_writer


# ---------------------------------------------------------------------------
# 1. Tool that emits custom events
# ---------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    writer(f"🔍 Looking up weather for: {city}")
    writer(f"📡 Data acquired for: {city}")
    return f"It's always sunny in {city}!"


# ---------------------------------------------------------------------------
# 2. Create agent
# ---------------------------------------------------------------------------
agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)


# ---------------------------------------------------------------------------
# 3. Stream THREE modes at once
# ---------------------------------------------------------------------------
print("=" * 60)
print("Streaming MULTIPLE MODES simultaneously")
print("=" * 60)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode=["updates", "messages", "custom"],   # 👈 list of modes
    version="v2",
):
    mode = chunk["type"]   # always present in v2 format

    # ── Mode 1: Agent step updates ───────────────────────────────────────
    if mode == "updates":
        for step, state in chunk["data"].items():
            last_msg = state["messages"][-1]
            print(f"\n[STEP] {step} → {last_msg.__class__.__name__}")

    # ── Mode 2: Individual LLM tokens ────────────────────────────────────
    elif mode == "messages":
        token, meta = chunk["data"]
        if isinstance(token, AIMessageChunk):
            text_blocks = [b for b in token.content_blocks if b["type"] == "text"]
            for b in text_blocks:
                print(b["text"], end="", flush=True)

    # ── Mode 3: Custom tool events ────────────────────────────────────────
    elif mode == "custom":
        print(f"\n[CUSTOM] {chunk['data']}")

print()

# ---------------------------------------------------------------------------
# EXPECTED OUTPUT (interleaved):
#
#   [STEP]   model → AIMessage
#   [CUSTOM] Looking up weather for: San Francisco
#   [CUSTOM] Data acquired for: San Francisco
#   [STEP]   tools → ToolMessage
#   [STEP]   model → AIMessage
#   It's always sunny in San Francisco!
# ---------------------------------------------------------------------------