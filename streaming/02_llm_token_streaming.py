"""
TOPIC: Streaming LLM Tokens (stream_mode="messages")

WHAT IT DOES:
    Streams individual tokens as the LLM produces them, token by token.
    This is the "live typing" effect you see in ChatGPT-style UIs.

HOW IT WORKS:
    - Each chunk has type="messages" and data=(token, metadata)
    - `token`    → an AIMessageChunk with partial content_blocks
    - `metadata` → dict with keys like `langgraph_node` (which node emitted it)
    - Tool call args also arrive incrementally as tool_call_chunks

WHEN TO USE:
    You want a ChatGPT-style streaming UI
    You want to show partial text as it arrives
    You want to see partial tool-call JSON as it's built up
"""

from langchain.agents import create_agent
from langchain.messages import AIMessageChunk


# ---------------------------------------------------------------------------
# 1. Define a tool
# ---------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# ---------------------------------------------------------------------------
# 2. Create an agent
# ---------------------------------------------------------------------------
agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)


# ---------------------------------------------------------------------------
# 3. Stream with stream_mode="messages"
# ---------------------------------------------------------------------------
print("=" * 60)
print("Streaming LLM TOKENS (token by token)")
print("=" * 60)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]

        # Only process actual LLM output chunks (not ToolMessages, etc.)
        if not isinstance(token, AIMessageChunk):
            continue

        node = metadata["langgraph_node"]   # e.g. "model" or "tools"
        blocks = token.content_blocks

        # ── Text tokens (final answer) ──────────────────────────────────
        text_blocks = [b for b in blocks if b["type"] == "text"]
        for b in text_blocks:
            print(b["text"], end="", flush=True)

        # ── Tool call chunks (partial JSON) ─────────────────────────────
        tool_chunks = [b for b in blocks if b["type"] == "tool_call_chunk"]
        for b in tool_chunks:
            # b["args"] is a partial JSON string, e.g. '{"', 'city', '":"SF"}'
            print(f"[TOOL_ARG] name={b['name']!r} args={b['args']!r}", flush=True)

print()  # newline at end

# ---------------------------------------------------------------------------
# EXPECTED OUTPUT (simplified):
#
#   [TOOL_ARG] name='get_weather' args=''
#   [TOOL_ARG] name=None args='{"'
#   [TOOL_ARG] name=None args='city'
#   [TOOL_ARG] name=None args='":"'
#   [TOOL_ARG] name=None args='San Francisco'
#   [TOOL_ARG] name=None args='"}'
#   Here's what I found: It's always sunny in San Francisco!
# ---------------------------------------------------------------------------