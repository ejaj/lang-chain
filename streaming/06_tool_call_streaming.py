"""
TOPIC: Streaming Tool Calls (partial JSON + completed parsed calls)

WHAT IT DOES:
    Streams BOTH:
      A) Partial JSON as tool-call arguments are generated token by token
      B) The fully-parsed, completed tool call once it's done

HOW IT WORKS:
    - Use stream_mode=["messages", "updates"]
    - From "messages" chunks → get incremental AIMessageChunk with tool_call_chunks
    - From "updates" chunks  → get completed AIMessage with parsed tool_calls list
    - This gives you live JSON preview PLUS the clean, validated result

WHEN TO USE:
    You want to show live "building the tool call..." feedback
    You need the parsed, validated tool call after it's complete
    You want to log exactly what arguments were sent to each tool
"""

from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage


# ---------------------------------------------------------------------------
# 1. Tool
# ---------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# ---------------------------------------------------------------------------
# 2. Agent
# ---------------------------------------------------------------------------
agent = create_agent("gpt-5-nano", tools=[get_weather])


# ---------------------------------------------------------------------------
# 3. Helper renderers
# ---------------------------------------------------------------------------
def render_chunk(token: AIMessageChunk) -> None:
    """Print partial text or partial tool-call JSON."""
    if token.text:
        # Partial final-answer text
        print(token.text, end="", flush=True)

    if token.tool_call_chunks:
        # Partial tool-call JSON (args are incomplete strings)
        for tc in token.tool_call_chunks:
            if tc["name"]:   # first chunk carries the tool name
                print(f"\n[TOOL BUILDING] name={tc['name']!r}", end="", flush=True)
            print(tc["args"], end="", flush=True)   # incremental JSON


def render_completed(message: AnyMessage) -> None:
    """Print fully parsed tool call or tool response."""
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"\n\n✅ [COMPLETE TOOL CALL]")
        for tc in message.tool_calls:
            print(f"   Tool : {tc['name']}")
            print(f"   Args : {tc['args']}")
            print(f"   ID   : {tc['id']}")

    if isinstance(message, ToolMessage):
        print(f"\n📤 [TOOL RESULT] {message.content_blocks}")


# ---------------------------------------------------------------------------
# 4. Stream
# ---------------------------------------------------------------------------
print("=" * 60)
print("Streaming TOOL CALLS (partial + complete)")
print("=" * 60)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in Boston?"}]},
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "messages":
        token, meta = chunk["data"]
        if isinstance(token, AIMessageChunk):
            render_chunk(token)

    elif chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source in ("model", "tools"):
                render_completed(update["messages"][-1])

print()

# ---------------------------------------------------------------------------
# EXPECTED OUTPUT:
#
#   [TOOL BUILDING] name='get_weather'{"city": "Boston"}
#
#   [COMPLETE TOOL CALL]
#      Tool : get_weather
#      Args : {'city': 'Boston'}
#      ID   : call_abc123
#
#   [TOOL RESULT] [{'type': 'text', 'text': "It's always sunny in Boston!"}]
#
#   The weather in Boston is sunny!
# ---------------------------------------------------------------------------