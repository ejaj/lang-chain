"""
01_agent_progress_streaming.py
==============================
TOPIC: Streaming Agent Progress (stream_mode="updates")

WHAT IT DOES:
    Emits a snapshot of the agent's state after EVERY step.
    Useful when you want to show users "what the agent did" step-by-step,
    rather than individual tokens.

HOW IT WORKS:
    - Each chunk has type="updates" and data={node_name: state_update}
    - You see one event per node that ran (e.g. "model", "tools", "model")
    - Good for progress bars, step logs, audit trails

WHEN TO USE:
    You want to show high-level agent steps
    You want to log which tools were called and with what args
    You don't need token-by-token output
"""

from langchain.agents import create_agent


# ---------------------------------------------------------------------------
# 1. Define a simple tool
# ---------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# ---------------------------------------------------------------------------
# 2. Create an agent with a model and that tool
# ---------------------------------------------------------------------------
agent = create_agent(
    model="gpt-5-nano",         # any supported model string
    tools=[get_weather],
)


# ---------------------------------------------------------------------------
# 3. Stream with stream_mode="updates"
# ---------------------------------------------------------------------------
print("=" * 60)
print("Streaming agent PROGRESS (one event per step)")
print("=" * 60)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="updates",
    version="v2",               # unified StreamPart format (recommended)
):
    # Every chunk is a StreamPart dict: {"type": ..., "ns": ..., "data": ...}
    if chunk["type"] == "updates":
        for step_name, step_data in chunk["data"].items():
            print(f"\n Step : {step_name}")
            # step_data["messages"] contains the latest messages list
            last_msg = step_data["messages"][-1]
            print(f"   Role : {last_msg.__class__.__name__}")
            print(f"   Content blocks: {last_msg.content_blocks}")

# ---------------------------------------------------------------------------
# EXPECTED OUTPUT (3 events):
#
#   Step : model
#      Role : AIMessage
#      Content blocks: [{'type': 'tool_call', 'name': 'get_weather', ...}]
#
#   Step : tools
#      Role : ToolMessage
#      Content blocks: [{'type': 'text', 'text': "It's always sunny..."}]
#
#   Step : model
#      Role : AIMessage
#      Content blocks: [{'type': 'text', 'text': 'It's always sunny...'}]
# ---------------------------------------------------------------------------