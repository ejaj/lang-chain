"""
08_sub_agent_streaming.py
=========================
TOPIC: Streaming from Sub-Agents (Multi-Agent Systems)

WHAT IT DOES:
    When one agent calls another agent as a tool, you have multiple LLMs
    producing tokens simultaneously (or sequentially). This example shows
    how to track WHICH agent is currently speaking.

HOW IT WORKS:
    - Give each agent a `name` when creating it
    - Pass `subgraphs=True` to agent.stream()
    - In "messages" chunks, check metadata["lc_agent_name"]
    - That key tells you which named agent emitted the token

ARCHITECTURE:
    supervisor  ──calls──▶  call_weather_agent (tool)
                                  │
                                  └──invokes──▶  weather_agent (sub-agent)
                                                      │
                                                      └──calls──▶  get_weather (tool)

WHEN TO USE:
    Multi-agent pipelines (orchestrator + specialist agents)
    You need to label which agent produced which output
    Debugging complex agent-to-agent flows
"""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage


# ---------------------------------------------------------------------------
# 1. Inner tool used by the sub-agent
# ---------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# ---------------------------------------------------------------------------
# 2. Create the WEATHER SUB-AGENT (specialist)
# ---------------------------------------------------------------------------
weather_model = init_chat_model("gpt-5-nano")

weather_agent = create_agent(
    model=weather_model,
    tools=[get_weather],
    name="weather_agent",        # named so we can identify it in the stream
)


# ---------------------------------------------------------------------------
# 3. Wrap the sub-agent as a tool for the supervisor
# ---------------------------------------------------------------------------
def call_weather_agent(query: str) -> str:
    """Query the weather sub-agent with a natural-language question."""
    result = weather_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].text   # return final text response


# ---------------------------------------------------------------------------
# 4. Create the SUPERVISOR AGENT (orchestrator)
# ---------------------------------------------------------------------------
supervisor_model = init_chat_model("gpt-5-nano")

agent = create_agent(
    model=supervisor_model,
    tools=[call_weather_agent],
    name="supervisor",           # named too
)


# ---------------------------------------------------------------------------
# 5. Helper renderers
# ---------------------------------------------------------------------------
def render_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="", flush=True)
    if token.tool_call_chunks:
        for tc in token.tool_call_chunks:
            if tc["name"]:
                print(f"\n  [tool→] {tc['name']}(", end="", flush=True)
            print(tc["args"], end="", flush=True)


def render_completed(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"\n  Parsed call: {[tc['name'] for tc in message.tool_calls]}")
    if isinstance(message, ToolMessage):
        print(f"\n  Result: {message.content}")


# ---------------------------------------------------------------------------
# 6. Stream with subgraphs=True so sub-agent chunks are included
# ---------------------------------------------------------------------------
print("=" * 60)
print("Streaming from MULTIPLE AGENTS (supervisor + sub-agent)")
print("=" * 60)

current_agent: str | None = None

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in Boston?"}]},
    stream_mode=["messages", "updates"],
    subgraphs=True,              #  include sub-agent stream events
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]

        # Detect agent switch
        agent_name = metadata.get("lc_agent_name")
        if agent_name and agent_name != current_agent:
            print(f"\n\n [{agent_name}]")
            print("-" * 40)
            current_agent = agent_name

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
#   [supervisor]
#   ----------------------------------------
#   [tool→] call_weather_agent({"query": "Boston weather right now"}
#     Parsed call: ['call_weather_agent']
#
#   [weather_agent]
#   ----------------------------------------
#   [tool→] get_weather({"city": "Boston"}
#     Parsed call: ['get_weather']
#     Result: It's always sunny in Boston!
#   It's always sunny in Boston!
#     Result: It's always sunny in Boston!
#
#   [supervisor]
#   ----------------------------------------
#   The weather in Boston is sunny!
# ---------------------------------------------------------------------------