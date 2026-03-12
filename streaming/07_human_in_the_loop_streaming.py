"""
TOPIC: Streaming with Human-in-the-Loop (Interrupts)

WHAT IT DOES:
    Pauses the agent before executing certain tools and waits for human
    approval/rejection/editing. The stream still flows — you get real-time
    token output RIGHT UP UNTIL the interrupt, then the agent pauses.

HOW IT WORKS:
    1. Create agent with HumanInTheLoopMiddleware and a checkpointer
    2. Stream normally — collect Interrupt objects from "__interrupt__" updates
    3. Build a decision dict (approve / reject / edit) for each interrupt
    4. Resume by calling agent.stream(Command(resume=decisions), ...)

KEY CONCEPTS:
    - Interrupt      : signals the agent needs human input before continuing
    - checkpointer   : saves agent state so it can be resumed later
    - Command(resume): sends your decisions back into the agent
    - thread_id      : ties the paused + resumed runs together

WHEN TO USE:
    Sensitive tool calls (deleting data, sending emails, spending money)
    Compliance workflows requiring a human sign-off
    Any agentic flow where you want human oversight mid-run
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt


# ---------------------------------------------------------------------------
# 1. Tool
# ---------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# ---------------------------------------------------------------------------
# 2. Checkpointer (stores state between the pause and resume)
# ---------------------------------------------------------------------------
checkpointer = InMemorySaver()   # use Redis/Postgres in production


# ---------------------------------------------------------------------------
# 3. Agent with human-in-the-loop middleware
# ---------------------------------------------------------------------------
agent = create_agent(
    "gpt-5-nano",
    tools=[get_weather],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"get_weather": True}   # pause BEFORE calling get_weather
        ),
    ],
    checkpointer=checkpointer,
)


# ---------------------------------------------------------------------------
# 4. Helper renderers
# ---------------------------------------------------------------------------
def render_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="", flush=True)


def render_completed(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"\n Tool calls planned: {[tc['name'] for tc in message.tool_calls]}")
    if isinstance(message, ToolMessage):
        print(f" Tool result: {message.content_blocks}")


def render_interrupt(interrupt: Interrupt) -> None:
    """Show the human what they need to approve."""
    print("\n INTERRUPT — Human approval required:")
    for req in interrupt.value["action_requests"]:
        print(f"   → {req['description']}")


# ---------------------------------------------------------------------------
# 5. First run — will PAUSE at the tool call
# ---------------------------------------------------------------------------
config = {"configurable": {"thread_id": "weather_thread_01"}}
collected_interrupts: list[Interrupt] = []

print("=" * 60)
print("RUN 1 — Agent streams until interrupt")
print("=" * 60)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Weather in Boston AND San Francisco?"}]},
    config=config,
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "messages":
        token, _ = chunk["data"]
        if isinstance(token, AIMessageChunk):
            render_chunk(token)

    elif chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source in ("model", "tools"):
                render_completed(update["messages"][-1])
            if source == "__interrupt__":
                # update is a list of Interrupt objects
                for interrupt in update:
                    collected_interrupts.append(interrupt)
                    render_interrupt(interrupt)


# ---------------------------------------------------------------------------
# 6. Build decisions  (edit Boston → Boston UK, approve San Francisco)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Human reviewing tool calls...")
print("=" * 60)

decisions = {}
for interrupt in collected_interrupts:
    action_decisions = []
    for req in interrupt.value["action_requests"]:
        desc = req["description"].lower()
        if "boston" in desc:
            # Edit: change the city to "Boston, UK"
            decision = {
                "type": "edit",
                "edited_action": {
                    "name": "get_weather",
                    "args": {"city": "Boston, UK"},
                },
            }
            print(f"  ✏️  EDITED: Boston → Boston, UK")
        else:
            # Approve as-is
            decision = {"type": "approve"}
            print(f"  APPROVED: San Francisco")
        action_decisions.append(decision)

    decisions[interrupt.id] = {"decisions": action_decisions}


# ---------------------------------------------------------------------------
# 7. Resume the agent with the human's decisions
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RUN 2 — Resuming with human decisions")
print("=" * 60)

for chunk in agent.stream(
    Command(resume=decisions),   #  send decisions back
    config=config,               #  same thread_id to resume from checkpoint
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "messages":
        token, _ = chunk["data"]
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
# RUN 1:
#   Tool calls planned: ['get_weather', 'get_weather']
#   INTERRUPT — Human approval required:
#      → Tool: get_weather | Args: {'city': 'Boston'}
#      → Tool: get_weather | Args: {'city': 'San Francisco'}
#
# Human reviewing:
#   ✏️  EDITED: Boston → Boston, UK
#   APPROVED: San Francisco
#
# RUN 2:
#   Tool result: "It's always sunny in Boston, UK!"
#   Tool result: "It's always sunny in San Francisco!"
#   Boston (UK): sunny. San Francisco: sunny.
# ---------------------------------------------------------------------------