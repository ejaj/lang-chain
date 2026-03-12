# TYPE: Trim Messages (Fit Context Window)
# DESCRIPTION: Long conversations eventually exceed the model's token limit.
# Trimming keeps only the most recent messages and drops the old ones.
# Use @before_model middleware — runs before every model call.
# Fast and simple BUT loses old information permanently.

from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window.
    Runs automatically BEFORE every model call.
    """
    messages = state["messages"]

    # If conversation is short enough → do nothing
    if len(messages) <= 3:
        return None

    first_msg = messages[0]   # keep very first message (usually system prompt)

    # Keep last 3 or 4 recent messages
    recent = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),  # delete ALL current messages
            *new_messages                            # replace with trimmed version
        ]
    }

agent = create_agent(
    "openai:gpt-4.1",
    tools=[],
    middleware=[trim_messages],          # ← runs before every model call
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "Hi, my name is Bob."}, config)
agent.invoke({"messages": "Write a poem about cats."}, config)
agent.invoke({"messages": "Now write one about dogs."}, config)
result = agent.invoke({"messages": "What is my name?"}, config)

result["messages"][-1].pretty_print()
# → "Your name is Bob. You told me that earlier."

# PROS: Simple and fast
# CONS: Old messages are gone forever — if name was in a deleted message, it's lost
# USE WHEN: context overflows and you don't need old messages