# TYPE: Delete Messages
# DESCRIPTION: RemoveMessage permanently deletes specific messages from state.
# Use @after_model to delete old messages AFTER the model replies.
# Delete specific messages by ID, or delete ALL with REMOVE_ALL_MESSAGES.
# Unlike trimming, you pick exactly WHICH messages to remove.

from langchain.messages import RemoveMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig

# --- Delete the 2 oldest messages after every reply ---
@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove the 2 oldest messages after every model reply.
    Runs automatically AFTER every model call.
    """
    messages = state["messages"]
    if len(messages) > 4:
        # delete the 2 oldest messages by their ID
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None  # nothing to delete yet

# --- Delete ALL messages at once (nuclear option) ---
def clear_all_messages(state):
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}

agent = create_agent(
    "openai:gpt-4.1",
    tools=[],
    middleware=[delete_old_messages],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

# Watch what's in state as conversation grows and old messages get deleted
for event in agent.stream(
    {"messages": [{"role": "user", "content": "Hi! I'm Bob."}]},
    config,
    stream_mode="values",
):
    print([(m.type, m.content[:30]) for m in event["messages"]])

for event in agent.stream(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    config,
    stream_mode="values",
):
    print([(m.type, m.content[:30]) for m in event["messages"]])

# IMPORTANT RULES when deleting:
# History should start with a human message (most providers require this)
# If AI called a tool, the tool result message must stay with it
# Don't delete an AIMessage that has tool_calls without its ToolMessage