# TYPE: Before and After Model Middleware
# DESCRIPTION:
# @before_model → runs BEFORE the model gets the messages (modify input)
# @after_model  → runs AFTER the model replies (modify or validate output)
# Use before_model to filter/clean messages going IN to the model.
# Use after_model to filter/validate messages coming OUT of the model.

from langchain.messages import RemoveMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any

# ============================================================
# BEFORE MODEL: filter messages before the model sees them
# ============================================================
@before_model
def inject_context(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Add a reminder message before every model call."""
    # You can modify the messages the model receives
    # Return None = no changes
    # Return dict = update state before model call
    messages = state["messages"]
    if len(messages) == 0:
        return None
    # example: just log what's happening
    print(f"[before_model] About to call model with {len(messages)} messages")
    return None  # no changes in this example

# ============================================================
# AFTER MODEL: validate or clean the model's reply
# ============================================================
@after_model
def validate_response(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove AI replies that contain sensitive words."""
    BANNED_WORDS = ["password", "secret", "credit card"]
    last_message = state["messages"][-1]

    if any(word in last_message.content.lower() for word in BANNED_WORDS):
        print(f"[after_model] Blocked reply containing sensitive words!")
        return {"messages": [RemoveMessage(id=last_message.id)]}  # delete the reply

    print(f"[after_model] Reply passed validation ")
    return None  # reply is fine, keep it

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[],
    middleware=[inject_context, validate_response],   # both run every turn
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is 2 + 2?"}]},
    config,
)
print(result["messages"][-1].content)
# [before_model] About to call model with 1 messages
# [after_model]  Reply passed validation
# → "4"

# EXECUTION ORDER:
# User message → @before_model → model call → @after_model → final reply