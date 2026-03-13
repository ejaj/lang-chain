"""
TOPIC: Tool Strategy — Custom tool_message_content

WHAT IT DOES:
    Controls what text appears in the chat/message history after the model
    calls the structured output "tool". By default it shows the raw dict.
    With tool_message_content you can show a friendlier message instead.

WHY THIS MATTERS:
    The agent's message history is what gets fed back to the model on the
    next turn. A cleaner message keeps the context smaller and more readable.

DEFAULT (no tool_message_content):
    ToolMessage:
      Returning structured response: {'task': '...', 'assignee': '...', ...}

WITH tool_message_content:
    ToolMessage:
      Action item captured and added to meeting notes!

WHEN TO USE:
    You want a human-readable confirmation in the chat history
    You're building a UI that shows message history to the user
    You want to keep model context cleaner / shorter
"""

from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------
class MeetingAction(BaseModel):
    """Action item extracted from a meeting transcript."""
    task: str = Field(description="The specific task to complete")
    assignee: str = Field(description="Person responsible")
    priority: Literal["low", "medium", "high"] = Field(description="Priority level")


# ---------------------------------------------------------------------------
# 2. Agent — with custom tool_message_content
# ---------------------------------------------------------------------------
agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=MeetingAction,
        tool_message_content="Action item captured and added to meeting notes!",
        # This replaces the default "Returning structured response: {...}"
    ),
)


# ---------------------------------------------------------------------------
# 3. Invoke and inspect the full message history
# ---------------------------------------------------------------------------
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "From our meeting: Sarah needs to update the project timeline ASAP."
    }]
})

print("=" * 60)
print("FULL MESSAGE HISTORY")
print("=" * 60)
for msg in result["messages"]:
    cls = type(msg).__name__
    print(f"\n[{cls}]")
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  Tool call: {tc['name']}({tc['args']})")
    elif hasattr(msg, "content"):
        print(f"  {msg.content}")

print("\n" + "=" * 60)
print("STRUCTURED RESULT")
print("=" * 60)
output: MeetingAction = result["structured_response"]
print(f"Task     : {output.task}")
print(f"Assignee : {output.assignee}")
print(f"Priority : {output.priority}")

# ---------------------------------------------------------------------------
# EXPECTED MESSAGE HISTORY:
#
#   [HumanMessage]
#     From our meeting: Sarah needs to update the project timeline ASAP.
#
#   [AIMessage]
#     Tool call: MeetingAction({'task': 'Update project timeline',
#                               'assignee': 'Sarah', 'priority': 'high'})
#
#   [ToolMessage]
#     Action item captured and added to meeting notes!
#       ← custom message instead of raw dict
#
# STRUCTURED RESULT:
#   Task     : Update the project timeline
#   Assignee : Sarah
#   Priority : high
# ---------------------------------------------------------------------------