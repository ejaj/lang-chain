"""
TOPIC: Auto-retry — Multiple Structured Outputs Error

WHAT IT DOES:
    When you use Union types, a confused model might call MULTIPLE
    structured-output tools at once. LangChain detects this, sends an
    error back to the model, and the model retries with just ONE call.

HOW IT WORKS:
    1. You pass Union[ContactInfo, EventDetails] to ToolStrategy
    2. LangChain registers both as tools
    3. Model (incorrectly) calls BOTH in one turn
    4. LangChain detects multiple calls → sends MultipleStructuredOutputsError
       as a ToolMessage to the model
    5. Model reads the error and retries — this time calling only ONE

WHAT THE RETRY LOOP LOOKS LIKE IN MESSAGES:
    AI  → calls ContactInfo AND EventDetails  (wrong)
    Tool→ "Error: multiple structured responses returned..."
    Tool→ "Error: multiple structured responses returned..."
    AI  → calls only ContactInfo              (correct)
    Tool→ "Returning structured response: {...}"

WHEN YOU SEE THIS:
    Ambiguous prompts ("extract info" when input contains both a person and event)
    Models that are aggressive about completing all possible tasks
"""

from pydantic import BaseModel, Field
from typing import Union
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


# ---------------------------------------------------------------------------
# 1. Two schemas
# ---------------------------------------------------------------------------
class ContactInfo(BaseModel):
    name:  str = Field(description="Person's name")
    email: str = Field(description="Email address")

class EventDetails(BaseModel):
    event_name: str = Field(description="Name of the event")
    date:       str = Field(description="Event date")


# ---------------------------------------------------------------------------
# 2. Agent — default handle_errors=True catches MultipleStructuredOutputsError
# ---------------------------------------------------------------------------
agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(Union[ContactInfo, EventDetails]),
)


# ---------------------------------------------------------------------------
# 3. Invoke with ambiguous input (both a person AND an event)
#    Some models will try to call both tools — LangChain auto-corrects this
# ---------------------------------------------------------------------------
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": (
            "Extract info: John Doe (john@email.com) is organizing "
            "Tech Conference on March 15th"
        ),
    }]
})

print("=" * 60)
print("RESULT AFTER AUTO-RETRY")
print("=" * 60)
output = result["structured_response"]
print(f"Type   : {type(output).__name__}")
print(f"Result : {output}")

print("\n" + "=" * 60)
print("FULL MESSAGE HISTORY (shows retry loop)")
print("=" * 60)
for msg in result["messages"]:
    cls = type(msg).__name__
    content = getattr(msg, "content", "")
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        calls = [tc["name"] for tc in msg.tool_calls]
        print(f"[{cls}] tool_calls={calls}")
    else:
        # Truncate long content
        preview = str(content)[:120].replace("\n", " ")
        print(f"[{cls}] {preview}")

# ---------------------------------------------------------------------------
# EXPECTED CONSOLE OUTPUT:
#
# RESULT AFTER AUTO-RETRY
# ──────────────────────────────────────────────────────────────
# Type   : ContactInfo
# Result : ContactInfo(name='John Doe', email='john@email.com')
#
# FULL MESSAGE HISTORY
# ──────────────────────────────────────────────────────────────
# [HumanMessage]  Extract info: John Doe ...
# [AIMessage]     tool_calls=['ContactInfo', 'EventDetails']   ← wrong
# [ToolMessage]   Error: Model incorrectly returned multiple structured responses...
# [ToolMessage]   Error: Model incorrectly returned multiple structured responses...
# [AIMessage]     tool_calls=['ContactInfo']                   ← corrected
# [ToolMessage]   Returning structured response: {'name': 'John Doe', ...}
# ---------------------------------------------------------------------------