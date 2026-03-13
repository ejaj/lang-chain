"""
TOPIC: Tool Strategy — Union Types (model picks the right schema)

WHAT IT DOES:
    When the input could be one of SEVERAL different shapes,
    use Union[SchemaA, SchemaB] and let the model decide which schema
    best fits the content.

HOW IT WORKS:
    - Wrap multiple schemas in typing.Union
    - Pass Union[...] to ToolStrategy(schema=...)
    - LangChain registers ALL schemas as separate tools
    - The model calls whichever tool matches the content
    - You get back whichever schema type the model chose

WHEN TO USE:
    Ambiguous inputs (could be contact info OR event details)
    Multi-intent classification pipelines
    Document parsing where document type varies

NOTE:
    Union types are ONLY available in ToolStrategy, not ProviderStrategy.
    If you try passing Union to ProviderStrategy you'll get an error.
"""

from pydantic import BaseModel, Field
from typing import Union
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


# ---------------------------------------------------------------------------
# 1. Two possible output schemas
# ---------------------------------------------------------------------------
class ContactInfo(BaseModel):
    """Contact information for a person."""
    name:  str = Field(description="Person's full name")
    email: str = Field(description="Email address")


class EventDetails(BaseModel):
    """Details about an event."""
    event_name: str = Field(description="Name of the event")
    date:       str = Field(description="Event date (e.g. March 15th)")
    location:   str = Field(description="Where the event takes place")


# ---------------------------------------------------------------------------
# 2. Agent with Union schema — model picks the right one
# ---------------------------------------------------------------------------
agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(Union[ContactInfo, EventDetails]),  # Union
)


# ---------------------------------------------------------------------------
# 3. Test with contact input → expect ContactInfo
# ---------------------------------------------------------------------------
result_contact = agent.invoke({
    "messages": [{"role": "user", "content":
        "Here's a contact: Jane Smith, jane@company.com"}]
})
print("Input: contact text")
print(f"Type   : {type(result_contact['structured_response']).__name__}")
print(f"Result : {result_contact['structured_response']}")
# Type   : ContactInfo
# Result : ContactInfo(name='Jane Smith', email='jane@company.com')
print()


# ---------------------------------------------------------------------------
# 4. Test with event input → expect EventDetails
# ---------------------------------------------------------------------------
result_event = agent.invoke({
    "messages": [{"role": "user", "content":
        "There's a Tech Summit on April 20th at the San Francisco Convention Center"}]
})
print("Input: event text")
print(f"Type   : {type(result_event['structured_response']).__name__}")
print(f"Result : {result_event['structured_response']}")
# Type   : EventDetails
# Result : EventDetails(event_name='Tech Summit', date='April 20th', location='...')
print()


# ---------------------------------------------------------------------------
# 5. Safely handle either type downstream
# ---------------------------------------------------------------------------
def process(result: dict) -> None:
    output = result["structured_response"]
    if isinstance(output, ContactInfo):
        print(f"Sending email to {output.name} at {output.email}")
    elif isinstance(output, EventDetails):
        print(f"Scheduling: {output.event_name} on {output.date} @ {output.location}")

process(result_contact)
process(result_event)