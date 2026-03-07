# TYPE: Structured Output
# Force the agent to return data in a specific shape (like a form),
# instead of a free-text reply.

from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain.tools import tool

# --- Define the shape you want back ---
class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

@tool
def search(query: str) -> str:
    """Search for contact information."""
    return f"Found: John Doe, john@example.com, (555) 123-4567"

# --- Option A: ToolStrategy (works with any model) ---
agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[search],
    response_format=ToolStrategy(ContactInfo),
)

# --- Option B: ProviderStrategy (uses model's native structured output) ---
agent = create_agent(
    model="openai:gpt-4.1",
    tools=[search],
    response_format=ProviderStrategy(ContactInfo),
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Find contact info for John Doe"}]
})

# result["structured_response"] is a real Python object, not just text
contact = result["structured_response"]
print(contact.name)   # → "John Doe"
print(contact.email)  # → "john@example.com"
print(contact.phone)  # → "(555) 123-4567"