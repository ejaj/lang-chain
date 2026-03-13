"""
01_provider_strategy_pydantic.py
=================================
TOPIC: Provider Strategy — Pydantic Model

WHAT IT DOES:
    Uses the model provider's NATIVE structured output API (e.g. OpenAI JSON mode,
    Anthropic tool_use schema enforcement) to guarantee the response matches
    your Pydantic model exactly — no parsing, no guessing.

HOW IT WORKS:
    - Pass a Pydantic BaseModel class directly to response_format=
    - LangChain detects the provider supports native structured output
    - Automatically wraps it in ProviderStrategy internally
    - The validated Pydantic INSTANCE is returned in result["structured_response"]

WHY PYDANTIC:
    Field-level validation (ge=, le=, regex=, etc.)
    Auto-generated JSON schema from type hints
    You get back a real Python object, not a raw dict
    IDE autocomplete works on the result

WHEN TO USE:
    You want the most reliable structured output available
    Your provider supports native structured output (OpenAI, Anthropic, xAI, Gemini)
    You need field validation (e.g. rating must be 1–5)
"""

from pydantic import BaseModel, Field
from langchain.agents import create_agent


# ---------------------------------------------------------------------------
# 1. Define your output schema as a Pydantic model
#    - Use Field(description=...) so the model knows what each field means
#    - Use validators (ge=, le=, etc.) to constrain values
# ---------------------------------------------------------------------------
class ContactInfo(BaseModel):
    """Contact information extracted from text."""
    name:  str = Field(description="Full name of the person")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number including area code")


# ---------------------------------------------------------------------------
# 2. Create agent — pass schema TYPE directly (not an instance)
#    LangChain auto-selects ProviderStrategy if the model supports it,
#    otherwise falls back to ToolStrategy automatically.
# ---------------------------------------------------------------------------
agent = create_agent(
    model="gpt-5",
    response_format=ContactInfo,    # just the class, not ContactInfo()
)


# ---------------------------------------------------------------------------
# 3. Invoke and read structured_response
# ---------------------------------------------------------------------------
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Extract contact info: John Doe, john@example.com, (555) 123-4567"
    }]
})

output: ContactInfo = result["structured_response"]

print("=" * 60)
print("STRUCTURED OUTPUT (Pydantic — ProviderStrategy)")
print("=" * 60)
print(f"Type   : {type(output)}")       # <class '__main__.ContactInfo'>
print(f"Name   : {output.name}")        # John Doe
print(f"Email  : {output.email}")       # john@example.com
print(f"Phone  : {output.phone}")       # (555) 123-4567
print()
print("As dict:", output.model_dump())  # {'name': '...', 'email': '...', 'phone': '...'}

# ---------------------------------------------------------------------------
# EXPECTED OUTPUT:
#
#   Type   : <class '__main__.ContactInfo'>
#   Name   : John Doe
#   Email  : john@example.com
#   Phone  : (555) 123-4567
#   As dict: {'name': 'John Doe', 'email': 'john@example.com', 'phone': '(555) 123-4567'}
# ---------------------------------------------------------------------------