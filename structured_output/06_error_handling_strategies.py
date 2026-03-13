"""
TOPIC: Tool Strategy — Error Handling (all 5 strategies)

WHAT IT DOES:
    When the model returns structured output that FAILS validation (wrong type,
    out-of-range value, missing field), LangChain can automatically retry.
    This file shows every handle_errors option.

THE RETRY LOOP:
    1. Model calls the structured output tool with args
    2. LangChain tries to validate the args against your schema
    3. If validation fails → sends error ToolMessage back to model
    4. Model sees the error and retries with corrected args
    5. Repeat until success OR max retries exceeded

FIVE handle_errors OPTIONS:
    True                  → catch ALL errors, use default message  (default)
    "custom string"       → catch ALL errors, always show this message
    ExceptionType         → only catch that specific exception type
    (ExcA, ExcB)          → only catch those exception types
    callable(err) -> str  → full custom logic, return the message to show
    False                 → never catch, let exceptions propagate
"""

from pydantic import BaseModel, Field
from typing import Union
from langchain.agents import create_agent
from langchain.agents.structured_output import (
    ToolStrategy,
    StructuredOutputValidationError,
    MultipleStructuredOutputsError,
)


# ──────────────────────────────────────────────────────────────────────────
# Shared schema (rating must be 1–5)
# ──────────────────────────────────────────────────────────────────────────
class ProductRating(BaseModel):
    rating:  int | None = Field(description="Rating 1–5", ge=1, le=5)
    comment: str        = Field(description="Review comment")


# ──────────────────────────────────────────────────────────────────────────
# STRATEGY 1: True (default) — catch everything, default error message
# The default error message describes the validation failure in detail.
# ──────────────────────────────────────────────────────────────────────────
agent_default = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=ProductRating,
        handle_errors=True,   # default — you don't need to write this
    ),
    system_prompt="Parse product reviews. Do not make up field values."
)

result = agent_default.invoke({
    "messages": [{"role": "user", "content": "Parse: Amazing product, 10/10!"}]
})
print("Strategy 1 (True):", result["structured_response"])
# Model initially says rating=10, gets error, retries with rating=5
print()


# ──────────────────────────────────────────────────────────────────────────
# STRATEGY 2: Custom string — always show this specific message on error
# Useful for clear domain-specific instructions to the model.
# ──────────────────────────────────────────────────────────────────────────
agent_str = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=ProductRating,
        handle_errors="Rating must be between 1 and 5. Use 5 if the user says 10/10.",
        # Model always sees this exact message when it makes a mistake
    ),
)

result = agent_str.invoke({
    "messages": [{"role": "user", "content": "Parse: Terrible product, 0/10."}]
})
print("Strategy 2 (str):", result["structured_response"])
print()


# ──────────────────────────────────────────────────────────────────────────
# STRATEGY 3: Exception type — only retry on ValueError, raise everything else
# ──────────────────────────────────────────────────────────────────────────
agent_exc = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=ProductRating,
        handle_errors=ValueError,   # only catches ValueError
        # If a TypeError or unexpected error occurs → it propagates up
    ),
)

result = agent_exc.invoke({
    "messages": [{"role": "user", "content": "Parse: Great! 4 stars."}]
})
print("Strategy 3 (ValueError only):", result["structured_response"])
print()


# ──────────────────────────────────────────────────────────────────────────
# STRATEGY 4: Tuple of exceptions — retry on any of these, raise others
# ──────────────────────────────────────────────────────────────────────────
agent_multi_exc = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=ProductRating,
        handle_errors=(ValueError, TypeError),  # retry on either
    ),
)

result = agent_multi_exc.invoke({
    "messages": [{"role": "user", "content": "Parse: Decent product, 3/5."}]
})
print("Strategy 4 (ValueError | TypeError):", result["structured_response"])
print()


# ──────────────────────────────────────────────────────────────────────────
# STRATEGY 5: Callable — full custom logic to produce the error message
# Best for complex workflows where different errors need different guidance.
# ──────────────────────────────────────────────────────────────────────────

class ContactInfo(BaseModel):
    name:  str = Field(description="Person's name")
    email: str = Field(description="Email address")

class EventDetails(BaseModel):
    event_name: str = Field(description="Event name")
    date:       str = Field(description="Event date")


def smart_error_handler(error: Exception) -> str:
    """Return a tailored error message based on what went wrong."""
    if isinstance(error, StructuredOutputValidationError):
        return (
            "The output format was wrong. Check field types and constraints, "
            "then try again with corrected values."
        )
    elif isinstance(error, MultipleStructuredOutputsError):
        return (
            "You returned multiple structured outputs but only one is expected. "
            "Pick the MOST relevant schema and return only that one."
        )
    else:
        # Fallback for unexpected errors
        return f"Unexpected error: {error}. Please retry."


agent_callable = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=Union[ContactInfo, EventDetails],
        handle_errors=smart_error_handler,   # callable
    ),
)

result = agent_callable.invoke({
    "messages": [{"role": "user", "content":
        "Info: Alice Brown, alice@work.com, organizing DevConf on June 5th"}]
})
print("Strategy 5 (callable):", result["structured_response"])
print()


# ──────────────────────────────────────────────────────────────────────────
# STRATEGY 6: False — never catch, let ALL errors propagate
# Use in testing/debugging when you want raw exceptions.
# ──────────────────────────────────────────────────────────────────────────
agent_no_retry = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=ProductRating,
        handle_errors=False,   # no retry, raises immediately
    ),
)

try:
    result = agent_no_retry.invoke({
        "messages": [{"role": "user", "content": "Parse: Amazing, 10/10!"}]
    })
    print("Strategy 6 (False):", result["structured_response"])
except Exception as e:
    print(f"Strategy 6 (False) — error raised: {type(e).__name__}: {e}")