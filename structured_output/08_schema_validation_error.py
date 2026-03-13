"""
TOPIC: Auto-retry — Schema Validation Error

WHAT IT DOES:
    When the model's output FAILS Pydantic validation (e.g. rating=10 when
    max is 5), LangChain sends the exact validation error back to the model
    so it can fix its answer and retry.

HOW IT WORKS:
    1. Model calls structured output tool with args
    2. Pydantic validation raises an error (e.g. "rating must be <= 5")
    3. LangChain catches it → wraps it as StructuredOutputValidationError
    4. Sends a ToolMessage to the model with the exact error text
    5. Model reads the error, corrects its args, retries
    6. Success → result["structured_response"] has the valid object

RETRY MESSAGE FORMAT:
    ToolMessage:
      "Error: Failed to parse structured output for tool 'ProductRating':
       1 validation error for ProductRating.rating
         Input should be less than or equal to 5 ..."

WHEN THIS HAPPENS:
    - User says "10/10" → model naively puts rating=10
    - User says "half a star" → model puts rating=0.5 (not an int)
    - User omits required info → model leaves a required field None
"""

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


# ---------------------------------------------------------------------------
# 1. Schema with strict validation constraint
# ---------------------------------------------------------------------------
class ProductRating(BaseModel):
    """Product review parsed from user text."""
    rating:  int | None = Field(
        description="Star rating. Must be an integer between 1 and 5.",
        ge=1,   # greater-than-or-equal to 1
        le=5,   # less-than-or-equal to 5
    )
    comment: str = Field(description="The review comment text")


# ---------------------------------------------------------------------------
# 2. Agent — system prompt tells it not to invent values
#    (this makes the failure more likely so we can see the retry)
# ---------------------------------------------------------------------------
agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(ProductRating),  # handle_errors=True by default
    system_prompt=(
        "You are a review parser. Do NOT make up field values. "
        "Extract exactly what the user provided."
    ),
)


# ---------------------------------------------------------------------------
# 3. Invoke with invalid data — "10/10" will cause validation failure
# ---------------------------------------------------------------------------
result = agent.invoke({
    "messages": [{"role": "user", "content": "Parse: Amazing product, 10/10!"}]
})

print("=" * 60)
print("RESULT AFTER AUTO-RETRY")
print("=" * 60)
output: ProductRating = result["structured_response"]
print(f"Rating  : {output.rating}")    # 5  (model corrected from 10)
print(f"Comment : {output.comment}")   # Amazing product

print("\n" + "=" * 60)
print("FULL MESSAGE HISTORY (shows the retry)")
print("=" * 60)
for i, msg in enumerate(result["messages"]):
    cls = type(msg).__name__
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"[{i}] {cls} → tool call: {tc['name']}({tc['args']})")
    else:
        content_preview = str(getattr(msg, "content", ""))[:150].replace("\n", " ")
        print(f"[{i}] {cls}: {content_preview}")

# ---------------------------------------------------------------------------
# EXPECTED OUTPUT:
#
# RESULT AFTER AUTO-RETRY
# ──────────────────────────────────────────────────────────────
# Rating  : 5
# Comment : Amazing product
#
# FULL MESSAGE HISTORY
# ──────────────────────────────────────────────────────────────
# [0] HumanMessage: Parse: Amazing product, 10/10!
# [1] AIMessage → tool call: ProductRating({'rating': 10, 'comment': 'Amazing product'})
# [2] ToolMessage: Error: Failed to parse structured output for tool 'ProductRating':
#                  1 validation error for ProductRating.rating
#                  Input should be less than or equal to 5 ...
# [3] AIMessage → tool call: ProductRating({'rating': 5, 'comment': 'Amazing product'})
# [4] ToolMessage: Returning structured response: {'rating': 5, 'comment': '...'}
# ---------------------------------------------------------------------------