"""
TOPIC: Tool Calling Strategy — Basic Usage

WHAT IT DOES:
    For models that don't support native structured output, LangChain uses
    TOOL CALLING to achieve the same result. The model calls a special
    hidden tool whose arguments ARE your structured output.

HOW IT WORKS:
    - Import ToolStrategy from langchain.agents.structured_output
    - Wrap your schema: response_format=ToolStrategy(MySchema)
    - Internally: LangChain registers your schema as a tool
    - The model "calls" that tool with the structured args
    - LangChain captures, validates, and returns the parsed object

DIFFERENCE FROM PROVIDER STRATEGY:
    ProviderStrategy  → provider enforces schema at the API level
    ToolStrategy      → schema enforced via tool-call argument parsing
                        Works on ANY model that supports tool calling

EXTRA POWER OF TOOL STRATEGY (not in ProviderStrategy):
    Union types: model chooses between multiple schemas
    Custom tool_message_content (what appears in chat history)
    Flexible handle_errors configuration
"""

from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------
class ProductReview(BaseModel):
    """Analysis of a product review."""
    rating:     int | None = Field(description="Star rating 1–5", ge=1, le=5)
    sentiment:  Literal["positive", "negative"] = Field(description="Overall sentiment")
    key_points: list[str] = Field(description="Key points. Lowercase, 1–3 words each.")


# ---------------------------------------------------------------------------
# 2. Agent — explicitly wrap in ToolStrategy
#    (Also works if you just pass ProductReview directly and the model
#     doesn't support native structured output — LangChain falls back
#     to ToolStrategy automatically in that case)
# ---------------------------------------------------------------------------
tools = []   # your regular tools here; ToolStrategy adds its own hidden tool

agent = create_agent(
    model="gpt-5",
    tools=tools,
    response_format=ToolStrategy(ProductReview),   # explicit
)


# ---------------------------------------------------------------------------
# 3. Invoke
# ---------------------------------------------------------------------------
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Analyze: 'Great product, 5 stars. Fast shipping, but expensive.'"
    }]
})

output: ProductReview = result["structured_response"]

print("=" * 60)
print("STRUCTURED OUTPUT (ToolStrategy)")
print("=" * 60)
print(f"Rating     : {output.rating}")       # 5
print(f"Sentiment  : {output.sentiment}")    # positive
print(f"Key points : {output.key_points}")   # ['fast shipping', 'expensive']

# ---------------------------------------------------------------------------
# WHAT YOU SEE IN MESSAGE HISTORY (tool calling under the hood):
#
#   AI Message:
#     Tool Calls: ProductReview(call_1)
#       Args: {rating: 5, sentiment: "positive", key_points: [...]}
#
#   Tool Message:
#     Name: ProductReview
#     Returning structured response: {rating: 5, ...}
# ---------------------------------------------------------------------------