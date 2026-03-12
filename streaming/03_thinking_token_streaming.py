"""
==============================
TOPIC: Streaming Thinking / Reasoning Tokens

WHAT IT DOES:
    Some models (e.g. Claude with extended thinking, OpenAI o-series) perform
    internal "chain-of-thought" reasoning BEFORE writing their final answer.
    This example streams those hidden thoughts in real time.

HOW IT WORKS:
    - Enable thinking/reasoning on the model (provider-specific config)
    - Use stream_mode="messages"
    - Filter content_blocks for type="reasoning"  → thinking text
    - Filter content_blocks for type="text"        → final answer text
    - LangChain normalizes Anthropic "thinking blocks" and OpenAI "reasoning
      summaries" into the same standard "reasoning" block type.

WHEN TO USE:
    You want to show users the model's reasoning process
    You're debugging why the model made a certain decision
    You want to build a "show thinking" toggle in your UI
"""

from langchain.agents import create_agent
from langchain.messages import AIMessageChunk
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import Runnable


# ---------------------------------------------------------------------------
# 1. Define a tool
# ---------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# ---------------------------------------------------------------------------
# 2. Create a model WITH thinking/reasoning enabled
#    (Anthropic example — change to your provider as needed)
# ---------------------------------------------------------------------------
model = ChatAnthropic(
    model_name="claude-sonnet-4-6",
    timeout=None,
    stop=None,
    thinking={
        "type": "enabled",
        "budget_tokens": 5000,   # max tokens the model may use for reasoning
    },
)

agent: Runnable = create_agent(
    model=model,
    tools=[get_weather],
)


# ---------------------------------------------------------------------------
# 3. Stream and separate thinking vs. final answer
# ---------------------------------------------------------------------------
print("=" * 60)
print("Streaming THINKING + FINAL ANSWER tokens")
print("=" * 60)

thinking_started = False
answer_started = False

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] != "messages":
        continue

    token, metadata = chunk["data"]
    if not isinstance(token, AIMessageChunk):
        continue

    blocks = token.content_blocks

    # ── Reasoning / thinking tokens ─────────────────────────────────────
    reasoning_blocks = [b for b in blocks if b["type"] == "reasoning"]
    for b in reasoning_blocks:
        if not thinking_started:
            print("\n[THINKING]\n" + "-" * 40)
            thinking_started = True
        print(b["reasoning"], end="", flush=True)

    # ── Final answer tokens ──────────────────────────────────────────────
    text_blocks = [b for b in blocks if b["type"] == "text"]
    for b in text_blocks:
        if not answer_started:
            print("\n\n[ANSWER]\n" + "-" * 40)
            answer_started = True
            thinking_started = False  # reset for next turn
        print(b["text"], end="", flush=True)

print()

# ---------------------------------------------------------------------------
# EXPECTED OUTPUT:
#
#   [THINKING]
#   ----------------------------------------
#   The user is asking about the weather in San Francisco.
#   I have a tool available to get this information...
#
#   [ANSWER]
#   ----------------------------------------
#   The weather in San Francisco is: It's always sunny in San Francisco!
# ---------------------------------------------------------------------------