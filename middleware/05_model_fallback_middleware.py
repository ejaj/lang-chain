"""
05_model_fallback_middleware.py
================================
TOPIC: ModelFallbackMiddleware

WHAT IT DOES:
    Automatically switches to a backup model when the primary model fails
    (API error, rate limit, timeout, outage). You can chain multiple fallbacks.

WHY THIS MATTERS:
    LLM APIs go down. Rate limits get hit. If your agent is mission-critical,
    you need a backup plan. ModelFallbackMiddleware handles this transparently
    without any changes to your business logic.

HOW IT WORKS:
    1. Agent calls primary model (e.g. gpt-4.1)
    2. If the call fails → tries first fallback (e.g. gpt-4.1-mini)
    3. If that fails too → tries next fallback (e.g. claude-3-5-sonnet)
    4. If all fail → raises the last exception

FALLBACK ORDER:
    Primary → Fallback 1 → Fallback 2 → ... → raise

CONFIGURATION:
    ModelFallbackMiddleware(
        "gpt-4.1-mini",                      # first fallback
        "claude-3-5-sonnet-20241022",         # second fallback
        "anthropic:claude-haiku-4-5",         # third fallback
    )

WHEN TO USE:
    High-availability production agents
    Cost optimization (fallback to cheaper model)
    Multi-provider redundancy (OpenAI + Anthropic)
    Rate limit handling
"""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware


# ---------------------------------------------------------------------------
# 1. Tool
# ---------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"It's sunny and 22°C in {city}."


# ---------------------------------------------------------------------------
# 2. Agent with fallback chain
#    Primary: gpt-4.1 (most capable)
#    Fallback 1: gpt-4.1-mini (cheaper, faster)
#    Fallback 2: claude-3-5-sonnet (different provider entirely)
# ---------------------------------------------------------------------------
agent = create_agent(
    model="gpt-4.1",                        # primary model
    tools=[get_weather],
    middleware=[
        ModelFallbackMiddleware(
            "gpt-4.1-mini",                  # fallback 1
            "claude-3-5-sonnet-20241022",    # fallback 2
        ),
    ],
)


# ---------------------------------------------------------------------------
# 3. Normal usage — no error, primary model responds
# ---------------------------------------------------------------------------
print("=" * 60)
print("ModelFallbackMiddleware — normal run (primary model used)")
print("=" * 60)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Paris?"}]
})
print(f"Response: {result['messages'][-1].content}")


# ---------------------------------------------------------------------------
# 4. Simulated failure scenario (shown as pseudocode — you can't force
#    a real API to fail in a demo, but this is how it works internally)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("What happens when primary fails:")
print("=" * 60)
print("""
  1. agent calls gpt-4.1  →  APIError: rate limit exceeded
  2. ModelFallbackMiddleware catches the error
  3. agent retries with gpt-4.1-mini  →  success 

  OR if gpt-4.1-mini also fails:
  3. agent retries with claude-3-5-sonnet  →  success

  OR if all fail:
  4. Last exception is raised to the caller
""")


# ---------------------------------------------------------------------------
# 5. Cost-optimization pattern: primary=best, fallback=cheapest
#    Use this when you want best quality normally but cheap during high load
# ---------------------------------------------------------------------------
cost_optimized_agent = create_agent(
    model="gpt-4.1",                     # high quality by default
    tools=[get_weather],
    middleware=[
        ModelFallbackMiddleware(
            "gpt-4.1-nano",              # ultra cheap fallback
        ),
    ],
)
print("Cost-optimized agent created: gpt-4.1 → gpt-4.1-nano on failure")