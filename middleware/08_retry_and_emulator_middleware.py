"""
08_retry_and_emulator_middleware.py
=====================================
TOPIC: ToolRetryMiddleware + ModelRetryMiddleware + LLMToolEmulator

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART A — ToolRetryMiddleware
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Automatically retries a TOOL call when it raises an exception.
    Uses exponential backoff: waits 1s, 2s, 4s, ... between retries.

    CONFIGURATION:
        max_retries    → how many times to retry (default: 3)
        backoff_factor → multiplier for delay (default: 2.0)
        initial_delay  → first wait in seconds (default: 1.0)

    WHEN TO USE:
        API calls to flaky/rate-limited external services
        Database connections that occasionally time out
        Any I/O that has transient failures

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART B — ModelRetryMiddleware
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Same as ToolRetry but for the MODEL call itself.
    Useful for rate-limit errors from the LLM provider.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART C — LLMToolEmulator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Replaces REAL tool execution with AI-generated fake responses.
    The emulator LLM reads the tool's docstring and args, then invents
    a plausible response — no actual tool code runs.

    WHEN TO USE:
        Testing agent behavior before tools are implemented
        Demos without live API keys
        Unit tests where tool side effects must be avoided
"""

import random
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ToolRetryMiddleware,
    ModelRetryMiddleware,
    LLMToolEmulator,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
_call_count = 0

def flaky_api(endpoint: str) -> str:
    """Call a flaky external API that sometimes fails."""
    global _call_count
    _call_count += 1
    # Fail on first 2 attempts, succeed on 3rd
    if _call_count < 3:
        raise ConnectionError(f"API timeout on attempt {_call_count}")
    return f"API response from {endpoint}: {{status: 'ok', data: [...]}}"


def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol."""
    # Not implemented yet — will be emulated by LLM
    raise NotImplementedError("Stock API not connected yet")


def send_notification(user_id: str, message: str) -> str:
    """Send a push notification to a user."""
    # Not implemented yet — will be emulated by LLM
    raise NotImplementedError("Notification service not connected yet")


# ──────────────────────────────────────────────────────────────────────────
# PART A: ToolRetryMiddleware
# ──────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PART A — ToolRetryMiddleware (retries flaky_api up to 3x)")
print("=" * 60)

agent_retry = create_agent(
    model="gpt-4.1",
    tools=[flaky_api],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,        # try up to 3 times after first failure
            backoff_factor=2.0,   # delay: 1s → 2s → 4s
            initial_delay=0.1,    # short delay for this demo (use 1.0 in prod)
        ),
    ],
)

_call_count = 0   # reset counter
result = agent_retry.invoke({
    "messages": [{"role": "user", "content": "Call the /users endpoint of the API."}]
})
print(f"Tool was called {_call_count} time(s) (failed {_call_count - 1}, succeeded 1)")
print(f"Response: {result['messages'][-1].content[:200]}")
print()


# ──────────────────────────────────────────────────────────────────────────
# PART B: ModelRetryMiddleware
# ──────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PART B — ModelRetryMiddleware (handles model API errors)")
print("=" * 60)

agent_model_retry = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        ModelRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,    # 1s → 2s → 4s between model retries
        ),
    ],
)

print("Agent created with ModelRetryMiddleware.")
print("If gpt-4.1 returns a rate-limit 429, it retries up to 3 times.")
result = agent_model_retry.invoke({
    "messages": [{"role": "user", "content": "Hello! What day is it?"}]
})
print(f"Response: {result['messages'][-1].content[:200]}")
print()


# ──────────────────────────────────────────────────────────────────────────
# PART C: LLMToolEmulator
# ──────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PART C — LLMToolEmulator (fake tool execution for testing)")
print("=" * 60)

agent_emulated = create_agent(
    model="gpt-4.1",
    tools=[get_stock_price, send_notification],
    middleware=[
        LLMToolEmulator(),   # ALL tools are emulated, none run for real
    ],
)

result = agent_emulated.invoke({
    "messages": [{
        "role": "user",
        "content": (
            "Check the stock price of AAPL and TSLA, "
            "then notify user 'u123' with a summary."
        ),
    }]
})
print(f"Response (emulated tools): {result['messages'][-1].content[:400]}")
print()
print("NOTE: get_stock_price and send_notification were NEVER actually called.")
print("The LLMToolEmulator generated plausible fake responses from their docstrings.")


# ──────────────────────────────────────────────────────────────────────────
# PART D: Combined resilience stack
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART D — Full resilience stack: model retry + tool retry")
print("=" * 60)

resilient_agent = create_agent(
    model="gpt-4.1",
    tools=[flaky_api],
    middleware=[
        ModelRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0),
        ToolRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0),
    ],
)
print("Resilient agent created:")
print("  • Model failures retry up to 3x with backoff")
print("  • Tool failures retry up to 3x with backoff")
print("  • Both layers protect against transient errors independently")