"""
TOPIC: Dynamic Model Selection + Dynamic Tool Selection at Runtime

WHAT IT DOES:
    Instead of using the same model or the same tool set for every request,
    this middleware inspects the current state at runtime and swaps in a
    different model or a filtered tool set for each call.

WHY THIS MATTERS:
    - Short simple messages → use a cheap/fast model
    - Long complex messages → use a capable model
    - Query about finance   → expose only finance tools
    - Query about weather   → expose only weather tools

HOW IT WORKS:
    Both use wrap_model_call and request.override():

        # Override the model
        handler(request.override(model=my_other_model))

        # Override the tool list
        handler(request.override(tools=filtered_tools))

    request.override() creates a shallow copy with the specified fields replaced.
    The original request is unchanged.

KEY RULES FOR TOOL SELECTION:
    - ALL tools must be registered at create_agent() time
    - override(tools=...) picks a subset from those registered tools
    - You CANNOT add new tools dynamically that weren't registered

WHEN TO USE:
    Cost optimization (cheap model for simple queries)
    Capability routing (best model for complex reasoning)
    Permission control (user role determines available tools)
    Context-aware tool filtering (reduce irrelevant tools per query)
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
)
from langchain.chat_models import init_chat_model
from typing import Callable


# ---------------------------------------------------------------------------
# EXAMPLE 1: Dynamic Model Selection
#    Uses a fast cheap model for short queries,
#    a powerful model for long or complex ones.
# ---------------------------------------------------------------------------
fast_model    = init_chat_model("gpt-4.1-mini")   # cheap, fast
capable_model = init_chat_model("gpt-4.1")        # powerful, expensive


@wrap_model_call
def dynamic_model_selector(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """
    Pick model based on:
      - conversation length (>10 messages → complex → capable model)
      - message content complexity (long messages → capable model)
    """
    msg_count = len(request.messages)
    last_human = next(
        (m for m in reversed(request.messages) if m.type == "human"),
        None,
    )
    last_len = len(str(last_human.content)) if last_human else 0

    # Routing logic
    if msg_count > 10 or last_len > 500:
        chosen_model = capable_model
        reason = f"complex ({msg_count} msgs, {last_len} chars)"
    else:
        chosen_model = fast_model
        reason = f"simple ({msg_count} msgs, {last_len} chars)"

    print(f"🤖 [model_selector] Using {chosen_model.model_name} — {reason}")

    # Override the model for this one call
    return handler(request.override(model=chosen_model))


# ---------------------------------------------------------------------------
# EXAMPLE 2: Dynamic Tool Selection
#    Shows a subset of tools based on what the current query is about.
# ---------------------------------------------------------------------------

# All available tools — must ALL be registered at create_agent() time
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"22°C and sunny in {city}."

def get_forecast(city: str, days: int) -> str:
    """Get multi-day weather forecast."""
    return f"{days}-day forecast for {city}: sunny throughout."

def get_stock_price(ticker: str) -> str:
    """Get current stock price."""
    return f"{ticker}: $142.50"

def get_portfolio_value(user_id: str) -> str:
    """Get portfolio value for a user."""
    return f"Portfolio for {user_id}: $12,345.00"

def search_web(query: str) -> str:
    """Search the web."""
    return f"Search results for '{query}': [...]"

def translate_text(text: str, lang: str) -> str:
    """Translate text to a language."""
    return f"[{lang}] {text}"


# Tool groups — map domain → tool names
TOOL_GROUPS: dict[str, list[str]] = {
    "weather":  ["get_weather", "get_forecast"],
    "finance":  ["get_stock_price", "get_portfolio_value"],
    "general":  ["search_web", "translate_text"],
}

ALL_TOOLS = [get_weather, get_forecast, get_stock_price,
             get_portfolio_value, search_web, translate_text]


@wrap_model_call
def dynamic_tool_selector(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """
    Reduce the tool set based on keywords in the last user message.
    Always include 'search_web' as a fallback.
    """
    last_human = next(
        (m for m in reversed(request.messages) if m.type == "human"),
        None,
    )
    content = str(last_human.content).lower() if last_human else ""

    # Detect domain from keywords
    if any(w in content for w in ["weather", "forecast", "temperature", "rain"]):
        domain = "weather"
    elif any(w in content for w in ["stock", "price", "portfolio", "invest"]):
        domain = "finance"
    else:
        domain = "general"

    selected_names = set(TOOL_GROUPS[domain]) | {"search_web"}   # always include fallback

    # Filter to matching tool objects from registered tools
    relevant_tools = [t for t in request.tools
                      if (getattr(t, "name", None) or getattr(t, "__name__", "")) in selected_names]

    print(f"🔧 [tool_selector] Domain={domain} → tools: {[getattr(t,'name',getattr(t,'__name__','?')) for t in relevant_tools]}")

    # Override tools for this one model call
    return handler(request.override(tools=relevant_tools))


# ---------------------------------------------------------------------------
# Wire both into one agent
# ---------------------------------------------------------------------------
agent = create_agent(
    model="gpt-4.1",               # this is the default/fallback model
    tools=ALL_TOOLS,               # ALL tools registered here
    middleware=[
        dynamic_model_selector,    # picks the model per call
        dynamic_tool_selector,     # picks the tool subset per call
    ],
)

print("=" * 60)
print("Dynamic Model + Tool Selection")
print("=" * 60)

# Test 1: Short weather query → fast model + weather tools
print("\n─── Test 1: Weather query ───")
result = agent.invoke({"messages": [{"role": "user", "content": "Weather in Paris?"}]})
print(f"Answer: {result['messages'][-1].content[:200]}")

# Test 2: Finance query → fast model + finance tools
print("\n─── Test 2: Finance query ───")
result = agent.invoke({"messages": [{"role": "user", "content": "What's the stock price of AAPL?"}]})
print(f"Answer: {result['messages'][-1].content[:200]}")

# Test 3: Long complex query → capable model + general tools
print("\n─── Test 3: Long complex query ───")
long_msg = "I need you to " + "analyze this carefully. " * 30 + "What's the best approach?"
result = agent.invoke({"messages": [{"role": "user", "content": long_msg}]})
print(f"Answer: {result['messages'][-1].content[:200]}")