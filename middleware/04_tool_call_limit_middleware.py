"""
TOPIC: ToolCallLimitMiddleware

WHAT IT DOES:
    Caps the number of times a TOOL (not the model) can be called.
    You can set a global limit for all tools, or a per-tool limit.

WHY THIS MATTERS:
    Tools like web search, database queries, or paid APIs cost money per call.
    An agent in a loop might call the same tool hundreds of times. Tool call
    limits act as a circuit breaker.

HOW IT WORKS:
    - thread_limit: max calls ever for this tool on this thread
    - run_limit:    max calls in a single .invoke() / .stream()
    - tool_name:    leave blank for global, set name for per-tool limit
    - Stack multiple ToolCallLimitMiddleware instances for different tools

CONFIGURATION:
    # Global limit — applies to ALL tools
    ToolCallLimitMiddleware(thread_limit=20, run_limit=10)

    # Per-tool limit — only applies to "search"
    ToolCallLimitMiddleware(tool_name="search", thread_limit=5, run_limit=3)

WHEN TO USE:
    Expensive paid APIs (search, embeddings, external data)
    Rate-limited endpoints
    Preventing database hammering
    Web scraping budgets
"""

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware


# ---------------------------------------------------------------------------
# 1. Tools — one expensive (search), one cheap (calculate)
# ---------------------------------------------------------------------------
_search_count = 0

def search(query: str) -> str:
    """Expensive web search — should be limited."""
    global _search_count
    _search_count += 1
    return f"[Search #{_search_count}] Results for '{query}': ..."


def calculate(expression: str) -> str:
    """Cheap calculator — can be called freely."""
    try:
        return str(eval(expression))
    except Exception:
        return "Invalid expression"


# ---------------------------------------------------------------------------
# 2. Agent with layered tool limits
#    - Global: max 10 tool calls per run across all tools
#    - search: additionally capped at 2 per run
# ---------------------------------------------------------------------------
agent = create_agent(
    model="gpt-4.1",
    tools=[search, calculate],
    middleware=[
        # Global guard — no more than 10 total tool calls per run
        ToolCallLimitMiddleware(
            thread_limit=50,
            run_limit=10,
        ),
        # Per-tool guard — search is expensive, cap it at 2 per run
        ToolCallLimitMiddleware(
            tool_name="search",
            thread_limit=10,
            run_limit=2,            # search can only fire 2 times per run
        ),
    ],
)


# ---------------------------------------------------------------------------
# 3. Run — ask the agent to search many times
# ---------------------------------------------------------------------------
print("=" * 60)
print("ToolCallLimitMiddleware — search capped at 2 per run")
print("=" * 60)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": (
            "Search for: Python, JavaScript, Rust, Go, and TypeScript. "
            "For each, tell me what it's best used for."
        ),
    }]
})

# Count search calls made
search_calls = [
    m for m in result["messages"]
    if type(m).__name__ == "ToolMessage"
    and getattr(m, "name", "") == "search"
]
print(f"Search calls made : {len(search_calls)} (limit was 2)")
print(f"\nFinal response excerpt:\n{result['messages'][-1].content[:400]}")

# ---------------------------------------------------------------------------
# EXPECTED BEHAVIOUR:
#
#   Agent tries to call search 5 times (one per language).
#   After 2 calls, the limit fires and search is no longer available.
#   Agent must answer the remaining 3 from its own knowledge.
#
#   search calls made: 2
# ---------------------------------------------------------------------------