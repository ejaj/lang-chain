"""
10_filesystem_and_subagent_middleware.py
==========================================
TOPIC: FilesystemMiddleware + SubAgentMiddleware

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART A — FilesystemMiddleware (from deep agents)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IT DOES:
    Gives the agent a virtual filesystem with 4 tools:
        ls          → list files
        read_file   → read file contents
        write_file  → create new file
        edit_file   → modify existing file

WHY THIS MATTERS:
    Tool outputs (web search, RAG, database) can be huge and fill the context
    window. With a filesystem, the agent can write results to disk and read
    only what it needs — context stays lean.

STORAGE BACKENDS:
    StateBackend    → ephemeral, lives in graph state (default)
    StoreBackend    → persistent, survives across threads
    CompositeBackend → route paths to different backends
                       e.g. /memories/ → persistent, everything else → ephemeral

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART B — SubAgentMiddleware (from deep agents)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IT DOES:
    Adds a `task` tool that lets the main agent delegate work to
    specialist sub-agents. Each sub-agent runs in its own context,
    keeping the supervisor's context window clean.

WHY THIS MATTERS:
    A supervisor that does everything accumulates huge context from
    intermediate tool calls. Delegating to a sub-agent means only the
    final answer comes back — the intermediate work is isolated.

SUBAGENT TYPES:
    Dict definition    → simple name + description + tools + model
    CompiledSubAgent   → wrap any LangGraph graph as a sub-agent
    General-purpose    → always available, same tools as supervisor,
                         primary use: context isolation

WHEN TO USE:
    Parallel specialist agents (weather, finance, code)
    Any task where intermediate tool calls would pollute supervisor context
    Long deep-dive tasks that need many steps
"""

# ── Imports ────────────────────────────────────────────────────────────────
from langchain.tools import tool
from langchain.agents import create_agent

# deep agents (install: pip install deepagents)
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents import CompiledSubAgent
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph


# ──────────────────────────────────────────────────────────────────────────
# PART A: FilesystemMiddleware — ephemeral (default)
# ──────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PART A1 — FilesystemMiddleware (ephemeral, default backend)")
print("=" * 60)

agent_fs = create_agent(
    model="claude-sonnet-4-6",
    middleware=[
        FilesystemMiddleware(),   # 👈 adds ls, read_file, write_file, edit_file
    ],
)

result = agent_fs.invoke({
    "messages": [{
        "role": "user",
        "content": (
            "Write a file called notes.txt with the content 'Meeting at 3pm'. "
            "Then read it back and confirm the content."
        ),
    }]
})
print(f"Response: {result['messages'][-1].content[:300]}")
print()


# ──────────────────────────────────────────────────────────────────────────
# PART A2: FilesystemMiddleware — persistent /memories/ path
# ──────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PART A2 — FilesystemMiddleware with persistent /memories/ path")
print("=" * 60)

store = InMemoryStore()   # use Redis/DynamoDB in production

agent_persistent = create_agent(
    model="claude-sonnet-4-6",
    store=store,
    middleware=[
        FilesystemMiddleware(
            backend=lambda rt: CompositeBackend(
                default=StateBackend(rt),           # /data/  → ephemeral
                routes={"/memories/": StoreBackend(rt)}  # /memories/ → persistent
            ),
            custom_tool_descriptions={
                "write_file": (
                    "Write to /memories/ for anything you want to remember "
                    "across conversations. Write to /data/ for temporary work."
                ),
                "ls": "List files. Use /memories/ for persistent storage.",
            },
        ),
    ],
)

# First run: agent saves a memory
result1 = agent_persistent.invoke({
    "messages": [{"role": "user", "content":
        "Remember that the user's name is Alice and she prefers dark mode."}],
    "configurable": {"thread_id": "thread_001"},
})
print(f"Run 1: {result1['messages'][-1].content[:200]}")

# Second run (different thread): agent can still read /memories/
result2 = agent_persistent.invoke({
    "messages": [{"role": "user", "content": "What do you remember about the user?"}],
    "configurable": {"thread_id": "thread_002"},  # different thread!
})
print(f"Run 2: {result2['messages'][-1].content[:200]}")
print("(Memory persisted across threads via StoreBackend)")
print()

print("=" * 60)
print("PART B1 — SubAgentMiddleware (dict-defined subagents)")
print("=" * 60)


@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is 22°C and sunny."


@tool
def get_stock_price(ticker: str) -> str:
    """Get the stock price for a ticker."""
    return f"{ticker}: $142.50 (+1.2%)"


agent_supervisor = create_agent(
    model="claude-sonnet-4-6",
    middleware=[
        SubAgentMiddleware(
            default_model="claude-sonnet-4-6",
            default_tools=[],
            subagents=[
                {
                    "name": "weather_agent",
                    "description": "Specialist for weather queries. Use for any weather question.",
                    "system_prompt": "Use the get_weather tool to answer weather questions.",
                    "tools": [get_weather],
                    "model": "gpt-4.1",      # subagent can use a different model
                    "middleware": [],
                },
                {
                    "name": "finance_agent",
                    "description": "Specialist for stock prices and financial data.",
                    "system_prompt": "Use get_stock_price to answer financial questions.",
                    "tools": [get_stock_price],
                    "model": "gpt-4.1-mini",  # cheaper model for simpler tasks
                    "middleware": [],
                },
            ],
        )
    ],
)

result = agent_supervisor.invoke({
    "messages": [{
        "role": "user",
        "content": "What's the weather in Tokyo AND the price of AAPL?",
    }]
})
print(f"Response: {result['messages'][-1].content[:400]}")
# Supervisor delegates to weather_agent for weather, finance_agent for stock.
# Supervisor's context only sees the final answers, not intermediate tool calls.
print()


# ──────────────────────────────────────────────────────────────────────────
# PART B2: SubAgentMiddleware — CompiledSubAgent (custom LangGraph graph)
# ──────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PART B2 — SubAgentMiddleware with CompiledSubAgent (custom graph)")
print("=" * 60)

# Build a minimal custom LangGraph graph
from typing import TypedDict, Annotated
from langgraph.graph import END
import operator

class WeatherState(TypedDict):
    messages: Annotated[list, operator.add]

def weather_node(state: WeatherState) -> dict:
    last = state["messages"][-1]
    return {"messages": [{"role": "assistant",
                          "content": "Custom graph says: It's sunny everywhere!"}]}

workflow = StateGraph(WeatherState)
workflow.add_node("weather", weather_node)
workflow.set_entry_point("weather")
workflow.add_edge("weather", END)
weather_graph = workflow.compile()

# Wrap in CompiledSubAgent
weather_subagent = CompiledSubAgent(
    name="custom_weather",
    description="A custom LangGraph graph that answers weather questions.",
    runnable=weather_graph,
)

agent_custom = create_agent(
    model="claude-sonnet-4-6",
    middleware=[
        SubAgentMiddleware(
            default_model="claude-sonnet-4-6",
            default_tools=[],
            subagents=[weather_subagent],   # CompiledSubAgent
        )
    ],
)

result = agent_custom.invoke({
    "messages": [{"role": "user", "content": "What's the weather like in London?"}]
})
print(f"Response: {result['messages'][-1].content[:300]}")