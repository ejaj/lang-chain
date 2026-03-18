"""
TOPIC: MCP Stateful Sessions

DEFAULT BEHAVIOUR (stateless):
    MultiServerMCPClient creates a fresh MCP session for every tool call.
    → Tool runs → session cleaned up → next call starts fresh.
    Safe, simple, works for most cases.

WHEN YOU NEED STATEFUL:
    Some MCP servers maintain context BETWEEN tool calls:
    - A shell server that remembers your working directory
    - A browser server that keeps a tab open
    - A database server with an active transaction
    - Any server where Tool Call 2 depends on what Tool Call 1 did

HOW TO USE STATEFUL SESSIONS:
    Use client.session("server_name") as an async context manager.
    The session stays alive for the entire block.
    Pass it to load_mcp_tools() to get tools bound to that session.

DIFFERENCE:
    Stateless: client.get_tools()               → new session per tool call
    Stateful:  async with client.session(...):  → one session for all calls
"""

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent


# ─────────────────────────────────────────────────────────────────────────────
# STATELESS (default) — each tool call is independent
# ─────────────────────────────────────────────────────────────────────────────

async def stateless_example():
    """
    Default stateless behaviour.
    Good for: tools that don't depend on each other's state.
    Example: math operations, weather lookups, independent API calls.
    """
    client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command":   "python",
            "args":      ["/path/to/math_server.py"],
        }
    })

    # get_tools() → stateless, each tool call gets its own session
    tools = await client.get_tools()
    agent = create_agent("claude-sonnet-4-6", tools)

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is 3 + 5?"}]
    })
    print(f"Stateless result: {result['messages'][-1].content}")


# ─────────────────────────────────────────────────────────────────────────────
# STATEFUL — one session shared across all tool calls
# ─────────────────────────────────────────────────────────────────────────────

async def stateful_example():
    """
    Stateful session — one persistent MCP connection.
    Good for: tools that build on each other's state.
    Example: shell commands (cd then ls), browser tabs, DB transactions.
    """
    client = MultiServerMCPClient({
        "shell": {
            "transport": "stdio",
            "command":   "python",
            "args":      ["/path/to/shell_server.py"],
        }
    })

    # async with client.session() keeps the connection alive for the entire block
    async with client.session("shell") as session:
        # load_mcp_tools() binds the tools to THIS specific session
        tools = await load_mcp_tools(session)

        agent = create_agent("claude-sonnet-4-6", tools)

        # Both tool calls use the SAME session — state is preserved between them
        # e.g. cd /tmp in call 1 is remembered in call 2
        result = await agent.ainvoke({
            "messages": [{
                "role":    "user",
                "content": "Change directory to /tmp, then list the files there.",
            }]
        })
        print(f"Stateful result: {result['messages'][-1].content}")

    # Session is automatically closed when the block exits


# ─────────────────────────────────────────────────────────────────────────────
# STATEFUL with resources and prompts
# The session also gives access to resources and prompts from that server.
# ─────────────────────────────────────────────────────────────────────────────

async def stateful_with_resources_example():
    """Load tools, resources, and prompts from the same session."""
    from langchain_mcp_adapters.resources import load_mcp_resources
    from langchain_mcp_adapters.prompts import load_mcp_prompt

    client = MultiServerMCPClient({
        "docs": {
            "transport": "http",
            "url":       "http://localhost:8001/mcp",
        }
    })

    async with client.session("docs") as session:
        # Load tools
        tools     = await load_mcp_tools(session)

        # Load resources (files, data) from the same session
        resources = await load_mcp_resources(session)

        # Load a specific prompt template from the same session
        messages  = await load_mcp_prompt(session, "summarize")

        print(f"Tools loaded     : {len(tools)}")
        print(f"Resources loaded : {len(resources)}")
        print(f"Prompt messages  : {len(messages)}")

        # Build context from resources
        context = "\n".join(r.as_string() for r in resources if r.mimetype == "text/plain")

        agent = create_agent("gpt-4.1", tools)
        result = await agent.ainvoke({
            "messages": messages + [{"role": "user", "content": context}]
        })
        print(f"Result: {result['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("""
Stateless vs Stateful:

  Stateless (default):
    tools = await client.get_tools()
    # Each tool call → new session → tool runs → session closed

  Stateful:
    async with client.session("server_name") as session:
        tools = await load_mcp_tools(session)
        # All tool calls → same session → state preserved between calls

  Use stateful when:
    Shell server (working directory must persist)
    Browser server (keep the same tab open)
    Database server (active transaction)
    Any server where Tool B depends on what Tool A did
""")

if __name__ == "__main__":
    asyncio.run(stateless_example())