"""
TOPIC: MCP Quickstart + Custom Servers

WHAT IS MCP:
    Model Context Protocol — an open protocol that standardizes how
    applications provide tools and context to LLMs.
    Instead of writing tools directly in Python, you run them as
    separate server processes that any MCP-compatible client can use.

WHY MCP:
    Tools run as isolated processes — language agnostic
    Reuse the same tool server across multiple agents
    Remote tool servers (HTTP) or local subprocesses (stdio)
    Ecosystem of pre-built MCP servers (databases, APIs, filesystems)

HOW IT WORKS:
    1. Start one or more MCP servers (local subprocess or remote HTTP)
    2. Create a MultiServerMCPClient pointing at those servers
    3. Call client.get_tools() to load tools as LangChain tool objects
    4. Pass those tools to create_agent() like any other tool

INSTALL:
    pip install langchain-mcp-adapters fastmcp

TWO PARTS:
    Part A — Custom MCP server (math_server.py)
    Part B — Agent that connects to it (this file)
"""

# ─────────────────────────────────────────────────────────────────────────────
# PART A: Custom MCP Server
# Save this as math_server.py and run it separately.
# The server exposes two tools: add and multiply.
# ─────────────────────────────────────────────────────────────────────────────

MATH_SERVER_CODE = '''
# math_server.py
from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")   # communicates via stdin/stdout
'''

print("=" * 60)
print("math_server.py contents:")
print("=" * 60)
print(MATH_SERVER_CODE)


# ─────────────────────────────────────────────────────────────────────────────
# PART B: Agent connecting to multiple MCP servers
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


async def main():
    # Connect to multiple MCP servers at once
    client = MultiServerMCPClient(
        {
            # Local server — launched as a subprocess, communicates via stdio
            "math": {
                "transport": "stdio",
                "command":   "python",
                "args":      ["/path/to/math_server.py"],   # absolute path
            },

            # Remote server — communicates over HTTP
            "weather": {
                "transport": "http",
                "url":       "http://localhost:8000/mcp",
            },
        }
    )

    # Load all tools from all connected servers as LangChain tools
    tools = await client.get_tools()

    print(f"Loaded {len(tools)} tools from MCP servers:")
    for t in tools:
        print(f"  - {t.name}: {t.description}")

    # Create agent with MCP tools — identical to using any other tool
    agent = create_agent("claude-sonnet-4-6", tools)

    # Use math tools (from local math server)
    math_response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is (3 + 5) x 12?"}]
    })
    print(f"\nMath: {math_response['messages'][-1].content}")

    # Use weather tools (from remote weather server)
    weather_response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is the weather in NYC?"}]
    })
    print(f"Weather: {weather_response['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())


# ─────────────────────────────────────────────────────────────────────────────
# KEY NOTES
# ─────────────────────────────────────────────────────────────────────────────
print("""
Key notes:

  MultiServerMCPClient is STATELESS by default.
  Each tool call creates a fresh MCP session, runs the tool, then cleans up.
  For stateful servers, see 03_mcp_stateful_sessions.py.

  Tool loading:
    tools = await client.get_tools()
    agent = create_agent("model-name", tools)

  Always use await / async — MCP client is fully async.
""")