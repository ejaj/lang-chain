"""
TOPIC: MCP Transports — HTTP, Headers, Auth, stdio

WHAT IS A TRANSPORT:
    The communication mechanism between your Python code (client)
    and the MCP server process. Different transports suit different
    deployment scenarios.

THREE TRANSPORTS:
    http   → remote server over HTTP (also called streamable-http)
             Best for: cloud-deployed servers, microservices, shared tools
    stdio  → local subprocess via stdin/stdout
             Best for: local tools, simple setups, development
    sse    → Server-Sent Events (deprecated by MCP spec, avoid for new work)

CHOOSING A TRANSPORT:
    Local dev / simple tools  → stdio
    Remote / production       → http
    Auth required             → http + headers or auth
"""

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


# ─────────────────────────────────────────────────────────────────────────────
# TRANSPORT 1: HTTP
# Remote server reachable via URL.
# ─────────────────────────────────────────────────────────────────────────────

async def http_transport_example():
    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "http",                    # streamable HTTP
                "url":       "http://localhost:8000/mcp",
            }
        }
    )

    tools = await client.get_tools()
    agent = create_agent("gpt-4.1", tools)
    return await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is the weather in NYC?"}]
    })


# ─────────────────────────────────────────────────────────────────────────────
# TRANSPORT 2: HTTP with custom headers
# Pass headers for authentication, tracing, tenant identification, etc.
# Supported for http and sse transports.
# ─────────────────────────────────────────────────────────────────────────────

async def http_with_headers_example():
    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "http",
                "url":       "http://localhost:8000/mcp",
                "headers": {
                    "Authorization":  "Bearer YOUR_TOKEN",   # auth token
                    "X-Tenant-ID":    "tenant-42",           # tenant routing
                    "X-Custom-Header": "custom-value",       # any header
                },
            }
        }
    )

    tools = await client.get_tools()
    agent = create_agent("gpt-4.1", tools)
    return await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is the weather in NYC?"}]
    })


# ─────────────────────────────────────────────────────────────────────────────
# TRANSPORT 3: HTTP with custom auth
# Implement the httpx.Auth interface for full control over authentication.
# Use when you need OAuth, API key signing, or custom auth schemes.
# ─────────────────────────────────────────────────────────────────────────────

import httpx

class BearerAuth(httpx.Auth):
    """Simple Bearer token auth — implements httpx.Auth interface."""

    def __init__(self, token: str):
        self.token = token

    def auth_flow(self, request: httpx.Request):
        """Add Authorization header to every request."""
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


class ApiKeyAuth(httpx.Auth):
    """API key auth — adds key as a query parameter."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def auth_flow(self, request: httpx.Request):
        request.url = request.url.copy_add_param("api_key", self.api_key)
        yield request


async def http_with_auth_example():
    auth = BearerAuth(token="my-secret-token")

    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "http",
                "url":       "http://localhost:8000/mcp",
                "auth":      auth,   # custom auth — applied to every request
            }
        }
    )

    tools = await client.get_tools()
    agent = create_agent("gpt-4.1", tools)
    return await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is the weather in NYC?"}]
    })


# ─────────────────────────────────────────────────────────────────────────────
# TRANSPORT 4: stdio
# Launches the server as a local subprocess.
# Communicates via standard input/output — no network needed.
# ─────────────────────────────────────────────────────────────────────────────

async def stdio_transport_example():
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",          # local subprocess
                "command":   "python",         # command to launch server
                "args":      ["/path/to/math_server.py"],  # server script
                # Optional: pass env vars to the subprocess
                # "env": {"PYTHONPATH": "/my/path"},
            }
        }
    )

    tools = await client.get_tools()
    agent = create_agent("claude-sonnet-4-6", tools)
    return await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is 7 multiplied by 8?"}]
    })


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED: Multiple servers, different transports
# ─────────────────────────────────────────────────────────────────────────────

async def combined_transports_example():
    """Connect to local and remote servers simultaneously."""
    client = MultiServerMCPClient(
        {
            # Local math server (stdio)
            "math": {
                "transport": "stdio",
                "command":   "python",
                "args":      ["/path/to/math_server.py"],
            },

            # Remote weather server (http, no auth)
            "weather": {
                "transport": "http",
                "url":       "http://weather-service.example.com/mcp",
            },

            # Remote database server (http + Bearer token)
            "database": {
                "transport": "http",
                "url":       "http://db-service.example.com/mcp",
                "headers": {"Authorization": "Bearer db-token-xyz"},
            },
        }
    )

    tools = await client.get_tools()
    print(f"Total tools loaded: {len(tools)}")
    for t in tools:
        print(f"  - {t.name}")

    agent = create_agent("gpt-4.1", tools)
    return await agent.ainvoke({
        "messages": [{"role": "user", "content": "Add 10 + 5 and check the weather in London"}]
    })


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("""
Transport Quick Reference:

  HTTP (remote server):
    {"transport": "http", "url": "http://host/mcp"}

  HTTP + headers:
    {"transport": "http", "url": "...", "headers": {"Authorization": "Bearer token"}}

  HTTP + custom auth:
    {"transport": "http", "url": "...", "auth": my_httpx_auth_instance}

  stdio (local subprocess):
    {"transport": "stdio", "command": "python", "args": ["/path/to/server.py"]}

  When to use:
    stdio  → local dev, simple setups, no network needed
    http   → remote/production, shared servers, microservices
    + auth → when server requires authentication
""")

if __name__ == "__main__":
    asyncio.run(http_transport_example())