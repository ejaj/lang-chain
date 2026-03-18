"""
TOPIC: MCP Resources

WHAT ARE MCP RESOURCES:
    Data exposed by MCP servers — files, database records, API responses,
    configuration, documents — anything that can be READ by the client.
    Resources are different from tools: tools DO things, resources ARE data.

HOW THEY WORK:
    Resources are converted to Blob objects — a unified interface for
    handling both text and binary content regardless of source.

    blob.as_string()          → text content
    blob.data                 → raw bytes
    blob.mimetype             → "text/plain", "application/json", "image/png" etc.
    blob.metadata["uri"]      → the resource identifier

TWO WAYS TO LOAD:
    client.get_resources("server")           → simple, stateless
    load_mcp_resources(session)              → inside a stateful session
"""

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain.agents import create_agent


# ─────────────────────────────────────────────────────────────────────────────
# LOADING RESOURCES — simple approach
# ─────────────────────────────────────────────────────────────────────────────

async def load_all_resources():
    """Load all available resources from a server."""
    client = MultiServerMCPClient({
        "docs": {
            "transport": "http",
            "url":       "http://localhost:8001/mcp",
        }
    })

    # Load ALL resources from the "docs" server
    blobs = await client.get_resources("docs")

    print(f"Loaded {len(blobs)} resources:")
    for blob in blobs:
        uri      = blob.metadata["uri"]
        mimetype = blob.mimetype
        print(f"  URI: {uri}  |  Type: {mimetype}")

        if mimetype and mimetype.startswith("text"):
            preview = blob.as_string()[:100]
            print(f"  Preview: {preview}")


async def load_specific_resources():
    """Load only specific resources by URI."""
    client = MultiServerMCPClient({
        "docs": {
            "transport": "http",
            "url":       "http://localhost:8001/mcp",
        }
    })

    # Load specific resources by their URI
    blobs = await client.get_resources(
        "docs",
        uris=[
            "file:///path/to/readme.txt",
            "file:///path/to/config.json",
        ]
    )

    for blob in blobs:
        print(f"URI: {blob.metadata['uri']}")
        print(f"Type: {blob.mimetype}")
        print(f"Content: {blob.as_string()[:200]}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# LOADING RESOURCES inside a stateful session
# Use when you also need tools or prompts from the same server.
# ─────────────────────────────────────────────────────────────────────────────

async def load_resources_in_session():
    """Load resources alongside tools in a stateful session."""
    from langchain_mcp_adapters.tools import load_mcp_tools

    client = MultiServerMCPClient({
        "docs": {
            "transport": "http",
            "url":       "http://localhost:8001/mcp",
        }
    })

    async with client.session("docs") as session:
        # Load both tools and resources from the same session
        tools = await load_mcp_tools(session)
        blobs = await load_mcp_resources(session)

        # Load specific resources by URI within the session
        specific = await load_mcp_resources(
            session,
            uris=["file:///docs/user_guide.txt"]
        )

        print(f"Tools    : {[t.name for t in tools]}")
        print(f"Resources: {len(blobs)}")

        # Build context from text resources to inject into agent
        text_context = "\n\n".join(
            b.as_string()
            for b in blobs
            if b.mimetype and b.mimetype.startswith("text")
        )

        agent = create_agent("gpt-4.1", tools)
        result = await agent.ainvoke({
            "messages": [
                {
                    "role":    "system",
                    "content": f"Here is the documentation context:\n{text_context}",
                },
                {
                    "role":    "user",
                    "content": "Summarize the key points from the documentation",
                },
            ]
        })
        print(f"Summary: {result['messages'][-1].content[:300]}")


# ─────────────────────────────────────────────────────────────────────────────
# USING RESOURCES AS AGENT CONTEXT
# Inject loaded resources into the agent's messages as context.
# ─────────────────────────────────────────────────────────────────────────────

async def resources_as_agent_context():
    """
    Practical pattern: load resources, inject as context, query the agent.
    Resources become part of the agent's knowledge for this session.
    """
    client = MultiServerMCPClient({
        "knowledge": {
            "transport": "http",
            "url": "http://localhost:8002/mcp",
        }
    })

    tools = await client.get_tools()
    blobs = await client.get_resources("knowledge")

    # Separate text and binary resources
    text_blobs   = [b for b in blobs if b.mimetype and b.mimetype.startswith("text")]
    binary_blobs = [b for b in blobs if b.mimetype and b.mimetype.startswith("image")]

    print(f"Text resources  : {len(text_blobs)}")
    print(f"Image resources : {len(binary_blobs)}")

    # Build a context message from text resources
    context_parts = []
    for blob in text_blobs:
        uri     = blob.metadata["uri"]
        content = blob.as_string()
        context_parts.append(f"[{uri}]\n{content}")

    context_message = {
        "role": "system",
        "content": "Available documents:\n\n" + "\n\n---\n\n".join(context_parts),
    }

    agent = create_agent("gpt-4.1", tools)
    result = await agent.ainvoke({
        "messages": [
            context_message,
            {"role": "user", "content": "What are the main topics covered in the documents?"},
        ]
    })
    print(f"Result: {result['messages'][-1].content[:300]}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("""
MCP Resources Quick Reference:

  Load all resources:
    blobs = await client.get_resources("server_name")

  Load specific resources:
    blobs = await client.get_resources("server_name", uris=["file:///path"])

  Inside a session:
    async with client.session("server_name") as session:
        blobs = await load_mcp_resources(session)
        blobs = await load_mcp_resources(session, uris=["file:///path"])

  Blob API:
    blob.as_string()        → text content (for text resources)
    blob.data                 → raw bytes (for binary resources)
    blob.mimetype             → "text/plain", "image/png", etc.
    blob.metadata["uri"]      → resource identifier

  Resources vs Tools:
    Tools  → DO things (run code, call APIs, query databases)
    Resources → ARE data (files, records, documents to read)
""")

if __name__ == "__main__":
    asyncio.run(load_all_resources())