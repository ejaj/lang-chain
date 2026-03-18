"""
04_mcp_tools.py
=================
TOPIC: MCP Tools — Loading, Structured Content, Multimodal

WHAT ARE MCP TOOLS:
    Executable functions exposed by MCP servers that LLMs can invoke.
    LangChain converts them into standard LangChain tool objects —
    usable in any agent exactly like a Python @tool function.

THREE CAPABILITIES:
    1. Loading tools   → get_tools() converts MCP tools to LangChain tools
    2. Structured content → tools can return machine-parseable JSON alongside text
    3. Multimodal content → tools can return images, text, and other content types
"""

import asyncio
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain.agents import create_agent
from langchain.messages import ToolMessage
from mcp.types import TextContent


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Loading tools
# The simplest usage — get tools from server, pass to agent.
# ─────────────────────────────────────────────────────────────────────────────

async def loading_tools_example():
    """
    Load MCP tools and use them in an agent.
    client.get_tools() converts MCP tool definitions into LangChain tool objects.
    """
    client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command":   "python",
            "args":      ["/path/to/math_server.py"],
        }
    })

    # Convert MCP tools → LangChain tools
    tools = await client.get_tools()

    print(f"Loaded {len(tools)} tools:")
    for t in tools:
        print(f"  - {t.name}: {t.description}")

    # Use exactly like any other LangChain tool
    agent = create_agent("claude-sonnet-4-6", tools)

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is 15 + 27?"}]
    })
    print(f"Result: {result['messages'][-1].content}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Structured content
# MCP tools can return both human-readable text AND machine-parseable JSON.
# The JSON lives in message.artifact["structured_content"].
# ─────────────────────────────────────────────────────────────────────────────

async def structured_content_example():
    """
    Extract structured JSON data returned by MCP tools.
    Useful when you need both a readable response AND data to process in code.
    """
    client = MultiServerMCPClient({
        "data": {
            "transport": "http",
            "url":       "http://localhost:8001/mcp",
        }
    })

    tools = await client.get_tools()
    agent = create_agent("claude-sonnet-4-6", tools)

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Get data from the server"}]
    })

    # Extract structured content from tool messages
    for message in result["messages"]:
        if isinstance(message, ToolMessage) and message.artifact:
            # structured_content is the machine-parseable JSON part
            structured = message.artifact["structured_content"]
            print(f"Structured data: {structured}")
            print(f"Type: {type(structured)}")


async def structured_content_via_interceptor():
    """
    Append structured content to the conversation history via interceptor.
    By default, structured content is NOT visible to the model.
    This interceptor makes it visible as an additional text message.
    """

    async def append_structured_content(
        request: MCPToolCallRequest,
        handler,
    ):
        """Make structured content visible to the model in conversation history."""
        result = await handler(request)

        if result.structuredContent:
            # Append the JSON as a text block so the model can see it
            result.content += [
                TextContent(
                    type="text",
                    text=json.dumps(result.structuredContent),
                ),
            ]
        return result

    client = MultiServerMCPClient(
        {
            "data": {
                "transport": "http",
                "url":       "http://localhost:8001/mcp",
            }
        },
        tool_interceptors=[append_structured_content],  # attach interceptor
    )

    tools = await client.get_tools()
    agent = create_agent("claude-sonnet-4-6", tools)

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Analyze the server data"}]
    })
    print(f"Result: {result['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: Multimodal tool content
# MCP tools can return text + images + other content types.
# Use content_blocks for provider-agnostic access to all content parts.
# ─────────────────────────────────────────────────────────────────────────────

async def multimodal_content_example():
    """
    Handle tools that return images, text, and other content types.
    content_blocks gives a normalized, provider-agnostic representation.
    """
    client = MultiServerMCPClient({
        "browser": {
            "transport": "http",
            "url":       "http://localhost:8002/mcp",
        }
    })

    tools = await client.get_tools()
    agent = create_agent("claude-sonnet-4-6", tools)

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Take a screenshot of the current page"}]
    })

    # Access multimodal content from tool messages
    for message in result["messages"]:
        if message.type == "tool":
            print(f"Raw content type: {type(message.content)}")

            # content_blocks = normalized, provider-agnostic format
            for block in message.content_blocks:
                if block["type"] == "text":
                    print(f"Text block: {block['text'][:100]}")

                elif block["type"] == "image":
                    url    = block.get("url")
                    b64    = block.get("base64", "")
                    mime   = block.get("mime_type", "image/png")
                    if url:
                        print(f"Image URL: {url}")
                    elif b64:
                        print(f"Image base64 ({mime}): {b64[:50]}...")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("""
MCP Tools Quick Reference:

  Loading:
    tools = await client.get_tools()
    agent = create_agent("model", tools)

  Structured content (post-invoke):
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and msg.artifact:
            data = msg.artifact["structured_content"]

  Structured content (via interceptor, visible to model):
    async def append_structured(request, handler):
        result = await handler(request)
        if result.structuredContent:
            result.content += [TextContent(type="text", text=json.dumps(result.structuredContent))]
        return result

  Multimodal content:
    for msg in result["messages"]:
        if msg.type == "tool":
            for block in msg.content_blocks:
                block["type"]  # "text" | "image"
                block["text"]  # if text
                block["url"]   # if image URL
                block["base64"] # if image base64
""")

if __name__ == "__main__":
    asyncio.run(loading_tools_example())