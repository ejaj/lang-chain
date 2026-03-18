"""
TOPIC: MCP Prompts

WHAT ARE MCP PROMPTS:
    Reusable prompt templates exposed by MCP servers.
    Instead of hardcoding prompts in your client code, the server
    manages them — they can be versioned, shared, and updated centrally.

    Use cases:
    Consistent prompts across multiple agents / teams
    Server-managed prompt versioning
    Parameterized templates (fill in language, topic, etc.)
    Standardized workflows (code review, summarization, translation)

HOW THEY WORK:
    client.get_prompt("server", "prompt_name") returns a list of messages
    (HumanMessage, AIMessage, SystemMessage) ready to pass to an agent.
    Parameterized prompts accept an arguments dict.

TWO WAYS TO LOAD:
    client.get_prompt(...)           → simple, stateless
    load_mcp_prompt(session, ...)    → inside a stateful session
"""

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langchain.agents import create_agent


# ─────────────────────────────────────────────────────────────────────────────
# LOADING PROMPTS — simple approach
# ─────────────────────────────────────────────────────────────────────────────

async def load_simple_prompt():
    """Load a prompt by name — no arguments needed."""
    client = MultiServerMCPClient({
        "templates": {
            "transport": "http",
            "url":       "http://localhost:8001/mcp",
        }
    })

    # Load the "summarize" prompt template from the server
    messages = await client.get_prompt("templates", "summarize")

    print(f"Prompt has {len(messages)} messages:")
    for msg in messages:
        print(f"  [{msg.type}]: {str(msg.content)[:100]}")

    return messages


async def load_parameterized_prompt():
    """Load a prompt with arguments — fills in the template."""
    client = MultiServerMCPClient({
        "templates": {
            "transport": "http",
            "url":       "http://localhost:8001/mcp",
        }
    })

    # Load "code_review" prompt with specific arguments
    messages = await client.get_prompt(
        "templates",
        "code_review",
        arguments={
            "language": "python",
            "focus":    "security",
        }
    )

    print(f"Code review prompt ({len(messages)} messages):")
    for msg in messages:
        print(f"  [{msg.type}]: {str(msg.content)[:150]}")

    return messages


# ─────────────────────────────────────────────────────────────────────────────
# LOADING PROMPTS inside a stateful session
# ─────────────────────────────────────────────────────────────────────────────

async def load_prompt_in_session():
    """Load prompts alongside tools in a stateful session."""
    from langchain_mcp_adapters.tools import load_mcp_tools

    client = MultiServerMCPClient({
        "assistant": {
            "transport": "http",
            "url":       "http://localhost:8001/mcp",
        }
    })

    async with client.session("assistant") as session:
        # Load tools and prompt from the same session
        tools    = await load_mcp_tools(session)
        messages = await load_mcp_prompt(session, "summarize")

        # Load parameterized prompt
        review_messages = await load_mcp_prompt(
            session,
            "code_review",
            arguments={"language": "python", "focus": "performance"},
        )

        print(f"Tools   : {[t.name for t in tools]}")
        print(f"Prompt  : {len(messages)} messages")
        print(f"Review  : {len(review_messages)} messages")

        agent  = create_agent("gpt-4.1", tools)
        result = await agent.ainvoke({"messages": messages})
        print(f"Result  : {result['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# USING PROMPTS AS AGENT STARTING MESSAGES
# Prompts set up the agent with pre-built instructions from the server.
# ─────────────────────────────────────────────────────────────────────────────

async def prompts_as_agent_messages():
    """
    Practical pattern: load a prompt, add a user message, run the agent.
    The server-managed prompt provides the system/context; you add the query.
    """
    client = MultiServerMCPClient({
        "templates": {
            "transport": "http",
            "url":       "http://localhost:8001/mcp",
        }
    })

    tools = await client.get_tools()

    # Load a translation prompt template
    prompt_messages = await client.get_prompt(
        "templates",
        "translate",
        arguments={"target_language": "French"},
    )

    # Append the user's actual content to the server-managed prompt
    user_message = {
        "role":    "user",
        "content": "The quick brown fox jumps over the lazy dog.",
    }

    all_messages = prompt_messages + [user_message]

    agent  = create_agent("gpt-4.1", tools)
    result = await agent.ainvoke({"messages": all_messages})
    print(f"Translation: {result['messages'][-1].content}")


async def multiple_prompts_example():
    """Load and compare multiple prompt variations from the server."""
    client = MultiServerMCPClient({
        "templates": {
            "transport": "http",
            "url":       "http://localhost:8001/mcp",
        }
    })

    tools  = await client.get_tools()
    agent  = create_agent("gpt-4.1", tools)
    sample = "Alice went to the store to buy groceries."

    for style in ["formal", "casual", "technical"]:
        messages = await client.get_prompt(
            "templates",
            "rephrase",
            arguments={"style": style},
        )
        messages = messages + [{"role": "user", "content": sample}]
        result   = await agent.ainvoke({"messages": messages})
        print(f"{style:10}: {result['messages'][-1].content[:150]}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("""
MCP Prompts Quick Reference:

  Load by name (no args):
    messages = await client.get_prompt("server_name", "prompt_name")

  Load with arguments (parameterized):
    messages = await client.get_prompt(
        "server_name",
        "prompt_name",
        arguments={"key": "value"}
    )

  Inside a session:
    async with client.session("server_name") as session:
        messages = await load_mcp_prompt(session, "prompt_name")
        messages = await load_mcp_prompt(session, "prompt_name", arguments={...})

  Using the loaded messages:
    messages returns a list of LangChain messages (HumanMessage, AIMessage, etc.)
    Pass them directly to agent.ainvoke({"messages": prompt_messages + [user_msg]})

  Prompts vs Tools vs Resources:
    Tools     → DO things (run code, call APIs)
    Resources → ARE data (files, documents to read)
    Prompts   → ARE templates (pre-built instructions to use)
""")

if __name__ == "__main__":
    asyncio.run(load_simple_prompt())