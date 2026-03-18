"""
TOPIC: MCP Elicitation

WHAT IS ELICITATION:
    A mechanism that lets MCP SERVERS ask the CLIENT for additional input
    WHILE a tool is running — without requiring all inputs upfront.

    Normal flow:  Client → tool args → Server → result → Client
    Elicitation:  Client → tool args → Server → "I need more info" →
                  Client handles request → Client sends data → Server → result

WHY IT EXISTS:
    Some tools don't know what they need until they start running.
    Example: "Create a user profile" — the server starts, realizes it needs
    email and age, and asks the client for those details mid-execution.

TWO PARTS:
    SERVER SIDE: tool calls ctx.elicit(message, schema) to pause and ask
    CLIENT SIDE: provide on_elicitation callback to handle the request

THREE RESPONSE ACTIONS:
    "accept"  → user provided the data, include in content field
    "decline" → user chose not to provide the information
    "cancel"  → user wants to abort the operation entirely
"""

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext

try:
    from mcp.shared.context import RequestContext
    from mcp.types import ElicitRequestParams, ElicitResult
    from pydantic import BaseModel
    from mcp.server.fastmcp import Context, FastMCP
except ImportError:
    RequestContext = ElicitRequestParams = ElicitResult = None
    BaseModel = object
    Context = FastMCP = None


# ─────────────────────────────────────────────────────────────────────────────
# SERVER SIDE — MCP server tool that uses elicitation
# Save this as profile_server.py and run separately.
# ─────────────────────────────────────────────────────────────────────────────

SERVER_CODE = '''
# profile_server.py
from pydantic import BaseModel
from mcp.server.fastmcp import Context, FastMCP

server = FastMCP("Profile")

class UserDetails(BaseModel):
    """Schema for the data the server needs from the client."""
    email: str
    age:   int

@server.tool()
async def create_profile(name: str, ctx: Context) -> str:
    """Create a user profile, asking for missing details via elicitation."""

    # Pause execution and ask the client for additional input
    result = await ctx.elicit(
        message=f"Please provide details for {name}\'s profile:",
        schema=UserDetails,       # tells client what fields are needed
    )

    if result.action == "accept" and result.data:
        return (
            f"Created profile for {name}: "
            f"email={result.data.email}, age={result.data.age}"
        )
    elif result.action == "decline":
        return f"User declined. Created minimal profile for {name} with no details."
    else:
        return "Profile creation cancelled by user."

if __name__ == "__main__":
    server.run(transport="http")
'''

print("=" * 60)
print("profile_server.py contents (run this separately):")
print("=" * 60)
print(SERVER_CODE)


# ─────────────────────────────────────────────────────────────────────────────
# CLIENT SIDE — handle elicitation requests
# ─────────────────────────────────────────────────────────────────────────────

async def on_elicitation_accept(
    mcp_context: "RequestContext",
    params:      "ElicitRequestParams",
    context:     CallbackContext,
) -> "ElicitResult":
    """
    Handle elicitation by auto-accepting with hardcoded data.
    In a real app: show a UI prompt to the user and return their input.

    params.message          → human-readable request from the server
    params.requestedSchema  → JSON schema of what data is needed
    context.server_name     → which server is asking
    context.tool_name       → which tool triggered the request
    """
    print(f"  [elicit] Server '{context.server_name}' requests: {params.message}")
    print(f"  [elicit] Schema: {params.requestedSchema}")

    # In production: prompt user in a UI, collect their input, return it
    # Here: auto-accept with mock data
    return ElicitResult(
        action="accept",
        content={
            "email": "user@example.com",
            "age":   25,
        },
    )


async def on_elicitation_decline(
    mcp_context: "RequestContext",
    params:      "ElicitRequestParams",
    context:     CallbackContext,
) -> "ElicitResult":
    """Handle elicitation by declining to provide information."""
    print(f"  [elicit] Declining request: {params.message}")
    return ElicitResult(action="decline")


async def on_elicitation_cancel(
    mcp_context: "RequestContext",
    params:      "ElicitRequestParams",
    context:     CallbackContext,
) -> "ElicitResult":
    """Handle elicitation by cancelling the operation."""
    print(f"  [elicit] Cancelling operation")
    return ElicitResult(action="cancel")


async def on_elicitation_conditional(
    mcp_context: "RequestContext",
    params:      "ElicitRequestParams",
    context:     CallbackContext,
) -> "ElicitResult":
    """
    Conditionally respond based on what the server is asking for.
    Real-world pattern: check what fields are needed, decide per-field.
    """
    schema = params.requestedSchema or {}
    props  = schema.get("properties", {})

    # Only provide data if the request seems safe
    if "password" in props or "ssn" in props:
        print(f"  [elicit] Refusing sensitive data request")
        return ElicitResult(action="decline")

    # Provide mock data for safe fields
    content = {}
    if "email" in props:
        content["email"] = "auto@example.com"
    if "age" in props:
        content["age"] = 30
    if "name" in props:
        content["name"] = "Auto User"

    print(f"  [elicit] Accepting with: {content}")
    return ElicitResult(action="accept", content=content)


# ─────────────────────────────────────────────────────────────────────────────
# FULL EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────

async def elicitation_accept_example():
    """Agent that accepts elicitation requests with provided data."""
    from langchain.agents import create_agent

    client = MultiServerMCPClient(
        {
            "profile": {
                "transport": "http",
                "url":       "http://localhost:8000/mcp",
            }
        },
        callbacks=Callbacks(on_elicitation=on_elicitation_accept),
    )

    tools  = await client.get_tools()
    agent  = create_agent("gpt-4.1", tools)

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Create a profile for Alice"}]
    })
    print(f"Accept result: {result['messages'][-1].content}")


async def elicitation_decline_example():
    """Agent that declines elicitation — server creates minimal profile."""
    from langchain.agents import create_agent

    client = MultiServerMCPClient(
        {
            "profile": {
                "transport": "http",
                "url":       "http://localhost:8000/mcp",
            }
        },
        callbacks=Callbacks(on_elicitation=on_elicitation_decline),
    )

    tools  = await client.get_tools()
    agent  = create_agent("gpt-4.1", tools)

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Create a profile for Bob"}]
    })
    print(f"Decline result: {result['messages'][-1].content}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("""
Elicitation Quick Reference:

  Server side (in your MCP server):
    result = await ctx.elicit(
        message="Please provide your email and age:",
        schema=MyPydanticModel,
    )
    if result.action == "accept":    # user provided data
        data = result.data           # instance of MyPydanticModel
    elif result.action == "decline": # user refused
        ...
    elif result.action == "cancel":  # user aborted
        ...

  Client side callback:
    async def on_elicitation(mcp_context, params, context) -> ElicitResult:
        # params.message         → what the server is asking for
        # params.requestedSchema → JSON schema of needed fields
        # context.server_name    → which server is asking
        # context.tool_name      → which tool triggered it

        return ElicitResult(action="accept",  content={"email": "x", "age": 25})
        return ElicitResult(action="decline")
        return ElicitResult(action="cancel")

  Attach to client:
    client = MultiServerMCPClient(
        {...},
        callbacks=Callbacks(on_elicitation=on_elicitation),
    )
""")

if __name__ == "__main__":
    asyncio.run(elicitation_accept_example())