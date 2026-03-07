# server.py

from mcp.server import Server           # the MCP server class
from mcp.server.stdio import stdio_server  # runs server on stdio
from mcp.types import Tool, TextContent, Resource    # Tool = tool definition, TextContent = result
import asyncio, json, sqlite3, httpx
import json
from pydantic import AnyUrl


# ─────────────────────────────────────────
# PART 1: Create the server
# ─────────────────────────────────────────
app = Server("my-server")
# "my-server" is just the name — like naming your app


# ─────────────────────────────────────────
# PART 2: Tell AI what tools exist
# ─────────────────────────────────────────
@app.list_tools()
async def list_tools() -> list[Tool]:
    # AI calls this ONCE when it connects
    # to know: "what can I use?"
    return [
        Tool(
            name="say_hello",           # AI will call it by this name
            description="Say hello to a person by name",  # AI reads this to decide when to use it
            inputSchema={               # what arguments the tool needs
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The person's name"
                    }
                },
                "required": ["name"]    # "name" is mandatory
            }
        ),
        Tool(
            name="add_numbers",
            description="Add two numbers together",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        )
    ]

@app.list_resources()
async def list_resources() -> list[Resource]:
    # tell AI what resources exist
    return [
        Resource(
            uri="data://customers/all",           # ← unique address
            name="All Customers",                 # ← human readable name
            description="List of all customers",  # ← AI reads this
            mimeType="application/json"           # ← type of data
        ),
        Resource(
            uri="data://customers/{customer_id}",
            name="Customer Detail",
            description="One specific customer by ID",
            mimeType="application/json"
        ),
        Resource(
            uri="data://faq",
            name="FAQ",
            description="Frequently asked questions",
            mimeType="application/json"
        ),
    ]
# ─────────────────────────────────────────
# PART 3: Run the tool when AI calls it
# ─────────────────────────────────────────
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # "name"      = which tool AI is calling  e.g. "say_hello"
    # "arguments" = what AI passed in         e.g. {"name": "Arjun"}

    if name == "say_hello":
        person = arguments["name"]
        result = f"Hello, {person}! Welcome!"
        return [TextContent(type="text", text=result)]
        #        ^^^^^^^^^^^ always wrap result in TextContent

    elif name == "add_numbers":
        a = arguments["a"]
        b = arguments["b"]
        result = a + b
        return [TextContent(type="text", text=str(result))]

    # always have a fallback
    return [TextContent(type="text", text=f"Unknown tool: {name}")]

@app.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    # uri = which resource AI is reading
    uri_str = str(uri)

    # ── resource 1: all customers (from DB) ──────────
    if uri_str == "data://customers/all":
        conn = sqlite3.connect("support.db")
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM customers").fetchall()
        conn.close()
        return json.dumps([dict(r) for r in rows])

    # ── resource 2: one customer (from DB) ───────────
    elif uri_str.startswith("data://customers/"):
        customer_id = uri_str.split("/")[-1]   # extract ID from URI
        conn = sqlite3.connect("support.db")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM customers WHERE id = ?", (customer_id,))
        row = c.fetchone()
        conn.close()
        return json.dumps(dict(row)) if row else json.dumps({"error": "not found"})

    # ── resource 3: faq (from external API) ──────────
    elif uri_str == "data://faq":
        async with httpx.AsyncClient() as client:
            r = await client.get("https://your-api.com/faq")
            return r.text

    return json.dumps({"error": f"Unknown resource: {uri_str}"})
# ─────────────────────────────────────────
# PART 4: Start the server
# ─────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(stdio_server(app))
    # stdio = server talks via stdin/stdout
    # this is how Claude Desktop communicates with it