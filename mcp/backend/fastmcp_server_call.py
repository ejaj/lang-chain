from fastapi import FastAPI
from pydantic import BaseModel
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Literal

app = FastAPI()


# ── helper: call any MCP tool ─────────────────────────
async def call_mcp(tool_name: str, arguments: dict) -> str:
    server_params = StdioServerParameters(
        command="python",
        args=["server/FastMCP.server.py"]
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return result.content[0].text


# ── endpoint 1: say hello ─────────────────────────────
class HelloRequest(BaseModel):
    name: str

@app.post("/hello")
async def hello(req: HelloRequest):
    result = await call_mcp("say_hello", {"name": req.name})
    return {"result": result}


# ── endpoint 2: add numbers ───────────────────────────
class AddRequest(BaseModel):
    a: float
    b: float

@app.post("/add")
async def add(req: AddRequest):
    result = await call_mcp("add_numbers", {"a": req.a, "b": req.b})
    return {"result": result}


# ── endpoint 3: create ticket ─────────────────────────
class TicketRequest(BaseModel):
    customer_id: str
    issue: str
    priority: Literal["low", "medium", "high", "urgent"]

@app.post("/ticket")
async def ticket(req: TicketRequest):
    result = await call_mcp("create_ticket", {
        "customer_id": req.customer_id,
        "issue":       req.issue,
        "priority":    req.priority
    })
    return {"result": result}


# ── endpoint 4: list all tools ────────────────────────
@app.get("/tools")
async def list_tools():
    server_params = StdioServerParameters(
        command="python", args=["server/FastMCP.server.py"]
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return {
                "tools": [
                    {"name": t.name, "description": t.description}
                    for t in tools.tools
                ]
            }

@app.get("/resource/customers")
async def get_all_customers():
    server_params = StdioServerParameters(
        command="python", args=["server.py"]
    )
    async with stdio_client(server_params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            result = await s.read_resource("data://customers/all")
            return {"data": result.contents[0].text}


@app.get("/resource/customers/{customer_id}")
async def get_customer(customer_id: str):
    server_params = StdioServerParameters(
        command="python", args=["server.py"]
    )
    async with stdio_client(server_params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            result = await s.read_resource(f"data://customers/{customer_id}")
            return {"data": result.contents[0].text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)