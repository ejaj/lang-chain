from fastapi import FastAPI
from pydantic import BaseModel
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio

app = FastAPI()

# ── helper: call any MCP tool ─────────────────────────
async def call_mcp(tool_name: str, arguments: dict) -> str:
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"]       # path to your MCP server
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(tool_name, arguments)
            return result.content[0].text


# ── API endpoint 1: say hello ─────────────────────────
class HelloRequest(BaseModel):
    name: str

@app.post("/hello")
async def hello(req: HelloRequest):
    # calls MCP tool → say_hello
    result = await call_mcp("say_hello", {"name": req.name})
    return {"result": result}


# ── API endpoint 2: add numbers ───────────────────────
class AddRequest(BaseModel):
    a: float
    b: float

@app.post("/add")
async def add(req: AddRequest):
    # calls MCP tool → add_numbers
    result = await call_mcp("add_numbers", {"a": req.a, "b": req.b})
    return {"result": result}


# ── API endpoint 3: list all tools ───────────────────
@app.get("/tools")
async def list_tools():
    server_params = StdioServerParameters(
        command="python", args=["server.py"]
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)