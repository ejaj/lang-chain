from fastapi import FastAPI
from pydantic import BaseModel
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Literal

app = FastAPI()


# ── shared helper ─────────────────────────────────────
async def get_session():
    server_params = StdioServerParameters(
        command="python", args=["server/FastMCP.server.py"]
    )
    return server_params


# ════════════════════════════════════════
# TOOL endpoints
# ════════════════════════════════════════

class HelloRequest(BaseModel):
    name: str

@app.post("/tool/hello")
async def hello(req: HelloRequest):
    params = await get_session()
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            result = await s.call_tool("say_hello", {"name": req.name})
            return {"result": result.content[0].text}


class TicketRequest(BaseModel):
    customer_id: str
    issue: str
    priority: Literal["low", "medium", "high", "urgent"]

@app.post("/tool/ticket")
async def create_ticket(req: TicketRequest):
    params = await get_session()
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            result = await s.call_tool("create_ticket", {
                "customer_id": req.customer_id,
                "issue":       req.issue,
                "priority":    req.priority
            })
            return {"result": result.content[0].text}


# ════════════════════════════════════════
# RESOURCE endpoints
# ════════════════════════════════════════

@app.get("/resource/customers")
async def get_all_customers():
    params = await get_session()
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            # read static resource
            result = await s.read_resource("data://customers/all")
            return {"data": result.contents[0].text}


@app.get("/resource/customers/{customer_id}")
async def get_customer(customer_id: str):
    params = await get_session()
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            # read dynamic resource — pass customer_id in URL
            result = await s.read_resource(f"data://customers/{customer_id}")
            return {"data": result.contents[0].text}


@app.get("/resource/faq")
async def get_faq():
    params = await get_session()
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            result = await s.read_resource("data://faq")
            return {"data": result.contents[0].text}


# ════════════════════════════════════════
# PROMPT endpoints
# ════════════════════════════════════════

class SupportPromptRequest(BaseModel):
    customer_name: str
    issue: str

@app.post("/prompt/support")
async def get_support_prompt(req: SupportPromptRequest):
    params = await get_session()
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            # get prompt template with filled values
            result = await s.get_prompt("support_agent_prompt", {
                "customer_name": req.customer_name,
                "issue":         req.issue
            })
            return {"prompt": result.messages[0].content.text}


class EscalateRequest(BaseModel):
    ticket_id: str
    reason: str

@app.post("/prompt/escalate")
async def get_escalate_prompt(req: EscalateRequest):
    params = await get_session()
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            result = await s.get_prompt("escalation_prompt", {
                "ticket_id": req.ticket_id,
                "reason":    req.reason
            })
            return {"prompt": result.messages[0].content.text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)